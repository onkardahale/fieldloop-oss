//! Generate the bundled demo recordings (synthetic, labeled as such).
//!
//! Writes TWO files:
//!   cargo run -p fieldloop-import --example gen_demo_mcap -- <demo.mcap> <demo-skew.mcap>
//!
//! `demo.mcap` is a single-boot, ~50 s timeline shaped like a real robot log — 10 Hz
//! decision bursts with millisecond jitter and three higher-rate sensor topics — with
//! four outcome beats that each show one engine behavior:
//!   1. an isolated decision, takeover 150 ms later  -> one clean high-confidence binding
//!   2. an isolated decision, e-stop 200 ms later    -> the 0.60 binding (200/500 ms)
//!   3. two near-simultaneous decisions, then e-stop -> credit split across both
//!   4. a collision 1 s after the last decision      -> correctly refuses to bind
//!      (the 150 ms collision window is long gone): fail-closed, not a guess
//!
//! `demo-skew.mcap` is a small companion whose `/range/front` topic publishes 32 s
//! behind its log time — a sensor on another time base — so `fieldloop doctor` has a
//! real catch to demonstrate.
//!
//! Kept as an example so the binaries are regenerable from source and provably real MCAP.

use std::collections::BTreeMap;

const MS: u64 = 1_000_000;
const S: u64 = 1_000_000_000;

struct Gen {
    writer: mcap::Writer<std::io::BufWriter<std::fs::File>>,
    channels: BTreeMap<String, u16>,
    seq: u32,
}

impl Gen {
    fn create(path: &str) -> Gen {
        let file = std::fs::File::create(path).expect("create output file");
        let writer = mcap::WriteOptions::new()
            .library("fieldloop demo generator (synthetic)")
            .create(std::io::BufWriter::new(file))
            .expect("create mcap writer");
        Gen {
            writer,
            channels: BTreeMap::new(),
            seq: 0,
        }
    }

    fn msg(&mut self, topic: &str, log_ns: u64, publish_ns: u64, data: &[u8]) {
        let id = match self.channels.get(topic) {
            Some(&id) => id,
            None => {
                let id = self
                    .writer
                    .add_channel(0, topic, "json", &BTreeMap::new())
                    .expect("add channel");
                self.channels.insert(topic.to_string(), id);
                id
            }
        };
        let header = mcap::records::MessageHeader {
            channel_id: id,
            sequence: self.seq,
            log_time: log_ns,
            publish_time: publish_ns,
        };
        self.seq += 1;
        self.writer
            .write_to_known_channel(&header, data)
            .expect("write message");
    }

    fn finish(mut self, path: &str) {
        self.writer.finish().expect("finish mcap");
        let buf = self.writer.into_inner();
        let file = buf.into_inner().expect("flush demo mcap");
        file.sync_all().expect("sync demo mcap");
        println!("wrote {path}");
    }
}

/// Deterministic per-index jitter in [-5, +5] ms — burst timestamps look sensor-real
/// without an RNG dependency or losing reproducibility.
fn jitter_ns(i: u64) -> i64 {
    ((i.wrapping_mul(2_654_435_761) >> 8) % 11) as i64 * (MS as i64) - 5 * (MS as i64)
}

fn decision(g: &mut Gen, t: u64) {
    g.msg("/policy/action", t, t, br#"{"action":"advance"}"#);
}

/// A 10 Hz decision burst over [start, end] (tenths of a second), with jitter.
fn burst(g: &mut Gen, start_s_x10: u64, end_s_x10: u64) {
    for i in start_s_x10..=end_s_x10 {
        let t = (i * S / 10).saturating_add_signed(jitter_ns(i));
        decision(g, t);
    }
}

/// Clean-clock distractor topics at sensor-like rates over [1 s, end].
fn distractors(g: &mut Gen, end_s: u64) {
    for i in 20..(end_s * 20) {
        let t = i * S / 20;
        g.msg(
            "/joint_states",
            t,
            t,
            br#"{"pos":[0.1,0.2,0.3,0.4,0.5,0.6]}"#,
        );
        g.msg("/tf", t, t, br#"{"frame":"base","xyz":[1.0,2.0,0.0]}"#);
    }
    for i in 5..(end_s * 5) {
        let t = i * S / 5;
        g.msg("/camera/info", t, t, br#"{"w":640,"h":480}"#);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let demo_path = args
        .next()
        .expect("usage: gen_demo_mcap <demo.mcap> <demo-skew.mcap>");
    let skew_path = args
        .next()
        .expect("usage: gen_demo_mcap <demo.mcap> <demo-skew.mcap>");

    // ---- demo.mcap: the showcase timeline -------------------------------------------
    let mut g = Gen::create(&demo_path);
    distractors(&mut g, 51);

    // Bursts end well before each beat so the beat's window holds exactly the intended
    // candidates (windows: takeover 1500 ms, e_stop 500 ms, collision 150 ms).
    burst(&mut g, 10, 329); // 1.0 s .. 32.9 s
    burst(&mut g, 360, 395); // 36.0 s .. 39.5 s
    burst(&mut g, 460, 490); // 46.0 s .. 49.0 s

    // Beat 1 — isolated decision, takeover 150 ms later: one clean binding (~0.90).
    decision(&mut g, 35_000 * MS);
    g.msg(
        "/teleop/takeover",
        35_150 * MS,
        35_150 * MS,
        br#"{"operator":"j.doe"}"#,
    );

    // Beat 2 — isolated decision, e-stop 200 ms later: the 0.60 binding (200/500 ms).
    decision(&mut g, 40_000 * MS);
    g.msg(
        "/safety/estop",
        40_200 * MS,
        40_200 * MS,
        br#"{"reason":"force_limit"}"#,
    );

    // Beat 3 — near-simultaneous decisions, e-stop after both: credit splits.
    decision(&mut g, 44_500 * MS); // outside the 500 ms window — not a candidate
    decision(&mut g, 45_000 * MS);
    decision(&mut g, 45_300 * MS);
    g.msg(
        "/safety/estop",
        45_400 * MS,
        45_400 * MS,
        br#"{"reason":"obstacle"}"#,
    );

    // Beat 4 — collision a full second after the last decision: the 150 ms window is
    // long past, so the engine refuses to bind rather than guess.
    g.msg(
        "/collision/event",
        50_000 * MS,
        50_000 * MS,
        br#"{"surface":"cart"}"#,
    );

    g.finish(&demo_path);

    // ---- demo-skew.mcap: the doctor's catch ------------------------------------------
    // Timeline starts at 40 s so `log_time - 32 s` stays in range (u64).
    let mut s = Gen::create(&skew_path);
    for i in 400..460 {
        let t = i * S / 10;
        decision(&mut s, t);
        // A sensor on another time base: publishes 32 s behind its log time.
        s.msg("/range/front", t, t - 32 * S, br#"{"range_m":1.4}"#);
    }
    s.msg(
        "/safety/estop",
        46_200 * MS,
        46_200 * MS,
        br#"{"reason":"demo"}"#,
    );
    s.finish(&skew_path);
}
