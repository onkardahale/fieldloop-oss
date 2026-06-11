//! Assembling the metadata batch the agent POSTs to the cloud ingest gateway.
//!
//! The agent ships opaque recorded files by object key, but the gateway's
//! `POST /v1/ingest` wants a *manifest* (`tenant_id`, `robot_id`, `batch_id`) plus the
//! rollout rows plus a list of blob claims (object key + checksum + size) it can verify
//! against the bucket. Nothing else in the system owns that translation: capture mints
//! rollouts with empty payload pointers, the uploader moves bytes by key, and the
//! gateway needs the two stitched together. This module is that stitch.
//!
//! It is shipped production code, not a test fixture, because the real on-robot agent
//! must build exactly this request body to talk to the real gateway — a test that built
//! the JSON by hand would prove nothing about whether the agent and gateway agree on the
//! wire shape. Building it here, once, means the same code the e2e test drives is the
//! code that runs on a robot.
//!
//! The whole module is dependency-free beyond `serde_json` + the canonical schema types,
//! so it sits in the agent's default build with no optional feature: assembling a request
//! body needs no network and no cloud SDK.

use fieldloop_types::Rollout;
use serde_json::Value;

/// One blob the agent uploaded, described the way the gateway needs to hear about it:
/// the object key it landed at, plus the checksum and byte size of the exact bytes that
/// were delivered.
///
/// The checksum and size are the *claim* the gateway re-verifies against the bucket, so
/// they must be computed from the real uploaded bytes (not asserted by hand) — that is
/// what makes the upload-then-verify chain a genuine end-to-end check rather than two
/// numbers that happen to agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadedBlob {
    /// The object key the file was uploaded under (its recorder filename).
    pub object_key: String,
    /// SHA-256 of the uploaded bytes, lowercase hex.
    pub content_sha256: String,
    /// Byte length of the uploaded bytes.
    pub size: u64,
}

/// Build the gateway's `POST /v1/ingest` request body from the drained rollouts and the
/// blobs the agent uploaded.
///
/// The body has three parts and each exists for a reason the gateway enforces:
/// 1. the **manifest** `{tenant_id, robot_id, batch_id}` — `tenant_id`/`robot_id` must
///    match what the auth token resolves to (the gateway 403s a mismatch, so a robot can
///    only ingest as itself), and `batch_id` is the idempotency key a re-delivered batch
///    reuses;
/// 2. each rollout with the recorded MCAP object attached as its `observation_ref`
///    (object key + verified checksum) AND that rollout's OWN message range preserved, so
///    the gateway's per-rollout pointer check runs against a real object and a consumer
///    can fetch exactly that one rollout's frames. Capture leaves the object key empty
///    (it is assigned cloud-side, which is the seam this fills) but DOES stamp each
///    rollout's own range when its frames were recorded; this carries that range through
///    rather than discarding it. A rollout with no recorded range references the whole
///    object (range null), which is correct — there is nothing finer to point at;
/// 3. every uploaded blob as a top-level claim carrying the checksum and size, which the
///    gateway HEADs against the bucket before recording anything.
///
/// Why the range must be per-rollout and not shared: attaching one blob pointer with a
/// null range to every rollout would make every decision point at the entire recording,
/// so a replay viewer could never isolate a single decision's frames. Preserving each
/// rollout's own range is the correspondence that makes a per-rollout fetch possible.
///
/// `tenant_id`/`robot_id` are taken as strings so the caller passes the same identity the
/// token verifier will resolve the bearer token to; the JSON is returned as a
/// `serde_json::Value` so the real wire-decode path the gateway uses is exercised, rather
/// than a private request struct being constructed directly.
#[must_use]
pub fn build_ingest_request_json(
    tenant_id: &str,
    robot_id: &str,
    batch_id: &str,
    rollouts: &[Rollout],
    blobs: &[UploadedBlob],
) -> Value {
    // The recorded MCAP object the rollouts' frames live in. Capture cannot name it (the
    // object key is assigned cloud-side), so the agent supplies it here from the blob it
    // actually uploaded. The range is each rollout's OWN — taken from the rollout, never
    // overwritten — so two rollouts in the same object get two different pointers.
    //
    // Select the SENSOR-class blob, not `blobs.first()`: an observation is sensor-stream
    // data, but `meta-0.mcap` sorts before `sensor-0.mcap`, so first() would point every
    // observation at the metadata file (a corrupt pointer — the range indexes the sensor
    // stream). With a single sensor file (the common case) this is exact. Pinning WHICH
    // sensor file holds a rollout's range when several exist (rotation) needs per-file
    // index coverage the recorder does not yet emit — the scoped coverage-sidecar
    // follow-up; until then a multi-sensor-file batch points at the first sensor file.
    let object = blobs
        .iter()
        .find(|b| {
            crate::StreamClass::from_filename(&b.object_key) == Some(crate::StreamClass::Sensor)
        })
        .or_else(|| blobs.first());
    let rollout_values: Vec<Value> = rollouts
        .iter()
        .map(|rollout| {
            let mut value =
                serde_json::to_value(rollout).expect("a Rollout must serialize to JSON");
            if let Some(blob) = object {
                // Preserve THIS rollout's own message range; serialize it back to the same
                // {start, end} wire shape ByteRange uses. A None range (no recorded frame)
                // becomes null, meaning "the whole object", which is the honest pointer
                // when there is nothing finer to reference.
                let range = match &rollout.observation_ref.range {
                    Some(r) => serde_json::json!({ "start": r.start, "end": r.end }),
                    None => Value::Null,
                };
                value["observation_ref"] = serde_json::json!({
                    "object_key": blob.object_key,
                    "range": range,
                    "content_sha256": blob.content_sha256,
                });
            }
            value
        })
        .collect();

    let blob_claims: Vec<Value> = blobs
        .iter()
        .map(|blob| {
            serde_json::json!({
                "object_key": blob.object_key,
                "content_sha256": blob.content_sha256,
                "size": blob.size,
            })
        })
        .collect();

    serde_json::json!({
        "manifest": {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "batch_id": batch_id,
        },
        "rollouts": rollout_values,
        "blobs": blob_claims,
    })
}

/// SHA-256 of `data` as lowercase hex.
///
/// A small, dependency-free implementation of FIPS 180-4. It is here so the agent can
/// compute the content hash of the bytes it uploads — the exact value the gateway will
/// re-verify against the bucket — without pulling a hashing crate into the audited
/// dependency surface, and so the upload side and any verify side call one function and
/// can never disagree on what "the checksum of these bytes" means.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    // FIPS 180-4 round constants: the first 32 bits of the fractional parts of the cube
    // roots of the first 64 primes.
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    // Initial hash values: the first 32 bits of the fractional parts of the square roots
    // of the first 8 primes.
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    // Pre-process: append the bit '1', then '0' bits until the length is 448 mod 512, then
    // the original message length in bits as a 64-bit big-endian integer.
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) chunk.
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let j = i * 4;
            *word = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4 / NIST published vectors, so the local implementation is the real
    /// SHA-256 and not merely a self-consistent hash.
    #[test]
    fn known_sha256_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"hello world"),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    /// The assembled request carries the manifest identity, one rollout per input, and
    /// one blob claim per uploaded blob — the shape the gateway deserializes.
    #[test]
    fn request_json_has_manifest_rollouts_and_blob_claims() {
        use fieldloop_types::{
            BootId, BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, RobotId,
            RobotIdentity, Rollout, TenantId,
        };
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 1,
            ts_wall_ns: 1,
        };
        let rollout = Rollout::new(
            robot,
            EpisodeId::new(),
            0,
            clock,
            PolicyVersion::new("pol@v1+abc123def456"),
            "sha256:w".into(),
            "arm6dof".into(),
            "pick".into(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            1,
        );
        let blobs = vec![UploadedBlob {
            object_key: "sensor-0.mcap".into(),
            content_sha256: sha256_hex(b"x"),
            size: 1,
        }];
        let v = build_ingest_request_json("acme", "r1", "batch-1", &[rollout], &blobs);
        assert_eq!(v["manifest"]["tenant_id"], serde_json::json!("acme"));
        assert_eq!(v["manifest"]["robot_id"], serde_json::json!("r1"));
        assert_eq!(v["manifest"]["batch_id"], serde_json::json!("batch-1"));
        assert_eq!(v["rollouts"].as_array().unwrap().len(), 1);
        assert_eq!(v["blobs"].as_array().unwrap().len(), 1);
        // The blob is attached to the rollout's observation_ref so the gateway's
        // per-rollout pointer check has a real object to verify.
        assert_eq!(
            v["rollouts"][0]["observation_ref"]["object_key"],
            serde_json::json!("sensor-0.mcap")
        );
    }

    /// Build a rollout whose observation_ref carries `range` (the per-rollout message
    /// range capture stamps on), with an empty object key (assigned cloud-side) — exactly
    /// the shape the agent reads back from the drained JSONL.
    fn rollout_with_range(step: u32, range: Option<fieldloop_types::ByteRange>) -> Rollout {
        use fieldloop_types::{
            BootId, BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, RobotId,
            RobotIdentity, TenantId,
        };
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 1,
            ts_wall_ns: 1,
        };
        Rollout::new(
            robot,
            EpisodeId::new(),
            step,
            clock,
            PolicyVersion::new("pol@v1+abc123def456"),
            "sha256:w".into(),
            "arm6dof".into(),
            "pick".into(),
            // The observation pointer carries this rollout's OWN range; key empty.
            PayloadRef {
                object_key: String::new(),
                range,
                content_sha256: None,
            },
            PayloadRef::none(),
            BoundedBlob::empty(),
            1,
        )
    }

    /// The audit regression, asserted directly: two rollouts recorded into the same MCAP
    /// object must end up with DIFFERENT observation_ref ranges in the request body — not
    /// the same shared blob pointer. Both point at the same object key (it is one file),
    /// but each carries its own [start, end) so a replay viewer can isolate one rollout.
    #[test]
    fn each_rollout_gets_its_own_range_not_a_shared_blob_pointer() {
        use fieldloop_types::ByteRange;
        // Rollout 0's frames are messages [0, 2); rollout 1's are [2, 5).
        let r0 = rollout_with_range(0, Some(ByteRange { start: 0, end: 2 }));
        let r1 = rollout_with_range(1, Some(ByteRange { start: 2, end: 5 }));
        let blobs = vec![UploadedBlob {
            object_key: "sensor-0.mcap".into(),
            content_sha256: sha256_hex(b"bytes"),
            size: 5,
        }];

        let v = build_ingest_request_json("acme", "r1", "batch-1", &[r0, r1], &blobs);

        let obs0 = &v["rollouts"][0]["observation_ref"];
        let obs1 = &v["rollouts"][1]["observation_ref"];
        // Same object (one recorded file), with the verified checksum on each.
        assert_eq!(obs0["object_key"], serde_json::json!("sensor-0.mcap"));
        assert_eq!(obs1["object_key"], serde_json::json!("sensor-0.mcap"));
        assert_eq!(
            obs0["content_sha256"],
            serde_json::json!(sha256_hex(b"bytes"))
        );
        // Each rollout's OWN range, carried through, not a shared null.
        assert_eq!(obs0["range"], serde_json::json!({ "start": 0, "end": 2 }));
        assert_eq!(obs1["range"], serde_json::json!({ "start": 2, "end": 5 }));
        // The whole point of the fix: the two pointers differ in their range.
        assert_ne!(
            obs0["range"], obs1["range"],
            "two rollouts must not share one blob range"
        );
    }

    /// The observation pointer names the SENSOR-class object even when a metadata blob
    /// sorts ahead of it — the regression where `blobs.first()` pointed every observation
    /// at `meta-0.mcap` (its range indexes the sensor stream, so a meta-file pointer is
    /// corrupt).
    #[test]
    fn observation_points_at_sensor_blob_not_the_first_sorted() {
        use fieldloop_types::ByteRange;
        let r = rollout_with_range(0, Some(ByteRange { start: 0, end: 2 }));
        // done_keys yields keys in lexicographic order: meta-0 precedes sensor-0.
        let blobs = vec![
            UploadedBlob {
                object_key: "meta-0.mcap".into(),
                content_sha256: sha256_hex(b"meta"),
                size: 4,
            },
            UploadedBlob {
                object_key: "sensor-0.mcap".into(),
                content_sha256: sha256_hex(b"sensor"),
                size: 6,
            },
        ];
        let v = build_ingest_request_json("acme", "r1", "batch-1", &[r], &blobs);
        assert_eq!(
            v["rollouts"][0]["observation_ref"]["object_key"],
            serde_json::json!("sensor-0.mcap"),
            "an observation must point at the sensor object, not the meta file that sorts first"
        );
    }

    /// The single-rollout case still works: one rollout with one range yields one pointer
    /// carrying that exact range against the object key.
    #[test]
    fn single_rollout_carries_its_one_range() {
        use fieldloop_types::ByteRange;
        let r = rollout_with_range(0, Some(ByteRange { start: 7, end: 8 }));
        let blobs = vec![UploadedBlob {
            object_key: "sensor-0.mcap".into(),
            content_sha256: sha256_hex(b"z"),
            size: 1,
        }];
        let v = build_ingest_request_json("acme", "r1", "batch-1", &[r], &blobs);
        assert_eq!(v["rollouts"].as_array().unwrap().len(), 1);
        let obs = &v["rollouts"][0]["observation_ref"];
        assert_eq!(obs["object_key"], serde_json::json!("sensor-0.mcap"));
        assert_eq!(obs["range"], serde_json::json!({ "start": 7, "end": 8 }));
    }

    /// A rollout with no recorded range references the whole object (range null), which is
    /// the honest pointer when there is nothing finer to point at — not a fabricated range.
    #[test]
    fn rollout_without_a_range_points_at_the_whole_object() {
        let r = rollout_with_range(0, None);
        let blobs = vec![UploadedBlob {
            object_key: "sensor-0.mcap".into(),
            content_sha256: sha256_hex(b"z"),
            size: 1,
        }];
        let v = build_ingest_request_json("acme", "r1", "batch-1", &[r], &blobs);
        let obs = &v["rollouts"][0]["observation_ref"];
        assert_eq!(obs["object_key"], serde_json::json!("sensor-0.mcap"));
        assert_eq!(obs["range"], Value::Null);
    }
}
