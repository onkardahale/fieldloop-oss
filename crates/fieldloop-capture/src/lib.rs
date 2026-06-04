//! On-robot capture SDK core — the in-control-loop hot path.
//!
//! The robot's control loop runs at ~50Hz and must never block on Fieldloop: a
//! stall here would stall the robot. So the in-loop call does the minimum possible
//! work — mint a [`fieldloop_types::RolloutId`], read a monotonic clock, build a
//! small `Copy` record, and hand it off to a bounded in-memory queue without
//! blocking. If the queue is full the record is dropped and counted rather than
//! waited on, so a slow consumer can never back-pressure the control loop. All the
//! heavier work (resolving strings, assembling full [`fieldloop_types::Rollout`]s)
//! happens off the hot path in the [`Drain`].
//!
//! Two halves share state via [`std::sync::Arc`]:
//! * [`Capture`] — lives on the control-loop thread; exposes the hot-path
//!   [`Capture::log_step`].
//! * [`Drain`] — lives on a separate worker thread; pulls compact records and
//!   builds full rollouts.

mod clock;

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};

use fieldloop_types::{
    BootId, BoundedBlob, ByteRange, EpisodeId, MonoClock, PayloadRef, PolicyVersion, RobotIdentity,
    Rollout, RolloutId, SchemaConformance,
};

pub use clock::{ClockSource, FakeClock, SystemClock};

/// A `Copy` handle to a slowly-changing context registered once, out of the loop.
///
/// The fields that rarely change between steps — policy version, model hash,
/// embodiment, task — are `String`s, and constructing them on the hot path would
/// allocate. Instead they are registered once via [`Capture::register_context`],
/// which returns this small integer handle. The hot path only copies the handle
/// onto the record; the [`Drain`] resolves it back to the strings off the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextHandle(u32);

/// The registered, slowly-changing context behind a [`ContextHandle`].
///
/// Held only in the shared registry and read off the hot path by the drain, so its
/// `String` fields never have to be touched inside the control loop.
#[derive(Debug, Clone)]
struct Context {
    policy_version: PolicyVersion,
    model_hash: String,
    embodiment: String,
    task_id: String,
}

/// The compact, `Copy`, heap-free record built on the hot path.
///
/// Every field is `Copy`, so building one is a few register moves with no
/// allocation — exactly what the control loop can afford. The slowly-changing
/// string context is referenced indirectly through [`ContextHandle`] rather than
/// inlined, keeping `String`s out of the loop entirely. The drain expands this into
/// a full [`fieldloop_types::Rollout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRecord {
    /// The id minted for this step (also returned to the caller so the robot can
    /// thread it onto its outbound action message).
    pub rollout_id: RolloutId,
    /// The trajectory this step belongs to.
    pub episode_id: EpisodeId,
    /// Step ordinal within the episode.
    pub step_index: u32,
    /// Handle to the registered slowly-changing context (policy/model/embodiment/task).
    pub ctx: ContextHandle,
    /// Monotonic nanoseconds at the step — the attribution authority within a boot.
    pub mono_ns: u64,
    /// Advisory wall-clock estimate (nanoseconds since the Unix epoch). Never an
    /// ordering input on its own.
    pub wall_ns: i64,
    /// Inference wall-time in microseconds (the SDK's own ops measurement).
    pub inference_us: u32,
    /// Where THIS step's observation frame(s) live within the recorded MCAP object, as a
    /// half-open `[start, end)` run of message indices in the recorder's class stream.
    ///
    /// `None` when the caller did not record a frame for this step (e.g. an action-only
    /// step, or a step whose frame was dropped under recorder backpressure). When set,
    /// the drain stamps it onto the assembled rollout's `observation_ref.range` so each
    /// rollout points at its OWN messages in the recording — letting a replay viewer
    /// fetch exactly one decision's frames instead of the whole blob. It is two `u64`s,
    /// so `StepRecord` stays `Copy` and the hot path stays allocation-free. The object
    /// key is left empty here and filled cloud-side (the robot does not choose where
    /// bytes land); only the range — which is a fact about the on-robot recording — is
    /// known and carried here.
    pub observation_range: Option<PayloadMessageRange>,
}

/// A half-open `[start, end)` run of recorder message indices locating one step's frames
/// within the recorded MCAP object.
///
/// A small `Copy` pair so it can ride on the `Copy` [`StepRecord`] without allocating on
/// the hot path. It mirrors the recorder's per-message range, but is named in capture's
/// own vocabulary so this crate needs no dependency on the recorder crate — the caller
/// translates the recorder's returned index range into these two integers. `start..end`
/// indexes the class stream in write order; a single frame is `start = N, end = N + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadMessageRange {
    /// First message index of this step's frames (inclusive), in class-stream write order.
    pub start: u64,
    /// One past the last message index of this step's frames (exclusive).
    pub end: u64,
}

/// State shared between the [`Capture`] front half and the [`Drain`] back half.
///
/// Holds the boot id minted once at construction, the robot identity, the registry
/// of slowly-changing contexts, the drained-record receiver, and the dropped
/// counter. Both halves hold an `Arc` to this so they see the same registry and
/// counter.
struct Shared {
    /// Minted once at construction. Every rollout from this `Capture` carries it, so
    /// the drain can stamp the same boot session onto each assembled clock — and
    /// monotonic comparisons are only valid within one boot id.
    boot_id: BootId,
    /// The `(tenant_id, robot_id)` this capture instance belongs to. Cloned into
    /// each assembled rollout off the hot path.
    robot: RobotIdentity,
    /// Registry of slowly-changing contexts, indexed by the handle's integer.
    ///
    /// Guarded by a `Mutex` because registration may happen from a setup thread, but
    /// the hot path NEVER touches this lock: `log_step` only carries an already
    /// minted handle. The drain takes the lock briefly, off the loop, to resolve a
    /// handle back to its strings.
    contexts: Mutex<Vec<Context>>,
    /// Count of records dropped because the bounded queue was full. Atomic so the
    /// hot path can bump it without a lock.
    dropped: AtomicU64,
}

/// The hot-path front half, held on the control-loop thread.
///
/// Cheap to call: [`Capture::log_step`] mints an id, reads the clock, builds a
/// [`StepRecord`], and tries to enqueue it without blocking. Construction returns a
/// `Capture` paired with its [`Drain`]; both share the same queue and registry.
pub struct Capture {
    shared: Arc<Shared>,
    /// The bounded sender. `try_send` is non-blocking: it returns immediately,
    /// either enqueueing into the preallocated buffer or reporting "full" so the
    /// caller can drop-and-count instead of waiting.
    tx: SyncSender<StepRecord>,
    /// The clock read on every step. Boxed behind the trait so tests can inject a
    /// deterministic source.
    clock: Box<dyn ClockSource>,
}

/// The off-hot-path back half, held on a worker thread.
///
/// Pulls compact [`StepRecord`]s out of the bounded queue and expands each into a
/// full [`fieldloop_types::Rollout`]. All the work that the hot path deliberately
/// avoided — resolving the context handle to its strings, cloning the robot
/// identity, building the clock stamp — happens here, away from the control loop.
pub struct Drain {
    shared: Arc<Shared>,
    /// The bounded receiver. `try_recv` pops whatever is currently buffered without
    /// blocking, so the drain can poll on its own schedule.
    rx: Receiver<StepRecord>,
    /// The optional durable rollout-metadata sink. `None` by default (off unless a
    /// caller opts in via [`Drain::with_rollout_sink`]), so the existing in-memory
    /// drain path is unchanged for callers that do not want on-disk metadata. When
    /// `Some`, every rollout assembled by [`Drain::drain_available`] is also appended,
    /// as one canonical-JSON line, to the sink. The sink lives on the drain side only,
    /// so this never adds work to the 50Hz hot path.
    ///
    /// Wrapped in a `Mutex` so the file write goes through a shared `&self` drain (the
    /// existing `drain_available(&self)` signature, which other crates already call
    /// through a shared reference): the lock is taken only on the drain side, off the
    /// control loop, never by the hot-path `log_step`.
    rollout_sink: Option<Mutex<RolloutSink>>,
}

/// An append-only JSONL sink for drained [`fieldloop_types::Rollout`] metadata, the
/// durable producer the deployed agent binary consumes.
///
/// The agent reads its rollouts file by splitting on newlines and parsing each
/// non-empty line with `serde_json::from_str::<Rollout>`. To be a real producer for
/// that reader, this sink writes exactly that shape: one `serde_json::to_string`
/// (canonical, compact, no embedded newline) per rollout, followed by a single `\n`.
/// The round-trip `to_string` → read line → `from_str` is therefore identity, which is
/// the contract the agent depends on.
///
/// Durability/crash-safety choices, matching the recorder's "finalize what you wrote"
/// style:
/// * The file is opened in **append** mode, so reopening after a crash or restart adds
///   to the existing lines instead of truncating them — prior rollouts are never
///   clobbered.
/// * Each rollout is serialized into a `String` first and written with a **single
///   `write_all`** of `line + "\n"`, so a line is handed to the OS as one buffer; we
///   never emit a half-encoded record from an encoder error mid-write.
/// * After each line we **`flush` then `sync_data`** the file, so an `Ok` return means
///   the bytes (and the length metadata needed to read them back) have reached the
///   storage layer, not just a process buffer. A crash after a returned `Ok` leaves
///   complete, parseable lines; a crash mid-call at worst loses the in-flight line and
///   never corrupts an already-flushed one.
pub struct RolloutSink {
    /// Buffered handle to the append-mode file. Buffering coalesces the small writes of
    /// a single line; we still `flush` + `sync_data` per rollout so a returned `Ok`
    /// means the line is durable.
    writer: BufWriter<File>,
    /// The path opened, kept for error messages so a failure names the file it was
    /// writing.
    path: PathBuf,
}

impl RolloutSink {
    /// Open (creating if absent) the JSONL file at `path` in append mode.
    ///
    /// Append mode is the crash-safety property: an existing file is never truncated, so
    /// reopening after a restart continues the log rather than clobbering already-written
    /// rollouts. Any parent directory must already exist (the caller owns directory
    /// layout), so a missing directory surfaces here as an error rather than being
    /// silently created.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<RolloutSink> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(RolloutSink {
            writer: BufWriter::new(file),
            path,
        })
    }

    /// Append one rollout as a single canonical-JSON line terminated by `\n`.
    ///
    /// The serialization is `serde_json::to_string`, the identical form the agent parses
    /// with `from_str::<Rollout>`, so the written line round-trips back to an equal
    /// rollout. We serialize fully into a `String` before writing so an encoder error
    /// aborts before any bytes hit the file (no half-line), then `flush` + `sync_data`
    /// so the returned `Ok` means the line is durable on disk.
    pub fn append(&mut self, rollout: &Rollout) -> std::io::Result<()> {
        // serde_json::to_string never emits a literal newline inside the JSON, so the
        // record stays exactly one line — which is what makes line-splitting on the read
        // side correct.
        let line = serde_json::to_string(rollout).map_err(std::io::Error::other)?;
        // One write_all of the whole line + terminator: the OS sees a complete record,
        // never a partially-encoded one.
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        // Push the buffered bytes to the file, then sync data so a crash after this
        // returns cannot lose the line we just reported as written.
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    /// The path this sink writes to, for diagnostics.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Capture {
    /// Build a capture/drain pair using the real system clock.
    ///
    /// `capacity` is the bounded queue depth: at most this many un-drained records
    /// are buffered before further records are dropped-and-counted. A larger
    /// capacity tolerates longer drain stalls at the cost of more preallocated
    /// memory. The buffer is allocated up front here, so a later `try_send` on the
    /// hot path performs no heap allocation.
    #[must_use]
    pub fn new(robot: RobotIdentity, capacity: usize) -> (Capture, Drain) {
        Self::with_clock(robot, capacity, Box::new(SystemClock::new()))
    }

    /// Build a capture/drain pair with an injected clock source.
    ///
    /// Lets tests substitute a deterministic [`FakeClock`] for the real system
    /// clock so clock-dependent behavior is reproducible.
    #[must_use]
    pub fn with_clock(
        robot: RobotIdentity,
        capacity: usize,
        clock: Box<dyn ClockSource>,
    ) -> (Capture, Drain) {
        // Preallocate the bounded buffer now. sync_channel's buffer is fixed-size,
        // so a hot-path try_send into it never allocates and never blocks.
        let (tx, rx) = sync_channel::<StepRecord>(capacity);
        let shared = Arc::new(Shared {
            // Take the boot session from the clock, not a fresh per-instance id. The
            // clock sources it from the OS boot session, so two Captures built in one
            // boot stamp the SAME boot id and only a real reboot (a new OS boot session)
            // changes it — which is exactly the boundary across which monotonic readings
            // stop being comparable. A fresh-per-instance id would instead make a mere
            // process restart look like a reboot.
            boot_id: clock.boot_id(),
            robot,
            contexts: Mutex::new(Vec::new()),
            dropped: AtomicU64::new(0),
        });
        let capture = Capture {
            shared: Arc::clone(&shared),
            tx,
            clock,
        };
        let drain = Drain {
            shared,
            rx,
            // Off by default: the durable JSONL sink is opt-in via
            // Drain::with_rollout_sink, so existing callers keep the in-memory-only path.
            rollout_sink: None,
        };
        (capture, drain)
    }

    /// Register a slowly-changing context once, off the hot path, and get a `Copy`
    /// handle to reference it in-loop.
    ///
    /// This takes the registry lock and may allocate (it stores `String`s), which is
    /// why it is a setup-time call, NOT part of `log_step`. The returned
    /// [`ContextHandle`] is a plain integer that the hot path can copy for free.
    pub fn register_context(
        &self,
        policy_version: PolicyVersion,
        model_hash: impl Into<String>,
        embodiment: impl Into<String>,
        task_id: impl Into<String>,
    ) -> ContextHandle {
        let ctx = Context {
            policy_version,
            model_hash: model_hash.into(),
            embodiment: embodiment.into(),
            task_id: task_id.into(),
        };
        // Lock held only here, off the control loop. The hot path never waits on it.
        let mut contexts = self
            .shared
            .contexts
            .lock()
            .expect("context registry mutex poisoned");
        let index = contexts.len();
        contexts.push(ctx);
        // The handle is just the slot index. The Vec only grows, so an index stays
        // valid for the life of the Capture.
        ContextHandle(u32::try_from(index).expect("too many registered contexts for a u32 handle"))
    }

    /// The hot-path call, invoked from the control-loop thread on every inference
    /// step. Non-blocking, allocation-free, and never panics on a full queue.
    ///
    /// It mints a fresh [`RolloutId`], reads the clock, builds a `Copy`
    /// [`StepRecord`], and tries to enqueue it. If the bounded queue is full the
    /// record is dropped and the dropped counter is incremented — but the freshly
    /// minted `RolloutId` is STILL returned. The robot may already be threading that
    /// id onto its outbound action message, so losing the capture record must never
    /// lose the id the caller depends on.
    pub fn log_step(
        &self,
        episode_id: EpisodeId,
        step_index: u32,
        ctx: ContextHandle,
        inference_us: u32,
    ) -> RolloutId {
        // No recorded-frame range supplied: the rollout's observation_ref carries no
        // per-step range. Callers that record the step's frames to the recorder use
        // log_step_with_observation_range so each rollout points at its own messages.
        self.log_step_with_observation_range(episode_id, step_index, ctx, inference_us, None)
    }

    /// Like [`Capture::log_step`], but carries the message range of the observation
    /// frame(s) the caller recorded for THIS step, so the assembled rollout points at its
    /// OWN slice of the recorded MCAP object rather than the whole blob.
    ///
    /// The caller obtains `observation_range` from the recorder: recording a step's
    /// frame(s) returns the message index each lands at, and `[first, last + 1)` is this
    /// step's run. Passing it here threads that locator onto the rollout (the drain
    /// stamps it onto `observation_ref.range`), which is what lets a replay viewer fetch
    /// exactly one decision's frames. It is still non-blocking and allocation-free: the
    /// range is two `Copy` integers and goes onto the same `Copy` [`StepRecord`]; a full
    /// queue still drops-and-counts and still returns the freshly minted id.
    pub fn log_step_with_observation_range(
        &self,
        episode_id: EpisodeId,
        step_index: u32,
        ctx: ContextHandle,
        inference_us: u32,
        observation_range: Option<PayloadMessageRange>,
    ) -> RolloutId {
        let rollout_id = RolloutId::new();
        let (mono_ns, wall_ns) = self.clock.read();
        let record = StepRecord {
            rollout_id,
            episode_id,
            step_index,
            ctx,
            mono_ns,
            wall_ns,
            inference_us,
            observation_range,
        };
        // try_send never blocks: it either places the record into the preallocated
        // buffer or returns Full/Disconnected. On either error we count a drop and
        // move on — the control loop must not stall waiting for a slow drain, and a
        // dropped capture record is recoverable while a stalled robot is not.
        match self.tx.try_send(record) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        rollout_id
    }

    /// Number of records dropped so far because the bounded queue was full (or the
    /// drain was gone). Ops telemetry: a rising count means the drain is not keeping
    /// up with the loop.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }
}

impl Drain {
    /// Attach a durable JSONL [`RolloutSink`] at `path`, turning this drain into a real
    /// on-disk producer for the deployed agent binary (which reads the same file as
    /// one canonical-JSON [`fieldloop_types::Rollout`] per line).
    ///
    /// Opt-in by construction: a `Drain` built by [`Capture::new`] /
    /// [`Capture::with_clock`] has no sink, so the default behavior is unchanged and
    /// nothing is written to disk unless a caller calls this. After this returns,
    /// every rollout produced by [`Drain::drain_available`] is also appended to the
    /// file. The sink only participates on the drain side, so enabling it adds no work
    /// to the 50Hz hot path.
    ///
    /// # Errors
    /// Returns the I/O error from opening the file (e.g. a missing parent directory or
    /// a permission failure) so a misconfigured sink path fails loudly at setup rather
    /// than silently dropping rollouts later.
    pub fn with_rollout_sink(mut self, path: impl AsRef<Path>) -> std::io::Result<Drain> {
        self.rollout_sink = Some(Mutex::new(RolloutSink::open(path)?));
        Ok(self)
    }

    /// Pop every currently-available compact record and assemble a full
    /// [`fieldloop_types::Rollout`] for each.
    ///
    /// Non-blocking: it drains whatever is buffered right now and returns. This is
    /// the off-hot-path stage that does the work the control loop avoided —
    /// resolving each [`ContextHandle`] to its registered strings, cloning the robot
    /// identity, and building the [`MonoClock`] stamp from
    /// `(boot_id, mono_ns, wall_ns)`.
    ///
    /// If a [`RolloutSink`] is configured (via [`Drain::with_rollout_sink`]), each
    /// assembled rollout is also appended to it as one canonical-JSON line before
    /// being returned, so the on-disk JSONL matches the returned batch exactly. This
    /// append happens here, on the drain side, never on the hot path.
    ///
    /// The gateway/recorder-set fields are left at their pre-ingest defaults here and
    /// are filled by later stages, not in the loop: the action payload reference is empty
    /// ([`PayloadRef::none`]), the inline context is empty ([`BoundedBlob::empty`]), and
    /// `server_anchor`/`trust` stay `None` with schema conformance `Unchecked` until the
    /// ingest gateway reconciles and validates the row. The one exception is the
    /// observation pointer's range: when the caller recorded this step's frame(s) and
    /// passed the locator to [`Capture::log_step_with_observation_range`], the assembled
    /// rollout's `observation_ref.range` carries this step's OWN message range (its
    /// object key still filled cloud-side), so each rollout points at its own slice of the
    /// recording rather than every rollout sharing the whole blob.
    #[must_use]
    pub fn drain_available(&self) -> Vec<Rollout> {
        let mut out = Vec::new();
        // try_recv returns Err the moment the buffer is empty (or the sender is
        // gone), which ends the loop without blocking.
        while let Ok(record) = self.rx.try_recv() {
            let rollout = self.assemble(record);
            if let Some(sink) = &self.rollout_sink {
                // The lock is taken only here, off the control loop; the hot-path
                // log_step never touches it. A sink write failure is logged rather than
                // panicking the drain: the rollout is still returned in-memory, so a
                // transient disk error degrades durability without dropping the batch the
                // caller is about to register.
                let mut sink = sink.lock().expect("rollout sink mutex poisoned");
                if let Err(e) = sink.append(&rollout) {
                    eprintln!(
                        "fieldloop-capture: failed to append rollout to sink {}: {e}",
                        sink.path().display()
                    );
                }
            }
            out.push(rollout);
        }
        out
    }

    /// Expand one compact record into a full rollout. Off the hot path, so it may
    /// take the registry lock and allocate the strings.
    fn assemble(&self, record: StepRecord) -> Rollout {
        let ctx = {
            let contexts = self
                .shared
                .contexts
                .lock()
                .expect("context registry mutex poisoned");
            // The handle is a slot index into the only-growing registry, so it is
            // always in range for a handle this Capture handed out.
            contexts[record.ctx.0 as usize].clone()
        };

        // Stamp the same boot session minted at construction onto every clock, so
        // monotonic readings across this Capture's rollouts are comparable.
        let clock = MonoClock::new(self.shared.boot_id, record.mono_ns, record.wall_ns);

        // Observation pointer: the object key is still assigned cloud-side (the robot does
        // not choose where bytes land), so it stays empty here. But this step's OWN
        // message range within the recording IS known on-robot, so stamp it onto the
        // pointer's range when the caller supplied one. That is what makes each rollout
        // point at its own slice of the recorded MCAP object instead of every rollout
        // sharing the whole blob — the per-rollout correspondence a replay viewer needs.
        let observation_ref = match record.observation_range {
            None => PayloadRef::none(),
            Some(range) => PayloadRef {
                // Empty key: filled cloud-side once the object's bucket location is known.
                object_key: String::new(),
                range: Some(ByteRange {
                    start: range.start,
                    end: range.end,
                }),
                // Checksum of the referenced bytes is computed by the uploader from the
                // real bytes, not here, so it stays unknown until then.
                content_sha256: None,
            },
        };

        // Build the rollout, then overwrite its id with the one the hot path minted
        // and returned to the caller. Rollout::new mints its own id, but the caller
        // already holds (and may have transmitted) record.rollout_id, so that id is
        // authoritative and must be preserved.
        let mut rollout = Rollout::new(
            self.shared.robot.clone(),
            record.episode_id,
            record.step_index,
            clock,
            ctx.policy_version,
            ctx.model_hash,
            ctx.embodiment,
            ctx.task_id,
            // The observation pointer carries this step's own message range (see above);
            // the action pointer is filled by later stages, not in the loop.
            observation_ref,
            PayloadRef::none(),
            // Inline context is filled by later stages, not in the loop.
            BoundedBlob::empty(),
            record.inference_us,
        );
        rollout.id = record.rollout_id;
        // server_anchor / trust stay None and schema_conformance stays Unchecked,
        // set by Rollout::new; reaffirmed here as the documented pre-ingest defaults.
        debug_assert!(rollout.server_anchor.is_none());
        debug_assert!(rollout.trust.is_none());
        debug_assert_eq!(rollout.schema_conformance, SchemaConformance::Unchecked);
        rollout
    }

    /// The boot session id minted at construction and stamped onto every assembled
    /// rollout's clock. Exposed so a consumer can confirm which boot a batch belongs
    /// to.
    #[must_use]
    pub fn boot_id(&self) -> BootId {
        self.shared.boot_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_types::{RobotId, TenantId};

    fn robot() -> RobotIdentity {
        RobotIdentity::new(TenantId::new("tenant-a"), RobotId::new("robot-1"))
    }

    fn fake_clock() -> Box<dyn ClockSource> {
        // Start at 1000ns, advance 10ns per read, fixed advisory wall time.
        Box::new(FakeClock::new(1000, 10, 42))
    }

    #[test]
    fn log_step_returns_id_and_drains_to_matching_rollout() {
        let (cap, drain) = Capture::with_clock(robot(), 16, fake_clock());
        let ctx = cap.register_context(
            PolicyVersion::new("nav@v1+abc123abc123"),
            "model-sha-xyz",
            "arm6dof",
            "pick-and-place",
        );
        let episode = EpisodeId::new();

        let id = cap.log_step(episode, 7, ctx, 250);

        let rollouts = drain.drain_available();
        assert_eq!(rollouts.len(), 1);
        let r = &rollouts[0];
        // Id / episode / step / inference round-trip through the compact record.
        assert_eq!(r.id, id);
        assert_eq!(r.episode_id, episode);
        assert_eq!(r.step_index, 7);
        assert_eq!(r.inference_us, 250);
        // Registered context resolved back to its strings.
        assert_eq!(r.policy_version, PolicyVersion::new("nav@v1+abc123abc123"));
        assert_eq!(r.model_hash, "model-sha-xyz");
        assert_eq!(r.embodiment, "arm6dof");
        assert_eq!(r.task_id, "pick-and-place");
        // Clock carries the construction boot id.
        assert_eq!(r.clock.boot_id, drain.boot_id());
        assert_eq!(r.clock.ts_wall_ns, 42);
        // Pre-ingest defaults left for later stages.
        assert!(r.observation_ref.is_empty());
        assert!(r.action_ref.is_empty());
        assert!(r.context.is_empty());
        assert!(r.server_anchor.is_none());
        assert!(r.trust.is_none());
        assert_eq!(r.schema_conformance, SchemaConformance::Unchecked);
    }

    #[test]
    fn full_queue_drops_and_counts_but_still_returns_ids() {
        // Capacity 2: the third and later un-drained sends overflow.
        let (cap, _drain) = Capture::with_clock(robot(), 2, fake_clock());
        let ctx = cap.register_context(PolicyVersion::new("p@v1+deadbeefcafe"), "h", "emb", "task");
        let episode = EpisodeId::new();

        // Fill the buffer (2) and overflow it (3 more), without draining.
        let mut ids = Vec::new();
        for step in 0..5 {
            ids.push(cap.log_step(episode, step, ctx, 100));
        }

        // Nothing blocked or panicked; every call returned a usable id.
        assert_eq!(ids.len(), 5);
        // Distinct, valid ids even for the dropped records.
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
        // Three records past capacity-2 were dropped and counted.
        assert_eq!(cap.dropped(), 3);
    }

    #[test]
    fn monotonic_value_is_non_decreasing_across_steps() {
        let (cap, drain) = Capture::with_clock(robot(), 64, fake_clock());
        let ctx = cap.register_context(PolicyVersion::new("p@v1+0011223344"), "h", "emb", "task");
        let episode = EpisodeId::new();

        for step in 0..10 {
            cap.log_step(episode, step, ctx, 10);
        }

        let rollouts = drain.drain_available();
        assert_eq!(rollouts.len(), 10);
        let mut prev = 0u64;
        for r in &rollouts {
            assert!(r.clock.mono_ns >= prev, "mono_ns must be non-decreasing");
            prev = r.clock.mono_ns;
        }
    }

    #[test]
    fn context_handle_resolves_to_the_right_registered_context() {
        let (cap, drain) = Capture::with_clock(robot(), 16, fake_clock());
        // Two distinct contexts; each handle must resolve to its own strings.
        let ctx_a = cap.register_context(
            PolicyVersion::new("a@v1+aaaaaaaaaaaa"),
            "ha",
            "emb-a",
            "task-a",
        );
        let ctx_b = cap.register_context(
            PolicyVersion::new("b@v1+bbbbbbbbbbbb"),
            "hb",
            "emb-b",
            "task-b",
        );
        let episode = EpisodeId::new();

        cap.log_step(episode, 0, ctx_b, 1);
        cap.log_step(episode, 1, ctx_a, 1);

        let rollouts = drain.drain_available();
        assert_eq!(rollouts.len(), 2);
        // First step used ctx_b.
        assert_eq!(rollouts[0].embodiment, "emb-b");
        assert_eq!(rollouts[0].task_id, "task-b");
        // Second step used ctx_a.
        assert_eq!(rollouts[1].embodiment, "emb-a");
        assert_eq!(rollouts[1].task_id, "task-a");
    }

    /// Parse a JSONL rollouts file exactly the way the deployed agent binary does:
    /// split on newlines, skip blank lines, and `serde_json::from_str::<Rollout>` each
    /// remaining line. Reusing this read path here is what proves the sink's lines match
    /// the format the agent consumes — not a bespoke decoder that could diverge from it.
    fn read_rollouts_jsonl(path: &Path) -> Vec<Rollout> {
        let text = std::fs::read_to_string(path).expect("rollouts file should be readable");
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| serde_json::from_str::<Rollout>(line).expect("line must parse as Rollout"))
            .collect()
    }

    #[test]
    fn draining_writes_one_jsonl_line_per_rollout_that_round_trips_to_equal_rollout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rollouts.jsonl");

        let (cap, drain) = Capture::with_clock(robot(), 64, fake_clock());
        let drain = drain.with_rollout_sink(&path).expect("sink should open");
        let ctx = cap.register_context(
            PolicyVersion::new("nav@v1+abc123abc123"),
            "model-sha-xyz",
            "arm6dof",
            "pick-and-place",
        );
        let episode = EpisodeId::new();

        // Drain N rollouts.
        let n = 5u32;
        for step in 0..n {
            cap.log_step(episode, step, ctx, 100 + step);
        }
        let in_memory = drain.drain_available();
        assert_eq!(in_memory.len(), n as usize);

        // N JSONL lines were written.
        let on_disk = read_rollouts_jsonl(&path);
        assert_eq!(
            on_disk.len(),
            n as usize,
            "draining N rollouts must write N JSONL lines"
        );

        // Each line round-trips back to an equal Rollout, in the same order the drain
        // returned — proving serde_json::to_string -> read line -> from_str is identity
        // and matches the in-memory batch the caller registers.
        assert_eq!(on_disk, in_memory);

        // And the canonical form the agent reads is exactly serde_json::to_string of the
        // rollout: assert byte-equality against the line we round-tripped.
        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), n as usize);
        for (line, rollout) in lines.iter().zip(in_memory.iter()) {
            assert_eq!(*line, serde_json::to_string(rollout).unwrap());
        }
    }

    #[test]
    fn append_after_reopen_does_not_clobber_prior_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rollouts.jsonl");

        // First session: drain 3 rollouts through a sink, then drop everything (closing
        // the file) to simulate a process restart between batches.
        let first_ids = {
            let (cap, drain) = Capture::with_clock(robot(), 64, fake_clock());
            let drain = drain.with_rollout_sink(&path).expect("sink should open");
            let ctx =
                cap.register_context(PolicyVersion::new("p@v1+aaaaaaaaaaaa"), "h", "emb", "task");
            let episode = EpisodeId::new();
            for step in 0..3 {
                cap.log_step(episode, step, ctx, 10);
            }
            let r = drain.drain_available();
            r.iter().map(|x| x.id).collect::<Vec<_>>()
        };
        assert_eq!(read_rollouts_jsonl(&path).len(), 3);

        // Second session: reopen the SAME path (append mode) and drain 2 more. The new
        // sink must extend the file, not truncate the 3 prior lines.
        let second_ids = {
            let (cap, drain) = Capture::with_clock(robot(), 64, fake_clock());
            let drain = drain.with_rollout_sink(&path).expect("sink should reopen");
            let ctx =
                cap.register_context(PolicyVersion::new("p@v1+bbbbbbbbbbbb"), "h", "emb", "task");
            let episode = EpisodeId::new();
            for step in 0..2 {
                cap.log_step(episode, step, ctx, 10);
            }
            let r = drain.drain_available();
            r.iter().map(|x| x.id).collect::<Vec<_>>()
        };

        // All 5 lines are present and parse; the first 3 ids are intact (not clobbered)
        // and the 2 new ones are appended after them in order.
        let all = read_rollouts_jsonl(&path);
        assert_eq!(all.len(), 5, "reopen must append, not truncate");
        let all_ids: Vec<_> = all.iter().map(|x| x.id).collect();
        let mut expected = first_ids;
        expected.extend(second_ids);
        assert_eq!(all_ids, expected);
    }

    #[test]
    fn log_step_does_no_sink_io_so_the_hot_path_stays_off_disk() {
        // The sink lives on the Drain; the hot-path log_step only touches the bounded
        // queue. Prove the write work happens on the drain side, not the log side: with a
        // sink configured, logging N steps WITHOUT draining writes zero lines, and the
        // lines only appear once we drain.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rollouts.jsonl");

        let (cap, drain) = Capture::with_clock(robot(), 64, fake_clock());
        let drain = drain.with_rollout_sink(&path).expect("sink should open");
        let ctx = cap.register_context(PolicyVersion::new("p@v1+0011223344"), "h", "emb", "task");
        let episode = EpisodeId::new();

        for step in 0..8 {
            cap.log_step(episode, step, ctx, 10);
        }
        // log_step never wrote to the sink: the file exists only after a drain, so before
        // draining there are no lines on disk.
        assert!(
            !path.exists() || read_rollouts_jsonl(&path).is_empty(),
            "log_step must not write to the sink; the hot path stays off disk"
        );

        // Draining is what produces the durable lines.
        let drained = drain.drain_available();
        assert_eq!(drained.len(), 8);
        assert_eq!(
            read_rollouts_jsonl(&path).len(),
            8,
            "the drain side, not log_step, performs the sink writes"
        );
    }

    #[test]
    fn no_sink_by_default_writes_nothing() {
        // Default Drain has no sink, so draining must not create any file and must not
        // change existing drain semantics.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rollouts.jsonl");

        let (cap, drain) = Capture::with_clock(robot(), 16, fake_clock());
        let ctx = cap.register_context(PolicyVersion::new("p@v1+5566778899"), "h", "emb", "task");
        let episode = EpisodeId::new();
        cap.log_step(episode, 0, ctx, 1);
        let rollouts = drain.drain_available();

        assert_eq!(rollouts.len(), 1);
        assert!(
            !path.exists(),
            "sink is off by default: no file should be written"
        );
    }

    #[test]
    fn each_rollout_carries_its_own_observation_range_not_a_shared_one() {
        // Two steps, each recording a distinct frame range, must assemble into two
        // rollouts whose observation_ref.range differ — the per-rollout correspondence,
        // the regression the capture audit flagged (every rollout sharing one pointer).
        let (cap, drain) = Capture::with_clock(robot(), 16, fake_clock());
        let ctx = cap.register_context(PolicyVersion::new("p@v1+abcabcabcabc"), "h", "emb", "task");
        let episode = EpisodeId::new();

        // Step 0's frame is message [0, 1); step 1's is [1, 2) — distinct, non-overlapping
        // ranges, exactly what the recorder hands back for two consecutive single-frame
        // steps.
        cap.log_step_with_observation_range(
            episode,
            0,
            ctx,
            10,
            Some(PayloadMessageRange { start: 0, end: 1 }),
        );
        cap.log_step_with_observation_range(
            episode,
            1,
            ctx,
            10,
            Some(PayloadMessageRange { start: 1, end: 2 }),
        );

        let rollouts = drain.drain_available();
        assert_eq!(rollouts.len(), 2);
        let r0 = &rollouts[0].observation_ref.range;
        let r1 = &rollouts[1].observation_ref.range;
        assert_eq!(*r0, Some(ByteRange { start: 0, end: 1 }));
        assert_eq!(*r1, Some(ByteRange { start: 1, end: 2 }));
        // The whole point: the two rollouts do NOT share a range.
        assert_ne!(r0, r1, "each rollout must point at its own frame range");
    }

    #[test]
    fn log_step_without_range_leaves_observation_ref_empty() {
        // The plain log_step path (no recorded frame) leaves observation_ref empty, so the
        // existing pre-ingest default is unchanged for callers that don't record frames.
        let (cap, drain) = Capture::with_clock(robot(), 16, fake_clock());
        let ctx = cap.register_context(PolicyVersion::new("p@v1+0a0a0a0a0a0a"), "h", "emb", "task");
        let episode = EpisodeId::new();
        cap.log_step(episode, 0, ctx, 1);
        let rollouts = drain.drain_available();
        assert_eq!(rollouts.len(), 1);
        assert!(rollouts[0].observation_ref.is_empty());
        assert!(rollouts[0].observation_ref.range.is_none());
    }

    #[test]
    fn real_system_clock_is_non_decreasing() {
        // Exercise the real clock path too, not just the fake.
        let (cap, drain) = Capture::new(robot(), 64);
        let ctx = cap.register_context(PolicyVersion::new("p@v1+5566778899"), "h", "emb", "task");
        let episode = EpisodeId::new();
        for step in 0..5 {
            cap.log_step(episode, step, ctx, 5);
        }
        let rollouts = drain.drain_available();
        let mut prev = 0u64;
        for r in &rollouts {
            assert!(r.clock.mono_ns >= prev);
            prev = r.clock.mono_ns;
        }
    }
}
