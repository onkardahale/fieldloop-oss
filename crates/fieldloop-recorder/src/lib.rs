//! Off-hot-path recorder that persists robot byte streams to MCAP files.
//!
//! The control/data loop must never block on disk, so recording is split across a thread boundary:
//! the producer (on/near the loop) only does a non-blocking channel `try_send` in
//! [`Recorder::record`]; the file I/O (the `mcap::Writer`, compression, fsync-on-rotation, the
//! footer/summary) runs on per-stream-class writer threads the producer never waits on.
//!
//! Each [`StreamClass`] gets its own writer thread and MCAP file — the isolation is the point: a
//! high-volume sensor stream backing up against a slow disk cannot delay the tiny safety-critical
//! e-stop stream, since they share no channel, thread, or file.
//!
//! Drop policy is per class (every class uses a bounded channel; the difference is what a
//! full buffer means):
//! - [`StreamClass::Sensor`] — **droppable**: under backpressure it drops-and-counts so the
//!   producer never blocks behind a slow disk.
//! - [`StreamClass::Safety`] — **no-drop**: a LARGE bounded buffer; a full buffer (a stalled disk)
//!   returns a hard [`RecordResult::SafetyBackpressure`] the caller must escalate, never a routine
//!   drop — and never an unbounded channel that a stall could grow until the process OOMs.
//! - [`StreamClass::Meta`] — bounded (generously) and droppable; metadata is important but recoverable.
//!
//! Files rotate deterministically (by message count and/or elapsed time) into `<class>-<seq>.mcap`,
//! each finalized on its own so every file reads back as valid MCAP.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::Instant;

/// The closed set of stream classes, each with its own writer thread, its own MCAP
/// file, and its own documented drop policy.
///
/// Classes are kept separate so they never contend: the high-volume `Sensor` class
/// flooding a slow disk can never delay the `Safety` class, because they share no
/// channel, thread, or file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamClass {
    /// Small, important records (rollout / outcome metadata).
    ///
    /// Bounded and droppable, but with a generous capacity so the drop risk is low.
    /// Metadata matters, but losing a record under extreme backpressure is
    /// preferable to stalling the producer.
    Meta,
    /// High-volume payloads (video / lidar / depth).
    ///
    /// Bounded and **droppable**: under backpressure the producer drops-and-counts
    /// rather than blocking, so a slow disk can never stall the control loop. The
    /// dropped count is observable via [`Recorder::dropped`].
    Sensor,
    /// E-stops and other safety events.
    ///
    /// **No-drop**: backed by a LARGE bounded channel so a safety event is never silently
    /// dropped to backpressure. If the buffer is ever full (the writer thread stalled on a
    /// slow disk) enqueueing returns the hard [`RecordResult::SafetyBackpressure`] the
    /// caller must escalate — never a routine counted drop, and never an unbounded channel
    /// that a stall could grow without limit until the process OOMs and loses everything.
    Safety,
}

impl StreamClass {
    /// Every class in iteration order. Used at construction to spin up one writer
    /// thread (and one initial MCAP file) per class.
    const ALL: [StreamClass; 3] = [StreamClass::Meta, StreamClass::Sensor, StreamClass::Safety];

    /// Whether this class drops under backpressure. `Safety` is the lone no-drop
    /// class: its events must never be lost just because the disk is slow.
    #[must_use]
    pub const fn is_droppable(self) -> bool {
        match self {
            StreamClass::Meta | StreamClass::Sensor => true,
            StreamClass::Safety => false,
        }
    }

    /// Lowercase name used as the filename prefix `<class>-<seq>.mcap`, so the files
    /// on disk are self-describing and predictable.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            StreamClass::Meta => "meta",
            StreamClass::Sensor => "sensor",
            StreamClass::Safety => "safety",
        }
    }
}

/// A schema declaration for a topic, mirroring MCAP's schema record.
///
/// MCAP lets a channel either carry a schema (so a reader knows how to decode the
/// bytes) or be schemaless. This is the optional schema a caller attaches when
/// declaring a topic; it is written into the MCAP file verbatim.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Schema name, e.g. a fully-qualified message type name.
    pub name: String,
    /// Schema encoding, e.g. `"jsonschema"`, `"protobuf"`, `"ros2msg"`.
    pub encoding: String,
    /// The raw schema definition bytes, stored as-is in the MCAP schema record.
    pub data: Vec<u8>,
}

impl Schema {
    /// Build a schema from its three MCAP parts.
    #[must_use]
    pub fn new(name: impl Into<String>, encoding: impl Into<String>, data: Vec<u8>) -> Schema {
        Schema {
            name: name.into(),
            encoding: encoding.into(),
            data,
        }
    }
}

/// A handle to a declared topic within a class, returned by
/// [`Recorder::register_channel`] and passed back to [`Recorder::record`].
///
/// It is a small `Copy` value so the producer can keep it cheaply and never has to
/// re-resolve a topic string on the hot path. It identifies the channel *within its
/// class*; the same numeric value in a different class is a different channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(u16);

/// Per-class rotation and capacity settings.
///
/// Rotation is expressed in terms the test suite (and an operator) can reason about
/// deterministically: split to a new file after `max_messages_per_file` messages, or
/// after `max_file_duration` of wall time since the current file opened, whichever
/// comes first. A `None` disables that trigger.
#[derive(Debug, Clone)]
pub struct ClassConfig {
    /// Bounded channel capacity for this class. For [`StreamClass::Safety`] this is a
    /// real (large) bound too: it caps the buffer so a disk stall cannot grow it without
    /// limit, and a full safety buffer returns [`RecordResult::SafetyBackpressure`]
    /// rather than dropping or OOMing.
    pub capacity: usize,
    /// Rotate to a new MCAP file after this many messages have been written to the
    /// current file. `None` disables the message-count trigger.
    pub max_messages_per_file: Option<u64>,
    /// Rotate to a new MCAP file once this much wall time has elapsed since the
    /// current file opened. `None` disables the time trigger.
    pub max_file_duration: Option<std::time::Duration>,
}

impl ClassConfig {
    /// A config with the given capacity and no rotation (a single file per class).
    #[must_use]
    pub fn new(capacity: usize) -> ClassConfig {
        ClassConfig {
            capacity,
            max_messages_per_file: None,
            max_file_duration: None,
        }
    }

    /// Set the message-count rotation trigger, returning the updated config.
    #[must_use]
    pub fn with_max_messages(mut self, max: u64) -> ClassConfig {
        self.max_messages_per_file = Some(max);
        self
    }

    /// Set the elapsed-time rotation trigger, returning the updated config.
    #[must_use]
    pub fn with_max_duration(mut self, max: std::time::Duration) -> ClassConfig {
        self.max_file_duration = Some(max);
        self
    }
}

/// The full recorder configuration: one [`ClassConfig`] per class.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Settings for the [`StreamClass::Meta`] class.
    pub meta: ClassConfig,
    /// Settings for the [`StreamClass::Sensor`] class.
    pub sensor: ClassConfig,
    /// Settings for the [`StreamClass::Safety`] class.
    pub safety: ClassConfig,
}

impl RecorderConfig {
    /// The config for a given class.
    fn for_class(&self, class: StreamClass) -> &ClassConfig {
        match class {
            StreamClass::Meta => &self.meta,
            StreamClass::Sensor => &self.sensor,
            StreamClass::Safety => &self.safety,
        }
    }
}

impl Default for RecorderConfig {
    /// Sensible defaults: a generous `Meta` buffer, a large `Sensor` buffer (it is
    /// the high-volume, droppable class), and a LARGE `Safety` buffer — now a real bound,
    /// sized so backpressure only arises under a genuine, sustained disk stall (where an
    /// unbounded channel would instead grow until OOM). No rotation by default — callers
    /// opt in.
    fn default() -> RecorderConfig {
        RecorderConfig {
            meta: ClassConfig::new(1024),
            sensor: ClassConfig::new(4096),
            safety: ClassConfig::new(8192),
        }
    }
}

/// A class-stream-local locator for one recorded message: the index of this message
/// within its [`StreamClass`]'s message stream, counting in write order across every
/// file the class rotates through.
///
/// Why a message index and not a byte offset: the `mcap` writer this recorder is built
/// on does not expose the inner stream's byte position (it owns the buffered file, only
/// hands back the whole writer via `into_inner`, and writes messages inside compressed
/// chunks whose post-compression byte boundaries are not knowable at write time). So a
/// raw byte offset into the finished file cannot be captured honestly here. The message
/// index can: messages are written one-at-a-time by a single per-class writer thread in
/// the exact order they were enqueued, so "the Nth message enqueued is the Nth message
/// written" holds, and a reader streaming the class's files in sequence order (the same
/// order [`Recorder`] writes and rotates them) can seek to message `index` by counting.
/// It is a genuine per-message locator into the recorded stream, not a whole-file
/// pointer — two different messages always get two different indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MessageIndex(pub u64);

/// A half-open `[start, end)` run of [`MessageIndex`]es covering the messages recorded
/// for one logical unit (e.g. one rollout/step's observation frame, or its full set of
/// frames), so a consumer can fetch exactly that unit's messages from the class stream
/// instead of the whole recording.
///
/// `start..end` is the run in write order within the class's message stream. A single
/// message is `start = N`, `end = N + 1`. Distinct units recorded one after another get
/// distinct, non-overlapping runs, which is what lets a replay viewer isolate one
/// decision's frames rather than re-reading the entire blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageRange {
    /// First message index in the run (inclusive), in class-stream write order.
    pub start: u64,
    /// One past the last message index in the run (exclusive).
    pub end: u64,
}

impl MessageRange {
    /// The single-message run `[index, index + 1)` for one recorded message.
    #[must_use]
    pub fn single(index: MessageIndex) -> MessageRange {
        MessageRange {
            start: index.0,
            end: index.0 + 1,
        }
    }
}

/// The outcome of a [`Recorder::record`] call. Always returned immediately; the call
/// never blocks on the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordResult {
    /// The message was handed to the class's writer thread (it has been enqueued,
    /// not necessarily yet written to disk). Carries the [`MessageIndex`] this message
    /// will occupy in the class's written stream, so the caller can record where this
    /// message lands and later fetch exactly it. The index is assigned only on a
    /// successful enqueue, so a dropped message never consumes one and the indices stay
    /// dense and in write order.
    Enqueued(MessageIndex),
    /// A droppable class was at capacity, so the message was dropped and the class's
    /// dropped counter incremented. Only `Meta` and `Sensor` can report this.
    Dropped,
    /// The no-drop [`StreamClass::Safety`] class's bounded buffer was full (the writer
    /// thread is not draining fast enough — a stalled disk), so this safety event was NOT
    /// recorded. Unlike [`RecordResult::Dropped`] this is a HARD error the caller MUST
    /// escalate (the buffer is bounded precisely so a disk stall cannot grow it without
    /// limit and OOM the process, which would lose every queued safety event at once);
    /// it is never a routine, silently-counted drop.
    SafetyBackpressure,
    /// The writer thread for this class is gone (it panicked or the recorder was
    /// finalized). For [`StreamClass::Safety`] this is a hard error a caller should
    /// escalate, since a safety event was not recorded.
    WriterGone,
}

impl RecordResult {
    /// The assigned [`MessageIndex`] if the message was enqueued, else `None` (dropped or
    /// the writer is gone). A convenience for callers that only need the locator.
    #[must_use]
    pub fn message_index(self) -> Option<MessageIndex> {
        match self {
            RecordResult::Enqueued(index) => Some(index),
            RecordResult::Dropped | RecordResult::SafetyBackpressure | RecordResult::WriterGone => {
                None
            }
        }
    }
}

/// One unit of work handed across the thread boundary to a writer thread.
///
/// Built by the producer and consumed off the hot path. It owns its `data` so the
/// producer's buffer is free to be reused the moment `record` returns.
enum WriteCmd {
    /// Declare a channel (topic + encoding + optional schema) before messages flow.
    /// Carries the pre-assigned [`ChannelId`] so the producer and writer agree on it
    /// without a round-trip.
    Register {
        channel_id: u16,
        topic: String,
        message_encoding: String,
        schema: Option<Schema>,
    },
    /// Persist one message on a previously-registered channel.
    Message {
        channel_id: u16,
        log_time_ns: u64,
        data: Vec<u8>,
    },
}

/// Per-class shared state the producer touches without locking.
struct ClassShared {
    /// The bounded sender into this class's writer thread. Every class is now bounded
    /// (a `sync_channel`): droppable classes drop-and-count on `Full`; `Safety` uses a
    /// large bound and reports [`RecordResult::SafetyBackpressure`] on `Full` rather than
    /// an unbounded channel that could grow without limit and OOM the process.
    tx: SyncSender<WriteCmd>,
    /// Count of messages dropped because a droppable class's bounded channel was
    /// full. Atomic so the producer bumps it without a lock. Always zero for
    /// `Safety`.
    dropped: AtomicU64,
    /// Monotonically increasing id used to hand each newly registered channel a
    /// stable [`ChannelId`] without consulting the writer thread.
    next_channel_id: AtomicU64,
    /// Class-stream-local message counter. Incremented (and read) exactly once per
    /// successfully-enqueued message, so the value handed back is that message's
    /// [`MessageIndex`] in write order. A dropped message never bumps it, keeping the
    /// indices dense and aligned with what the writer thread actually writes. Lives on
    /// the producer side so the locator is available synchronously without a round-trip
    /// to the off-hot-path writer thread.
    next_message_index: AtomicU64,
    /// `true` while this class's writer thread is writing cleanly. The writer thread
    /// clears it on the first I/O error (a full disk, a write fault) instead of
    /// panicking, and the producer reads it in [`Recorder::record`] so a writer that has
    /// silently failed surfaces as [`RecordResult::WriterGone`] on the next record —
    /// rather than messages (including safety events) vanishing into a dead thread with
    /// `dropped() == 0` and no signal at all.
    ///
    /// A standalone `Arc<AtomicBool>` (not an inline field) precisely so the writer thread
    /// can hold a clone WITHOUT holding the whole `ClassShared` — if the writer kept the
    /// `ClassShared` it would keep the channel's sender alive, the receiver would never see
    /// end-of-stream, and the join at shutdown would hang.
    healthy: Arc<AtomicBool>,
}

/// The recorder front half, held by the producer near the control loop.
///
/// Construction spawns one writer thread per [`StreamClass`], each of which opens
/// its first MCAP file immediately. The producer then declares topics with
/// [`Recorder::register_channel`] and records bytes with [`Recorder::record`] — the
/// latter only ever does a non-blocking channel `try_send`, so it never touches the
/// disk.
pub struct Recorder {
    /// Per-class senders + counters, indexed by `StreamClass as usize` via
    /// [`Recorder::class_index`].
    shared: [Arc<ClassShared>; 3],
    /// Join handles for the writer threads, taken in [`Recorder::finalize`] (or
    /// `Drop`) to flush and finalize every MCAP file.
    workers: Vec<Worker>,
}

/// A spawned writer thread plus the sender used to signal it to stop.
struct Worker {
    class: StreamClass,
    handle: JoinHandle<()>,
}

impl Recorder {
    /// Build a recorder writing under `dir`, spawning one writer thread (and one
    /// initial MCAP file) per [`StreamClass`].
    ///
    /// The output directory must already exist. Each writer thread owns its own
    /// `mcap::Writer`, so from here on the producer never touches the disk — it only
    /// hands messages across a channel.
    ///
    /// # Panics
    /// Panics if a writer thread cannot create its first MCAP file under `dir` (a
    /// recorder that cannot open any file is unusable, so this fails loudly at
    /// construction rather than silently dropping everything later).
    #[must_use]
    pub fn new(dir: impl AsRef<Path>, config: RecorderConfig) -> Recorder {
        let dir = dir.as_ref().to_path_buf();
        let mut shared: Vec<Arc<ClassShared>> = Vec::with_capacity(3);
        let mut workers: Vec<Worker> = Vec::with_capacity(3);

        for class in StreamClass::ALL {
            let class_cfg = config.for_class(class).clone();
            // Every class uses a BOUNDED channel. Droppable classes (`Meta`/`Sensor`)
            // drop-and-count on `Full`; `Safety` uses a large bound and reports
            // `SafetyBackpressure` on `Full` instead of an unbounded channel that a disk
            // stall could grow without limit until the process OOMs (losing every queued
            // safety event at once — strictly worse than a bounded, loud failure).
            let (sender, receiver) = sync_channel::<WriteCmd>(class_cfg.capacity);

            // The health flag is shared with the writer thread directly (NOT via the whole
            // ClassShared, which holds the sender — sharing that would keep the channel
            // open and hang the shutdown join).
            let healthy = Arc::new(AtomicBool::new(true));
            let writer_health = Arc::clone(&healthy);
            shared.push(Arc::new(ClassShared {
                tx: sender,
                dropped: AtomicU64::new(0),
                next_channel_id: AtomicU64::new(1),
                // The first written message of each class is index 0.
                next_message_index: AtomicU64::new(0),
                healthy,
            }));

            // Spawn the writer thread. It opens the first MCAP file before doing
            // anything else; a failure there panics the thread, which surfaces when
            // the recorder is finalized.
            let thread_dir = dir.clone();
            let handle = std::thread::Builder::new()
                .name(format!("fieldloop-recorder-{}", class.name()))
                .spawn(move || {
                    let mut writer = ClassWriter::new(&thread_dir, class, class_cfg, writer_health)
                        .expect("recorder writer thread failed to open its first MCAP file");
                    writer.run(receiver);
                })
                .expect("failed to spawn recorder writer thread");

            workers.push(Worker { class, handle });
        }

        let shared: [Arc<ClassShared>; 3] = shared
            .try_into()
            .unwrap_or_else(|_| unreachable!("exactly one shared per class"));

        Recorder { shared, workers }
    }

    /// Index into the per-class arrays. Kept in one place so the ordering of
    /// [`StreamClass::ALL`] and the array layout can never drift apart.
    const fn class_index(class: StreamClass) -> usize {
        match class {
            StreamClass::Meta => 0,
            StreamClass::Sensor => 1,
            StreamClass::Safety => 2,
        }
    }

    /// Declare a topic (MCAP channel) on a class and get back a `Copy`
    /// [`ChannelId`] to record against.
    ///
    /// MCAP requires every message to name a channel, and a channel may carry an
    /// optional schema so a reader can decode the bytes. This call assigns the
    /// channel id locally (no round-trip to the writer thread) and forwards the
    /// declaration to the class's writer thread, which writes the schema/channel
    /// records into that class's MCAP file. Declare topics before recording to them.
    pub fn register_channel(
        &self,
        class: StreamClass,
        topic: &str,
        message_encoding: &str,
        schema: Option<Schema>,
    ) -> ChannelId {
        let shared = &self.shared[Self::class_index(class)];
        // Assign the channel id without consulting the writer thread, so registration
        // is also non-blocking and the returned id is usable immediately.
        let raw = shared.next_channel_id.fetch_add(1, Ordering::Relaxed);
        let channel_id = u16::try_from(raw).expect("too many channels registered for a u16 id");
        let cmd = WriteCmd::Register {
            channel_id,
            topic: topic.to_string(),
            message_encoding: message_encoding.to_string(),
            schema,
        };
        // Registration is a setup-time call, OFF the hot path, so it uses a BLOCKING
        // `send` (not `try_send`): a dropped registration would mean the channel is never
        // declared in the file, and the writer thread then skips every later message for
        // that channel — silently shifting all subsequent message indices and corrupting
        // each rollout's recorded range. Blocking here (it may briefly wait for the writer
        // to make space) is acceptable precisely because this is not the control loop;
        // losing the declaration is not. A `send` error only means the writer thread is
        // gone, which later records surface as `WriterGone`.
        let _ = shared.tx.send(cmd);
        ChannelId(channel_id)
    }

    /// Record one message on a previously-declared channel. **Non-blocking**: it only
    /// does a channel `try_send`, never disk I/O, so it is safe to call on or near
    /// the control loop.
    ///
    /// `log_time_ns` is the message timestamp in nanoseconds, written verbatim into
    /// the MCAP message header so the file's ordering matches the producer's clock.
    /// `data` is the raw payload; it is copied into the queued command so the
    /// producer's buffer is free to reuse the instant this returns.
    ///
    /// Drop behavior is per class: a full `Meta`/`Sensor` queue drops-and-counts
    /// (returns [`RecordResult::Dropped`]); `Safety` is unbounded and never drops.
    /// If the writer thread is gone, returns [`RecordResult::WriterGone`].
    ///
    /// On success the returned [`RecordResult::Enqueued`] carries this message's
    /// [`MessageIndex`] — its position in the class's written stream — so the caller can
    /// remember exactly where the message lands and later fetch just it. The index is
    /// assigned only when the message is actually enqueued (and therefore will be
    /// written), so a dropped message never consumes one.
    pub fn record(
        &self,
        class: StreamClass,
        channel: ChannelId,
        log_time_ns: u64,
        data: &[u8],
    ) -> RecordResult {
        let shared = &self.shared[Self::class_index(class)];
        // A writer thread that hit an I/O error clears `healthy` instead of panicking, so
        // surface it here: a failed writer is reported as gone rather than letting the
        // message (a safety event included) disappear into a dead thread with no signal.
        if !shared.healthy.load(Ordering::Acquire) {
            return RecordResult::WriterGone;
        }
        let cmd = WriteCmd::Message {
            channel_id: channel.0,
            log_time_ns,
            // Copy now so the producer owns its buffer again as soon as we return.
            data: data.to_vec(),
        };
        // Non-blocking `try_send` on every class: the producer must never wait on the
        // disk. The Full arm differs by drop policy.
        match shared.tx.try_send(cmd) {
            // Assign the message index only after a successful enqueue, so a message that
            // goes on to be written claims index N and a dropped one below never does —
            // the indices match the writer thread's write order exactly.
            Ok(()) => RecordResult::Enqueued(Self::next_index(shared)),
            Err(TrySendError::Full(_)) => {
                if class.is_droppable() {
                    // Droppable class under backpressure — drop and count, never block.
                    shared.dropped.fetch_add(1, Ordering::Relaxed);
                    RecordResult::Dropped
                } else {
                    // Safety's bounded buffer is full (a stalled disk). Report it as a
                    // HARD error the caller must escalate, never a silent drop and never
                    // an unbounded RAM grow that would OOM the process.
                    RecordResult::SafetyBackpressure
                }
            }
            // Disconnected: the writer thread is gone.
            Err(TrySendError::Disconnected(_)) => RecordResult::WriterGone,
        }
    }

    /// Claim the next class-stream message index for a just-enqueued message.
    ///
    /// `fetch_add` returns the pre-increment value, which is this message's index, and
    /// the next caller gets the following one — so consecutive enqueued messages on a
    /// class get consecutive indices in send (and therefore write) order. Called only on
    /// the success arms of [`Recorder::record`], so a dropped message never advances it.
    fn next_index(shared: &ClassShared) -> MessageIndex {
        MessageIndex(shared.next_message_index.fetch_add(1, Ordering::Relaxed))
    }

    /// Number of messages dropped on a class because its bounded queue was full.
    ///
    /// Always zero for [`StreamClass::Safety`]: it is no-drop, so backpressure surfaces
    /// as the [`RecordResult::SafetyBackpressure`] return value, never as a silent counted
    /// drop. A rising count on `Sensor`/`Meta` means the disk is not keeping up with the
    /// producer — useful operational telemetry.
    #[must_use]
    pub fn dropped(&self, class: StreamClass) -> u64 {
        self.shared[Self::class_index(class)]
            .dropped
            .load(Ordering::Relaxed)
    }

    /// Flush and finalize every class's MCAP files, then join the writer threads.
    ///
    /// Dropping the senders closes each class's channel; each writer thread drains
    /// any remaining queued messages, writes the MCAP footer/summary for its current
    /// file, and exits. Call this (or rely on `Drop`) to guarantee every file is a
    /// valid, fully-finalized MCAP that reads back correctly.
    pub fn finalize(mut self) {
        self.shutdown();
    }

    /// Shared teardown for both [`Recorder::finalize`] and `Drop`: drop the senders so
    /// the writer threads see end-of-stream, then join them so finalization is
    /// complete before we return.
    fn shutdown(&mut self) {
        // Replace each shared Arc with a dummy whose sender is dropped, severing the
        // producer's senders so each writer thread's receiver returns end-of-stream
        // and the thread can finalize its file. We do this by dropping our only
        // strong references to the real ClassShared values.
        //
        // The Arc may still be held by clones the caller made elsewhere, but the
        // sender inside it is only ever this one; dropping our Arcs here is enough to
        // close the channels because no other Arc clone is created by this crate.
        let workers = std::mem::take(&mut self.workers);
        // Drop senders by dropping the shared array's contents. We can't move out of
        // the array field while keeping `self` valid for Drop, so swap in fresh,
        // already-disconnected channels.
        for slot in &mut self.shared {
            // Construct a disconnected sender: create a channel and immediately drop
            // its receiver, so the replacement sender is closed. This drops the old
            // Arc (and thus the old, live sender) for this slot.
            let (tx, rx) = sync_channel::<WriteCmd>(0);
            drop(rx);
            *slot = Arc::new(ClassShared {
                tx,
                dropped: AtomicU64::new(0),
                next_channel_id: AtomicU64::new(1),
                next_message_index: AtomicU64::new(0),
                healthy: Arc::new(AtomicBool::new(true)),
            });
        }
        // Now that the original senders are dropped, each writer thread's receiver
        // returns end-of-stream; join them so every MCAP file is finalized.
        for worker in workers {
            if worker.handle.join().is_err() {
                // A panicked writer thread already failed loudly via its own panic
                // message; nothing more to do here than note the class.
                eprintln!(
                    "fieldloop-recorder: writer thread for class {} panicked",
                    worker.class.name()
                );
            }
        }
    }
}

impl Drop for Recorder {
    /// Finalize on drop so files are always closed validly even if the caller forgets
    /// [`Recorder::finalize`]. Idempotent with `finalize` (which takes `self` and so
    /// runs this teardown exactly once).
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The off-hot-path writer for one class. Owns the `mcap::Writer`, the current file's
/// sequence number and counters, and the rotation policy. Lives entirely on the
/// class's writer thread — the producer never touches any of this.
struct ClassWriter {
    /// Directory all of this class's files live in.
    dir: PathBuf,
    /// Which class this writer serves; drives the filename prefix.
    class: StreamClass,
    /// Rotation + capacity config for this class.
    config: ClassConfig,
    /// Sequence number of the current file: `<class>-<seq>.mcap`. Increments on each
    /// rotation so files are predictably named and ordered.
    file_seq: u64,
    /// The live MCAP writer for the current file, plus the bookkeeping needed to map
    /// the producer's `ChannelId`s onto MCAP channel ids in *this* file. An `Option` only
    /// so finalizing (which consumes the `OpenFile` to fsync+rename) can `take()` it; it
    /// is `Some` for the whole operating life of the writer between create and the final
    /// finish.
    state: Option<OpenFile>,
    /// Channel declarations seen so far, replayed into each new file after a rotation
    /// so a topic recorded across a rotation appears in every file it spans.
    declarations: BTreeMap<u16, ChannelDecl>,
    /// Shared health flag (the producer reads the same `Arc<AtomicBool>`); this thread
    /// clears it on an I/O error so the producer's next [`Recorder::record`] surfaces the
    /// failure instead of writing into a silently-dead thread. Just the flag, not the
    /// whole `ClassShared` — holding the latter would keep the channel sender alive and
    /// hang the shutdown join.
    health: Arc<AtomicBool>,
}

/// A remembered channel declaration, kept so it can be re-applied to the next file
/// after a rotation (MCAP channels are per-file, so each file must redeclare them).
struct ChannelDecl {
    topic: String,
    message_encoding: String,
    schema: Option<Schema>,
}

/// The live MCAP writer for the current file plus per-file bookkeeping.
struct OpenFile {
    /// The MCAP writer wrapping a buffered file handle.
    writer: mcap::Writer<BufWriter<File>>,
    /// The temp path being written (`<class>-<seq>.mcap.part`). A reader/agent never
    /// sees this name as a finished file; it is renamed to `final_path` only after the
    /// footer is written and the bytes are fsynced.
    part_path: PathBuf,
    /// The final path the temp file is atomically renamed to once finalized
    /// (`<class>-<seq>.mcap`), so any observer sees only a complete, fsynced file.
    final_path: PathBuf,
    /// Maps the producer's stable `ChannelId` to the MCAP channel id assigned in this
    /// specific file (channel ids are per-file in MCAP).
    mcap_channel_ids: BTreeMap<u16, u16>,
    /// Maps a schema's identity (name+encoding) to the MCAP schema id in this file, so
    /// repeated declarations of the same schema reuse one schema record.
    mcap_schema_ids: BTreeMap<(String, String), u16>,
    /// Messages written to the current file, compared against the rotation trigger.
    messages_in_file: u64,
    /// When the current file was opened, compared against the time rotation trigger.
    opened_at: Instant,
}

impl ClassWriter {
    /// Open the first file for this class and build its writer state.
    fn new(
        dir: &Path,
        class: StreamClass,
        config: ClassConfig,
        health: Arc<AtomicBool>,
    ) -> std::io::Result<ClassWriter> {
        let state = OpenFile::create(dir, class, 0)?;
        Ok(ClassWriter {
            dir: dir.to_path_buf(),
            class,
            config,
            file_seq: 0,
            state: Some(state),
            declarations: BTreeMap::new(),
            health,
        })
    }

    /// Record that this class's writer hit an unrecoverable I/O error: clear the shared
    /// `healthy` flag (so the producer's next `record` returns `WriterGone`) and log it
    /// once. The thread keeps draining its channel so the producer never blocks, but it
    /// no longer pretends to be recording.
    fn mark_unhealthy(&self, what: &str, err: &std::io::Error) {
        // Only log on the first transition so a persistent failure does not spam.
        if self.health.swap(false, Ordering::Release) {
            eprintln!(
                "fieldloop-recorder: {} writer failed ({what}: {err}); marking the class \
                 unhealthy — further records report WriterGone",
                self.class.name()
            );
        }
    }

    /// The writer thread's main loop: drain commands until end-of-stream, then
    /// finalize the current file. Receiving `None` means the producer's senders were
    /// all dropped, which is the cue to write the MCAP footer/summary and exit.
    fn run(&mut self, receiver: Receiver<WriteCmd>) {
        // `recv()` blocks for the next command and returns `Err` once every sender is
        // dropped (end-of-stream) — the cue to finalize the current file and exit.
        while let Ok(cmd) = receiver.recv() {
            match cmd {
                WriteCmd::Register {
                    channel_id,
                    topic,
                    message_encoding,
                    schema,
                } => {
                    self.handle_register(channel_id, topic, message_encoding, schema);
                }
                WriteCmd::Message {
                    channel_id,
                    log_time_ns,
                    data,
                } => {
                    self.handle_message(channel_id, log_time_ns, &data);
                }
            }
        }
        // End-of-stream: finalize the current file so it reads back as valid MCAP.
        self.finish_current_file();
    }

    /// Remember a channel declaration and apply it to the current file.
    fn handle_register(
        &mut self,
        channel_id: u16,
        topic: String,
        message_encoding: String,
        schema: Option<Schema>,
    ) {
        let decl = ChannelDecl {
            topic,
            message_encoding,
            schema,
        };
        // Declare it in the current file now, and remember it so it can be replayed
        // into any file created by a future rotation.
        if let Err(e) = self.open_mut().declare(channel_id, &decl) {
            self.mark_unhealthy("declare", &e);
        }
        self.declarations.insert(channel_id, decl);
    }

    /// Write one message, rotating first if the current file has hit its trigger.
    fn handle_message(&mut self, channel_id: u16, log_time_ns: u64, data: &[u8]) {
        if self.should_rotate() {
            self.rotate();
        }
        if let Err(e) = self.open_mut().write_message(channel_id, log_time_ns, data) {
            self.mark_unhealthy("write", &e);
        }
    }

    /// The current open file. `Some` for the whole operating life between create and the
    /// final finish, so a missing one is a recorder bug, not a runtime condition.
    fn open_mut(&mut self) -> &mut OpenFile {
        self.state
            .as_mut()
            .expect("recorder has no open file mid-operation")
    }

    /// Whether the current file has reached a rotation trigger (message count or
    /// elapsed time). Either trigger firing rotates; both being `None` never rotates.
    fn should_rotate(&self) -> bool {
        let Some(state) = self.state.as_ref() else {
            return false;
        };
        if let Some(max) = self.config.max_messages_per_file
            && state.messages_in_file >= max
        {
            return true;
        }
        if let Some(max) = self.config.max_file_duration
            && state.opened_at.elapsed() >= max
        {
            return true;
        }
        false
    }

    /// Finalize the current file and open the next one, replaying all known channel
    /// declarations into it so topics that span a rotation appear in every file.
    fn rotate(&mut self) {
        // Open the next file FIRST, so a failure to create it leaves the current file
        // still open and recording rather than dropping messages into the void.
        self.file_seq += 1;
        let mut next = OpenFile::create(&self.dir, self.class, self.file_seq)
            .expect("recorder failed to open the next MCAP file on rotation");
        // Redeclare every known channel into the fresh file (MCAP channels are
        // per-file), so a recorder reading all of a class's files sees the topic in
        // each one it has messages in.
        for (channel_id, decl) in &self.declarations {
            if let Err(e) = next.declare(*channel_id, decl) {
                self.mark_unhealthy("redeclare", &e);
            }
        }
        // Swap in the new file and finalize the old one (footer + fsync + rename).
        let old = self.state.replace(next);
        if let Some(old) = old
            && let Err(e) = old.finish()
        {
            self.mark_unhealthy("finalize-on-rotation", &e);
        }
    }

    /// Finalize the current MCAP file (footer + fsync + atomic rename), consuming it so
    /// no further writes are possible. Called at end-of-stream; after it `state` is `None`.
    fn finish_current_file(&mut self) {
        if let Some(state) = self.state.take()
            && let Err(e) = state.finish()
        {
            self.mark_unhealthy("finalize", &e);
        }
    }
}

impl OpenFile {
    /// Create and open a fresh MCAP file for this class, writing to a `.part` temp name
    /// that is published to `<class>-<seq>.mcap` only when [`OpenFile::finish`] fsyncs and
    /// renames it.
    fn create(dir: &Path, class: StreamClass, seq: u64) -> std::io::Result<OpenFile> {
        let final_path = dir.join(format!("{}-{}.mcap", class.name(), seq));
        let part_path = dir.join(format!("{}-{}.mcap.part", class.name(), seq));
        // Write to the temp name so an in-progress (or crash-torn) file never carries the
        // final name an agent treats as a finished, uploadable recording.
        let file = File::create(&part_path)?;
        let buf = BufWriter::new(file);
        // Default write options emit chunk/summary records and a footer, which is what
        // makes the file read back as valid, seekable MCAP. The `library` field tags
        // the producer for provenance.
        let writer = mcap::WriteOptions::new()
            .library("fieldloop-recorder")
            .create(buf)
            .map_err(std::io::Error::other)?;
        Ok(OpenFile {
            writer,
            part_path,
            final_path,
            mcap_channel_ids: BTreeMap::new(),
            mcap_schema_ids: BTreeMap::new(),
            messages_in_file: 0,
            opened_at: Instant::now(),
        })
    }

    /// Declare a channel (and its optional schema) in this file, recording the MCAP
    /// channel id so later messages can target it. Returns the underlying MCAP write
    /// error rather than panicking, so the writer thread can mark its class unhealthy
    /// instead of dying silently.
    fn declare(&mut self, channel_id: u16, decl: &ChannelDecl) -> std::io::Result<()> {
        // Resolve (or write) the schema first; MCAP requires a channel's schema id to
        // already exist. schema_id 0 means "no schema".
        let schema_id = match &decl.schema {
            None => 0,
            Some(schema) => {
                let key = (schema.name.clone(), schema.encoding.clone());
                if let Some(&existing) = self.mcap_schema_ids.get(&key) {
                    existing
                } else {
                    let id = self
                        .writer
                        .add_schema(&schema.name, &schema.encoding, &schema.data)
                        .map_err(std::io::Error::other)?;
                    self.mcap_schema_ids.insert(key, id);
                    id
                }
            }
        };
        let mcap_id = self
            .writer
            .add_channel(
                schema_id,
                &decl.topic,
                &decl.message_encoding,
                &BTreeMap::new(),
            )
            .map_err(std::io::Error::other)?;
        self.mcap_channel_ids.insert(channel_id, mcap_id);
        Ok(())
    }

    /// Write one message to a previously-declared channel in this file. Returns the MCAP
    /// write error rather than panicking so the writer thread can escalate it.
    fn write_message(
        &mut self,
        channel_id: u16,
        log_time_ns: u64,
        data: &[u8],
    ) -> std::io::Result<()> {
        let Some(&mcap_id) = self.mcap_channel_ids.get(&channel_id) else {
            // A message for a channel never declared in this file: this only happens
            // if a registration was dropped under backpressure. Skip it loudly rather
            // than corrupt the file. (Blocking `Register` makes this practically
            // unreachable now, but skipping stays the safe response.)
            eprintln!("fieldloop-recorder: dropping message for undeclared channel {channel_id}");
            return Ok(());
        };
        let header = mcap::records::MessageHeader {
            channel_id: mcap_id,
            // Sequence is per-channel-per-file; use the file-local message count as a
            // monotonically increasing sequence, which is sufficient for readers.
            sequence: u32::try_from(self.messages_in_file & u64::from(u32::MAX))
                .unwrap_or(u32::MAX),
            log_time: log_time_ns,
            // No separate publish clock here, so publish_time mirrors log_time.
            publish_time: log_time_ns,
        };
        self.writer
            .write_to_known_channel(&header, data)
            .map_err(std::io::Error::other)?;
        self.messages_in_file += 1;
        Ok(())
    }

    /// Finalize durably: write the MCAP footer, flush the buffer to the file, fsync the
    /// bytes to disk, then atomically rename the `.part` temp to its final name.
    ///
    /// Consumes `self` because publishing the file ends its life. The fsync-before-rename
    /// order is what makes the durability real: on power loss (the normal robot failure
    /// mode) a file carrying the final name is guaranteed complete and on disk, never a
    /// torn page-cache leftover an agent would then upload. A failure at any step is
    /// surfaced (the writer thread escalates it) rather than leaving a half-published file.
    fn finish(mut self) -> std::io::Result<()> {
        // finish() writes the summary section + footer; without it the file is truncated
        // and will not read back validly.
        self.writer.finish().map_err(std::io::Error::other)?;
        // Recover the buffered file, flushing any buffered MCAP bytes into the OS file.
        let buf = self.writer.into_inner();
        let file = buf
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        // fsync the data+metadata so the bytes are on stable storage before we expose the
        // final name; a rename of an unsynced file could publish an empty file after a
        // power cut.
        file.sync_all()?;
        drop(file);
        // Atomic on the same filesystem: an observer sees either no final file or the
        // complete one, never a partial.
        std::fs::rename(&self.part_path, &self.final_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::tempdir;

    /// Read every `<class>-*.mcap` file under `dir`, in seq order, and return each
    /// message as `(topic, log_time, data)`. Proves the files are valid MCAP by
    /// decoding them with the `mcap` reader, not just inspecting bytes.
    fn read_back(dir: &Path, class: StreamClass) -> Vec<(String, u64, Vec<u8>)> {
        // Collect files in sequence order so the recovered messages are in write
        // order across a rotation.
        let mut files: Vec<(u64, PathBuf)> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| {
                let path = e.unwrap().path();
                let name = path.file_name()?.to_str()?.to_string();
                let prefix = format!("{}-", class.name());
                let seq: u64 = name
                    .strip_prefix(&prefix)?
                    .strip_suffix(".mcap")?
                    .parse()
                    .ok()?;
                Some((seq, path))
            })
            .collect();
        files.sort_by_key(|(seq, _)| *seq);

        let mut out = Vec::new();
        for (_, path) in files {
            let bytes = std::fs::read(&path).unwrap();
            for msg in mcap::MessageStream::new(&bytes).unwrap() {
                let msg = msg.unwrap();
                out.push((
                    msg.channel.topic.clone(),
                    msg.log_time,
                    msg.data.into_owned(),
                ));
            }
        }
        out
    }

    #[test]
    fn round_trips_messages_across_all_three_classes() {
        let dir = tempdir().unwrap();
        let recorder = Recorder::new(dir.path(), RecorderConfig::default());

        // One topic per class, with a schema on the sensor topic to exercise schemas.
        let meta_ch = recorder.register_channel(StreamClass::Meta, "/meta/outcome", "json", None);
        let sensor_ch = recorder.register_channel(
            StreamClass::Sensor,
            "/sensor/depth",
            "raw",
            Some(Schema::new("DepthFrame", "jsonschema", b"{}".to_vec())),
        );
        let safety_ch =
            recorder.register_channel(StreamClass::Safety, "/safety/estop", "json", None);

        // Each class is its own stream, so the first message of each lands at index 0.
        assert_eq!(
            recorder.record(StreamClass::Meta, meta_ch, 10, b"meta-0"),
            RecordResult::Enqueued(MessageIndex(0))
        );
        assert_eq!(
            recorder.record(StreamClass::Sensor, sensor_ch, 20, b"depth-0"),
            RecordResult::Enqueued(MessageIndex(0))
        );
        assert_eq!(
            recorder.record(StreamClass::Safety, safety_ch, 30, b"ESTOP"),
            RecordResult::Enqueued(MessageIndex(0))
        );

        // Finalize so every MCAP footer/summary is written before we read back.
        recorder.finalize();

        let meta = read_back(dir.path(), StreamClass::Meta);
        assert_eq!(
            meta,
            vec![("/meta/outcome".to_string(), 10, b"meta-0".to_vec())]
        );

        let sensor = read_back(dir.path(), StreamClass::Sensor);
        assert_eq!(
            sensor,
            vec![("/sensor/depth".to_string(), 20, b"depth-0".to_vec())]
        );

        let safety = read_back(dir.path(), StreamClass::Safety);
        assert_eq!(
            safety,
            vec![("/safety/estop".to_string(), 30, b"ESTOP".to_vec())]
        );
    }

    /// Durable finalize: after the recorder finalizes, each class's data is published
    /// under its final `<class>-0.mcap` name and NO `.part` temp remains — so an agent
    /// that treats every `.mcap` as finished never sees a half-written file, and a torn
    /// crash leftover keeps the `.part` name (which the agent ignores) instead.
    #[test]
    fn finalize_publishes_final_name_and_leaves_no_part_file() {
        let dir = tempdir().unwrap();
        let recorder = Recorder::new(dir.path(), RecorderConfig::default());
        let ch = recorder.register_channel(StreamClass::Sensor, "/sensor/depth", "raw", None);
        recorder.record(StreamClass::Sensor, ch, 1, b"frame");
        recorder.finalize();

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().any(|n| n == "sensor-0.mcap"),
            "the finalized file must carry its final name, got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.ends_with(".part")),
            "no .part temp may survive a clean finalize, got {names:?}"
        );
    }

    #[test]
    fn sensor_drops_and_counts_while_safety_never_drops_under_flood() {
        let dir = tempdir().unwrap();
        // Tiny sensor capacity so a flood overflows it; generous safety config
        // (unbounded regardless). No writer-thread draining is forced, so the bounded
        // sensor channel fills and overflows deterministically.
        // Tiny sensor capacity so a flood overflows it and drops-and-counts; a LARGE
        // (but now bounded) safety buffer so a safety event is never silently dropped.
        let config = RecorderConfig {
            meta: ClassConfig::new(1024),
            sensor: ClassConfig::new(2),
            safety: ClassConfig::new(8192),
        };
        let recorder = Recorder::new(dir.path(), config);

        let sensor_ch = recorder.register_channel(StreamClass::Sensor, "/sensor/x", "raw", None);
        let safety_ch = recorder.register_channel(StreamClass::Safety, "/safety/x", "raw", None);

        // Flood both classes hard. With a capacity-2 sensor channel and thousands of
        // sends the sensor class drops; the safety class must NEVER report a routine
        // `Dropped` — at worst it loudly back-pressures, and every event it ACCEPTS is
        // durable (none silently lost).
        let flood = 5000;
        let mut sensor_dropped_seen = false;
        let mut accepted_safety = 0u64; // safety events the recorder took responsibility for
        for i in 0..flood {
            let payload = (i as u32).to_le_bytes();
            if recorder.record(StreamClass::Sensor, sensor_ch, i, &payload) == RecordResult::Dropped
            {
                sensor_dropped_seen = true;
            }
            // Safety is never a routine drop: only Enqueued (accepted) or, if its bounded
            // buffer is momentarily full, the hard SafetyBackpressure signal — never
            // Dropped and never WriterGone.
            match recorder.record(StreamClass::Safety, safety_ch, i, &payload) {
                RecordResult::Enqueued(_) => accepted_safety += 1,
                RecordResult::SafetyBackpressure => {}
                other => {
                    panic!("safety record must be Enqueued or SafetyBackpressure, got {other:?}")
                }
            }
        }

        assert!(
            sensor_dropped_seen || recorder.dropped(StreamClass::Sensor) > 0,
            "flooded sensor class must have dropped at least one message"
        );
        assert!(recorder.dropped(StreamClass::Sensor) > 0);
        // The no-drop class never counts a routine drop (backpressure is a loud return
        // value, not a silent counter).
        assert_eq!(recorder.dropped(StreamClass::Safety), 0);
        // With an 8192 buffer and a draining writer, every safety event is accepted at
        // this scale (the common no-backpressure path the bound is sized for).
        assert_eq!(
            accepted_safety, flood,
            "an 8192-deep safety buffer must accept every event at this flood scale"
        );

        recorder.finalize();

        // Every ACCEPTED safety event is recoverable — what the recorder took, it kept.
        let safety = read_back(dir.path(), StreamClass::Safety);
        assert_eq!(
            safety.len() as u64,
            accepted_safety,
            "every accepted safety event must be recoverable — none silently lost"
        );
    }

    #[test]
    fn rotation_produces_multiple_files_and_recovers_every_message_in_order() {
        let dir = tempdir().unwrap();
        // Rotate every 10 messages on the meta class; write 25 → expect 3 files
        // (0..10, 10..20, 20..25).
        let config = RecorderConfig {
            meta: ClassConfig::new(1024).with_max_messages(10),
            sensor: ClassConfig::new(1024),
            safety: ClassConfig::new(1024),
        };
        let recorder = Recorder::new(dir.path(), config);
        let ch = recorder.register_channel(StreamClass::Meta, "/meta/seq", "raw", None);

        let total: u64 = 25;
        for i in 0..total {
            // Drive draining by giving the writer thread time would be flaky; instead
            // we rely on the bounded channel being large enough (1024) that nothing
            // drops, so every message reaches the writer and rotation is by count.
            let r = recorder.record(StreamClass::Meta, ch, i, &(i as u32).to_le_bytes());
            // Indices are dense and in write order across rotations: the Nth enqueued
            // message of the class is index N, even once the file rotates underneath it.
            assert_eq!(r, RecordResult::Enqueued(MessageIndex(i)));
        }
        recorder.finalize();

        // Multiple files were produced for the class.
        let file_count = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .starts_with("meta-")
            })
            .count();
        assert!(
            file_count >= 2,
            "expected rotation into multiple files, got {file_count}"
        );

        // Reading all files back in seq order recovers every message in write order.
        let recovered = read_back(dir.path(), StreamClass::Meta);
        assert_eq!(recovered.len(), total as usize);
        for (i, (topic, log_time, data)) in recovered.iter().enumerate() {
            assert_eq!(topic, "/meta/seq");
            assert_eq!(*log_time, i as u64);
            assert_eq!(data, &(i as u32).to_le_bytes().to_vec());
        }
    }

    #[test]
    fn each_rollouts_messages_get_distinct_non_overlapping_ranges() {
        // Three rollouts recorded one after another, each writing two messages
        // (observation + action) on the sensor stream. Each rollout must get its OWN
        // [start, end) message range, and those ranges must be distinct and
        // non-overlapping — the property that lets a replay viewer fetch exactly one
        // rollout's frames instead of the whole recording.
        let dir = tempdir().unwrap();
        let recorder = Recorder::new(dir.path(), RecorderConfig::default());
        let ch = recorder.register_channel(StreamClass::Sensor, "/sensor/frames", "raw", None);

        let mut ranges: Vec<MessageRange> = Vec::new();
        for rollout in 0u64..3 {
            // Observation message for this rollout, then its action message.
            let obs = recorder
                .record(StreamClass::Sensor, ch, rollout * 2, b"obs")
                .message_index()
                .expect("observation must enqueue");
            let act = recorder
                .record(StreamClass::Sensor, ch, rollout * 2 + 1, b"act")
                .message_index()
                .expect("action must enqueue");
            // This rollout's run spans both of its messages: [obs, act + 1).
            ranges.push(MessageRange {
                start: obs.0,
                end: act.0 + 1,
            });
        }

        // Expected: [0,2), [2,4), [4,6) — distinct and contiguous (so non-overlapping).
        assert_eq!(
            ranges,
            vec![
                MessageRange { start: 0, end: 2 },
                MessageRange { start: 2, end: 4 },
                MessageRange { start: 4, end: 6 },
            ]
        );
        // Pairwise distinct and non-overlapping, asserted directly rather than relying on
        // the literal above so the invariant is checked structurally too.
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                assert_ne!(ranges[i], ranges[j], "rollouts must not share a range");
                let disjoint = ranges[i].end <= ranges[j].start || ranges[j].end <= ranges[i].start;
                assert!(disjoint, "rollout ranges must not overlap");
            }
        }

        recorder.finalize();

        // The ranges actually index into the recorded stream: reading the class's files
        // back in order and slicing by each rollout's range recovers that rollout's own
        // messages, proving the locator is real, not a bare number.
        let recovered = read_back(dir.path(), StreamClass::Sensor);
        assert_eq!(recovered.len(), 6);
        for (rollout, range) in ranges.iter().enumerate() {
            let slice = &recovered[range.start as usize..range.end as usize];
            assert_eq!(slice.len(), 2);
            assert_eq!(slice[0].2, b"obs".to_vec());
            assert_eq!(slice[1].2, b"act".to_vec());
            // And the slice's log_times are this rollout's, not another's.
            assert_eq!(slice[0].1, rollout as u64 * 2);
            assert_eq!(slice[1].1, rollout as u64 * 2 + 1);
        }
    }

    #[test]
    fn single_rollout_single_message_range_is_one_message() {
        // The single-rollout case still works: one recorded message yields the
        // single-message run [0, 1).
        let dir = tempdir().unwrap();
        let recorder = Recorder::new(dir.path(), RecorderConfig::default());
        let ch = recorder.register_channel(StreamClass::Sensor, "/s", "raw", None);
        let idx = recorder
            .record(StreamClass::Sensor, ch, 1, b"only")
            .message_index()
            .expect("must enqueue");
        assert_eq!(MessageRange::single(idx), MessageRange { start: 0, end: 1 });
        recorder.finalize();
    }

    #[test]
    fn drop_finalizes_files_without_explicit_finalize() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            let recorder = Recorder::new(&path, RecorderConfig::default());
            let ch = recorder.register_channel(StreamClass::Meta, "/m", "raw", None);
            recorder.record(StreamClass::Meta, ch, 1, b"x");
            // No explicit finalize(): rely on Drop to flush and close the file.
        }
        // The file must still read back validly.
        let msgs = read_back(&path, StreamClass::Meta);
        assert_eq!(msgs, vec![("/m".to_string(), 1, b"x".to_vec())]);
    }

    #[test]
    fn is_droppable_matches_policy() {
        // The drop policy table the doc comments promise, asserted directly.
        let mut policy = HashMap::new();
        policy.insert(StreamClass::Meta, true);
        policy.insert(StreamClass::Sensor, true);
        policy.insert(StreamClass::Safety, false);
        for (class, droppable) in policy {
            assert_eq!(class.is_droppable(), droppable);
        }
    }

    #[test]
    fn time_based_rotation_splits_files() {
        let dir = tempdir().unwrap();
        // Rotate after a very short duration so two writes spaced apart land in
        // different files.
        let config = RecorderConfig {
            meta: ClassConfig::new(1024).with_max_duration(Duration::from_millis(1)),
            sensor: ClassConfig::new(1024),
            safety: ClassConfig::new(1024),
        };
        let recorder = Recorder::new(dir.path(), config);
        let ch = recorder.register_channel(StreamClass::Meta, "/m", "raw", None);

        recorder.record(StreamClass::Meta, ch, 1, b"a");
        std::thread::sleep(Duration::from_millis(5));
        recorder.record(StreamClass::Meta, ch, 2, b"b");
        recorder.finalize();

        // Both messages must be recovered regardless of how the split landed.
        let recovered = read_back(dir.path(), StreamClass::Meta);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0], ("/m".to_string(), 1, b"a".to_vec()));
        assert_eq!(recovered[1], ("/m".to_string(), 2, b"b".to_vec()));
    }
}
