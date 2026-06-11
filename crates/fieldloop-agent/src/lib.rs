//! On-robot sidecar that ships *finished* recorded files to the cloud — store-and-forward,
//! resumable, safety-first, never dropping a file.
//!
//! Runs as a separate background service so the control loop and recorder never touch the network:
//! the agent owns the network (upload, retry, resume); the robot just records bytes to local files.
//!
//! Contract (each property exists because the field link is unreliable and robots restart):
//! - **Never drops a file** — a finished file stays pending and is retried every sync until the
//!   cloud confirms the whole thing arrived (vs the recorder's droppable sensor lane).
//! - **Resumable** — a partial transfer records the bytes already received and the next attempt
//!   resumes from that offset, so a 2 GB file never restarts from zero on a link blip.
//! - **Safety-first** — files upload in priority order (safety events → metadata → bulk payloads),
//!   so a flood of sensor video can never starve a safety record of bandwidth.
//!
//! Survives crash/power-cycle: progress lives in a persisted JSON manifest written after every file,
//! so a fresh agent reads back exactly what is done and where each in-flight upload resumes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use fieldloop_types::Rollout;

pub mod batch;
pub use batch::{UploadedBlob, build_ingest_request_json, sha256_hex};

// The real S3-compatible uploader lives behind the optional `s3` feature so the default
// agent build (and the offline closed-loop gate) never compiles the AWS SDK and never
// opens a socket. Production and the live e2e test enable `s3` to ship bytes for real.
#[cfg(feature = "s3")]
pub mod s3;
#[cfg(feature = "s3")]
pub use s3::S3Uploader;

/// The class of a recorded stream, recovered from a finished file's name.
///
/// The recorder writes files named `<class>-<seq>.mcap` where the prefix is one
/// of these classes, so the agent infers a file's class — and therefore its
/// upload priority — straight from the filename without opening the file.
///
/// The ordering of the variants is the upload priority order: `Safety` is most
/// important and is uploaded first, then `Meta`, then bulk `Sensor` payloads.
/// Deriving `Ord` in this declared order is what lets a simple sort put safety
/// records ahead of everything else on a thin uplink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamClass {
    /// E-stops and other safety events: smallest and most important, so they are
    /// uploaded before anything else can consume the link's bandwidth.
    Safety,
    /// Rollout / outcome metadata: important and small, uploaded after safety but
    /// ahead of bulk sensor payloads.
    Meta,
    /// High-volume payloads (video / lidar / depth): uploaded last so a flood of
    /// sensor data can never delay the small, important classes above.
    Sensor,
}

impl StreamClass {
    /// Parse the class from a filename prefix `<class>-<seq>.mcap`, matching the
    /// names the recorder writes, so the agent and recorder agree on the prefix
    /// without sharing code. Returns `None` for an unrecognized prefix so a
    /// stray file in the directory is ignored rather than mis-prioritized.
    #[must_use]
    pub fn from_filename(name: &str) -> Option<StreamClass> {
        // Split on the first '-' so `safety-12.mcap` yields the `safety` prefix.
        let prefix = name.split('-').next()?;
        match prefix {
            "safety" => Some(StreamClass::Safety),
            "meta" => Some(StreamClass::Meta),
            "sensor" => Some(StreamClass::Sensor),
            _ => None,
        }
    }
}

/// The result of attempting to upload (a remaining slice of) a file.
///
/// The three variants distinguish the cases the agent must treat differently so
/// it can never lose data: a clean finish, a partial transfer worth resuming, and
/// a transfer that made no progress and should simply be retried later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadOutcome {
    /// The whole file (from the requested start offset to the end) reached the
    /// cloud. The agent can mark the file done and never touch it again.
    Done,
    /// A mid-transfer disconnect got partway: the cloud now holds bytes up to
    /// `uploaded_to`. The agent records this offset and resumes from it next sync,
    /// so the already-sent prefix is not re-sent.
    Interrupted {
        /// Absolute byte offset within the file that the cloud has received. The
        /// next attempt passes this as its `start_offset` to continue mid-file.
        uploaded_to: u64,
    },
    /// The attempt could not make any progress (e.g. the link is fully down). The
    /// file stays pending at its existing offset and is retried on a later sync;
    /// it is never dropped.
    Failed,
}

/// Abstraction over the cloud destination, so the agent's store-and-forward logic
/// is independent of the real transport (S3, HTTP, ...), and tests can drive it
/// with a fake that deterministically interrupts or refuses uploads.
pub trait Uploader {
    /// Upload `data[start_offset..]` to the object named `key`, returning whether
    /// it finished, got interrupted partway, or failed to make progress.
    ///
    /// `start_offset` lets a resumed upload continue mid-file: on a fresh upload
    /// it is `0`, and after an [`UploadOutcome::Interrupted`] the next call passes
    /// the reported offset so the cloud appends rather than restarts. `key`
    /// identifies the object and is stable across attempts for the same file, so a
    /// resume targets the same object the earlier attempt was building.
    fn upload(&self, key: &str, data: &[u8], start_offset: u64) -> UploadOutcome;
}

/// A finished file discovered by a [`FileSource`], ready to be uploaded.
///
/// "Finished" is the key word: a source only yields files the recorder has fully
/// written and closed, never a file still being appended to, so the agent never
/// uploads a torn, half-written file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundFile {
    /// Absolute path to the finished file on local disk.
    pub path: PathBuf,
    /// The file's stream class, parsed from its name, which drives upload priority.
    pub class: StreamClass,
    /// The file's size in bytes at discovery time.
    pub size: u64,
}

/// Source of *finished* recorded files for the agent to upload.
///
/// It is a trait so the production directory scanner and a deterministic
/// in-memory test source can be swapped freely: the agent's upload, retry, and
/// resume logic is then exercised without any dependency on filesystem timing.
pub trait FileSource {
    /// Return the finished files currently available to upload. A file still
    /// being written must never appear here — see each implementation for how it
    /// decides a file is finished.
    fn finished_files(&self) -> Vec<FoundFile>;
}

/// A [`FileSource`] backed by a fixed list, for deterministic tests.
///
/// It hands back exactly the files it was constructed with, so the agent's
/// priority/resume/retry behavior can be tested without any filesystem-timing
/// flakiness from a real directory scan.
pub struct VecSource {
    files: Vec<FoundFile>,
}

impl VecSource {
    /// Wrap a fixed list of finished files.
    #[must_use]
    pub fn new(files: Vec<FoundFile>) -> VecSource {
        VecSource { files }
    }
}

impl FileSource for VecSource {
    fn finished_files(&self) -> Vec<FoundFile> {
        self.files.clone()
    }
}

/// A [`FileSource`] that scans a directory for `*.mcap` files and reports only the
/// ones that have gone quiet — unchanged in size and modification time for a
/// configurable quiet period.
///
/// The quiet-period check is how it distinguishes a finished file from one the
/// recorder is still appending to: a file the recorder just wrote to has a recent
/// mtime, so it is held back until it has been stable long enough, and only then
/// uploaded. Without this, the agent could ship a file mid-write and send a torn,
/// half-written object to the cloud.
///
/// The clock is injected as a `now` closure so a test can advance time
/// deterministically instead of sleeping, keeping the finished-only behavior
/// testable without real wall-clock waits.
pub struct DirSource {
    dir: PathBuf,
    quiet_period: Duration,
    now: Box<dyn Fn() -> SystemTime + Send + Sync>,
}

impl DirSource {
    /// Scan `dir`, treating a file as finished once it has been unchanged for
    /// `quiet_period`, using the real system clock.
    #[must_use]
    pub fn new(dir: impl AsRef<Path>, quiet_period: Duration) -> DirSource {
        DirSource {
            dir: dir.as_ref().to_path_buf(),
            quiet_period,
            now: Box::new(SystemTime::now),
        }
    }

    /// Same as [`DirSource::new`] but with an injected clock, so a test can decide
    /// exactly what "now" is and thus whether a file has gone quiet — making the
    /// finished-only behavior deterministic instead of timing-dependent.
    #[must_use]
    pub fn with_clock(
        dir: impl AsRef<Path>,
        quiet_period: Duration,
        now: impl Fn() -> SystemTime + Send + Sync + 'static,
    ) -> DirSource {
        DirSource {
            dir: dir.as_ref().to_path_buf(),
            quiet_period,
            now: Box::new(now),
        }
    }

    /// Whether a file last modified at `mtime` and currently `now` has gone quiet,
    /// i.e. has been untouched for at least the quiet period. A file the recorder
    /// just wrote has a recent mtime and so is not yet finished.
    fn is_quiet(&self, mtime: SystemTime, now: SystemTime) -> bool {
        match now.duration_since(mtime) {
            // Untouched for at least the quiet period: treat it as finished.
            Ok(age) => age >= self.quiet_period,
            // mtime is in the future relative to our clock (clock skew): not yet
            // safe to call finished, so hold it back.
            Err(_) => false,
        }
    }
}

impl FileSource for DirSource {
    fn finished_files(&self) -> Vec<FoundFile> {
        let now = (self.now)();
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.dir) else {
            // A missing or unreadable directory just means nothing to upload yet;
            // the next sync will try again rather than crash the agent.
            return out;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Only `*.mcap` files are recorded outputs; ignore anything else.
            if !name.ends_with(".mcap") {
                continue;
            }
            let Some(class) = StreamClass::from_filename(name) else {
                // An mcap file whose prefix is not a known class is not something
                // the recorder produced; skip it rather than guess a priority.
                continue;
            };
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            let Ok(mtime) = meta.modified() else {
                continue;
            };
            // Hold back a file that has been touched recently: it may still be
            // mid-write, so it is not yet safe to upload.
            if !self.is_quiet(mtime, now) {
                continue;
            }
            // Defense in depth against uploading a torn file: a finalized MCAP ends with
            // the MCAP end magic. A file that is quiet (mtime-stable) but lacks the end
            // magic is a crash-torn leftover, not a finished recording — skip it rather
            // than ship a truncated file. (The recorder's `.part`->rename finalize already
            // keeps torn files out of the `.mcap` namespace; this also covers files from a
            // recorder predating that change, and a crash between footer write and rename.)
            if !has_mcap_end_magic(&path) {
                continue;
            }
            out.push(FoundFile {
                path,
                class,
                size: meta.len(),
            });
        }
        out
    }
}

/// The 8-byte MCAP magic that bookends a valid file. A finalized MCAP ends with these
/// exact bytes after its footer, so their presence at end-of-file is a cheap, reliable
/// "the writer finished this file" signal.
const MCAP_MAGIC: [u8; 8] = [0x89, b'M', b'C', b'A', b'P', 0x30, b'\r', b'\n'];

/// Whether the file at `path` ends with the MCAP magic — i.e. was fully written and
/// finalized, not torn off mid-write by a crash.
///
/// Reads only the last 8 bytes (one `seek` + one `read`), so it is cheap even for a
/// multi-GB recording. A file shorter than the magic, or unreadable, or whose tail does
/// not match, is treated as not-finished so a torn leftover is never uploaded.
fn has_mcap_end_magic(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let Ok(len) = file.metadata().map(|m| m.len()) else {
        return false;
    };
    if len < MCAP_MAGIC.len() as u64 {
        return false;
    }
    if file
        .seek(SeekFrom::End(-(MCAP_MAGIC.len() as i64)))
        .is_err()
    {
        return false;
    }
    let mut tail = [0u8; 8];
    if file.read_exact(&mut tail).is_err() {
        return false;
    }
    tail == MCAP_MAGIC
}

/// Upload state of a single file in the persisted manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileStatus {
    /// Not yet fully uploaded. Combined with `uploaded_offset`, this is enough to
    /// resume: a pending file with offset N has its first N bytes already in the
    /// cloud and the next sync continues from N. Pending files are retried on
    /// every sync and are never dropped.
    Pending,
    /// Fully uploaded and confirmed by the cloud. A done file is never read or
    /// uploaded again, even after the agent restarts.
    Done,
}

/// Per-file persisted record: enough to skip completed files and resume in-flight
/// ones after a crash or restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Whether this file is still pending or fully done.
    pub status: FileStatus,
    /// Absolute byte offset already confirmed in the cloud. A resume passes this
    /// as the upload's start offset so the already-sent prefix is not re-sent.
    pub uploaded_offset: u64,
}

/// The persisted manifest: a map from each file's stable upload key to its
/// [`FileRecord`].
///
/// It is keyed by the upload key (derived deterministically from the filename) so
/// that the same file maps to the same entry across restarts, and it is written
/// to a JSON file after every file is processed so a crash mid-sync loses nothing:
/// a fresh agent reads it back and continues exactly where it left off.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct Manifest {
    /// Map of upload key -> per-file progress. `BTreeMap` so the on-disk JSON has a
    /// stable key order, making the manifest easy to diff and inspect by hand.
    files: BTreeMap<String, FileRecord>,
}

impl Manifest {
    /// Load the manifest from `path`, or start empty if it does not exist yet.
    ///
    /// A first run (no manifest yet) is normal, so a missing file is not an error
    /// — it just means nothing has been uploaded yet. A corrupt manifest is a
    /// real problem and is surfaced as an error rather than silently discarded,
    /// because silently starting empty would re-upload everything.
    fn load(path: &Path) -> std::io::Result<Manifest> {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(std::io::Error::other),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Manifest::default()),
            Err(e) => Err(e),
        }
    }

    /// Persist the manifest to `path` durably-ish: write to a temp file in the same
    /// directory then atomically rename over the target, so a crash mid-write can
    /// never leave a half-written, unparseable manifest — the old one stays intact
    /// until the new one is complete.
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, json)?;
        // Rename is atomic on the same filesystem, so a reader (a restarting
        // agent) ever sees either the old complete manifest or the new complete
        // one, never a torn write.
        fs::rename(&tmp, path)
    }
}

/// Tracks how far the agent has registered the append-only rollouts file, so each pass
/// POSTs only the NEW lines as one delta batch.
///
/// Without this the agent re-read the whole growing `rollouts.jsonl` every pass and hashed
/// ALL ids into a fresh `batch_id`: because the set grows between passes, every pass minted
/// a new batch id over an overlapping set, the gateway's `(tenant, batch_id)` idempotency
/// gate saw a "new" batch each time, and ClickHouse (a plain append table) recorded the
/// same rollouts again and again. Anchoring on a persisted byte offset makes a pass POST
/// exactly the rollouts appended since the last acked offset; a retry before the offset
/// advances re-sends the identical delta under the identical `batch_id`, so the gateway
/// dedups it instead of double-recording.
pub struct RolloutCheckpoint {
    /// The sidecar file holding the last acked byte offset, next to the upload manifest.
    offset_path: PathBuf,
    /// Bytes of `rollouts.jsonl` already registered (and acked by the gateway).
    offset: u64,
}

impl RolloutCheckpoint {
    /// Load the checkpoint from `state_dir`, starting at offset 0 if none exists yet.
    ///
    /// A missing or unparseable offset file reads as 0 (register from the start) rather
    /// than erroring: 0 is the safe default — at worst the first batch re-sends rollouts
    /// the gateway then dedups by `batch_id`, never silent loss.
    #[must_use]
    pub fn load(state_dir: &Path) -> RolloutCheckpoint {
        let offset_path = state_dir.join("rollouts.offset");
        let offset = fs::read_to_string(&offset_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        RolloutCheckpoint {
            offset_path,
            offset,
        }
    }

    /// The byte offset registered so far.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Read the rollouts appended to `rollouts_path` since the committed offset, plus the
    /// new end offset to commit once the gateway acks them.
    ///
    /// Only COMPLETE lines (up to the last newline) are consumed; a trailing partial line —
    /// capture mid-write — is left for a later pass so a half-written rollout is never
    /// parsed. A missing file, or an offset at/after end (nothing appended, or the file was
    /// rotated away), yields an empty delta at the unchanged offset. The returned offset is
    /// NOT applied to `self` — the caller [`Self::commit`]s it only after a successful POST,
    /// so a crash between POST and commit re-sends the identical delta (idempotent).
    pub fn read_new(&self, rollouts_path: &Path) -> std::io::Result<(Vec<Rollout>, u64)> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = match fs::File::open(rollouts_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Vec::new(), self.offset));
            }
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        if self.offset >= len {
            // Nothing appended since the last commit (or the file shrank — a rotation we do
            // not chase here; the common case is append-only).
            return Ok((Vec::new(), self.offset));
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut tail = String::new();
        file.read_to_string(&mut tail)?;
        let Some(last_nl) = tail.rfind('\n') else {
            // Bytes present but no complete line yet (mid-write): consume nothing.
            return Ok((Vec::new(), self.offset));
        };
        let complete = &tail[..=last_nl];
        let mut rollouts = Vec::new();
        for line in complete.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let rollout: Rollout = serde_json::from_str(line).map_err(std::io::Error::other)?;
            rollouts.push(rollout);
        }
        let new_offset = self.offset + complete.len() as u64;
        Ok((rollouts, new_offset))
    }

    /// Commit `new_offset` durably (temp file + atomic rename, like the manifest), advancing
    /// the in-memory offset only after the on-disk write lands.
    ///
    /// Called only AFTER the gateway acked the delta, so the offset advances past rollouts
    /// that are safely registered; a crash before this leaves the old offset and the same
    /// delta re-sends under the same `batch_id`.
    pub fn commit(&mut self, new_offset: u64) -> std::io::Result<()> {
        let tmp = self.offset_path.with_extension("offset.tmp");
        fs::write(&tmp, new_offset.to_string())?;
        fs::rename(&tmp, &self.offset_path)?;
        self.offset = new_offset;
        Ok(())
    }
}

/// A snapshot of agent progress, returned by [`Agent::sync_once`] so a caller (or
/// a test) can observe how much work is left and how much has been shipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncStats {
    /// Number of known files still pending (not yet fully uploaded).
    pub pending: usize,
    /// Number of files fully uploaded and confirmed.
    pub done: usize,
    /// Total bytes confirmed in the cloud across all files (sum of every file's
    /// `uploaded_offset`), so a partially-uploaded file counts its sent prefix.
    pub bytes_uploaded: u64,
}

/// The sidecar agent: owns an [`Uploader`], a [`FileSource`], and a persisted
/// manifest, and ships finished files Safety-first with resume and retry.
///
/// The agent is deliberately a passive, single-method engine: a caller drives it
/// by calling [`Agent::sync_once`] on a loop (e.g. from a systemd service). Each
/// call discovers finished files, skips ones already done, uploads the rest in
/// priority order, and persists progress after every file — so the loop can be
/// interrupted at any point and resumed without losing or duplicating work.
pub struct Agent<U: Uploader, S: FileSource> {
    uploader: U,
    source: S,
    manifest_path: PathBuf,
    manifest: Manifest,
}

impl<U: Uploader, S: FileSource> Agent<U, S> {
    /// Build an agent, loading any existing manifest from `state_dir` so a restart
    /// resumes prior progress instead of re-uploading completed files.
    ///
    /// The state directory is created if missing. The manifest lives at
    /// `state_dir/manifest.json`; an existing one is read back here, which is the
    /// mechanism that makes the agent crash- and restart-safe.
    ///
    /// # Errors
    /// Returns an error if the state directory cannot be created or an existing
    /// manifest cannot be read/parsed — both are fatal because proceeding would
    /// risk re-uploading or losing files.
    pub fn new(
        uploader: U,
        source: S,
        state_dir: impl AsRef<Path>,
    ) -> std::io::Result<Agent<U, S>> {
        let state_dir = state_dir.as_ref();
        fs::create_dir_all(state_dir)?;
        let manifest_path = state_dir.join("manifest.json");
        let manifest = Manifest::load(&manifest_path)?;
        Ok(Agent {
            uploader,
            source,
            manifest_path,
            manifest,
        })
    }

    /// Derive the stable cloud object key for a file from its filename.
    ///
    /// The key must be deterministic so that a resumed upload after an interruption
    /// targets the exact same object the earlier attempt was building, and so the
    /// manifest entry for a file is found again across restarts. Using the
    /// filename keeps the key human-readable and unique within a recorder's output
    /// (the recorder names files `<class>-<seq>.mcap`).
    fn key_for(file: &FoundFile) -> String {
        file.path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or_else(|| file.path.to_string_lossy().into_owned(), String::from)
    }

    /// Run one synchronization pass: discover finished files, skip done ones,
    /// upload the rest Safety-first, and persist progress after each file.
    ///
    /// The flow, and why each step exists:
    /// 1. Discover finished files from the source (never a file mid-write).
    /// 2. Skip any already marked `Done` in the manifest, so a restart never
    ///    re-uploads a completed file.
    /// 3. Sort by priority — `Safety` before `Meta` before `Sensor`, and within a
    ///    class by filename (sequence/age) — so important records go first on a
    ///    thin uplink.
    /// 4. For each pending file, read its bytes and upload from its recorded
    ///    offset: on `Done` mark it done; on `Interrupted` save the new offset and
    ///    keep it pending so the next sync resumes; on `Failed` leave it pending to
    ///    retry later. A file is never dropped.
    /// 5. Persist the manifest after every file, so a crash mid-sync resumes
    ///    correctly rather than losing or duplicating an upload.
    ///
    /// Returns a [`SyncStats`] snapshot of pending/done counts and total bytes
    /// uploaded.
    ///
    /// # Errors
    /// Returns an error if the manifest cannot be persisted, since silently failing
    /// to record progress would break the no-drop / resume guarantees. A file that
    /// cannot be read is skipped (left pending) rather than failing the whole pass.
    pub fn sync_once(&mut self) -> std::io::Result<SyncStats> {
        let mut files = self.source.finished_files();
        // Priority order: by class first (Safety < Meta < Sensor via the derived
        // Ord), then by filename so that within a class lower sequence numbers
        // (older files) go first. Sorting by the full path's filename is a stable,
        // deterministic tiebreak.
        files.sort_by(|a, b| {
            a.class
                .cmp(&b.class)
                .then_with(|| Self::key_for(a).cmp(&Self::key_for(b)))
        });

        for file in &files {
            let key = Self::key_for(file);
            // Skip files already confirmed done so a restart never re-uploads them.
            if let Some(record) = self.manifest.files.get(&key)
                && record.status == FileStatus::Done
            {
                continue;
            }
            // Current resume offset for this file: 0 if we have never seen it.
            let start_offset = self
                .manifest
                .files
                .get(&key)
                .map_or(0, |r| r.uploaded_offset);

            // Read the file's bytes. If it cannot be read right now, leave it
            // pending (recorded with its current offset) so it is retried next
            // sync — a transient read error must never drop the file.
            let data = match fs::read(&file.path) {
                Ok(data) => data,
                Err(_) => {
                    self.manifest
                        .files
                        .entry(key.clone())
                        .or_insert(FileRecord {
                            status: FileStatus::Pending,
                            uploaded_offset: start_offset,
                        });
                    self.manifest.save(&self.manifest_path)?;
                    continue;
                }
            };

            let outcome = self.uploader.upload(&key, &data, start_offset);
            let record = match outcome {
                // Whole file confirmed: mark done with the full length as the
                // uploaded offset so the bytes-uploaded stat reflects the complete
                // file.
                UploadOutcome::Done => FileRecord {
                    status: FileStatus::Done,
                    uploaded_offset: data.len() as u64,
                },
                // Partial transfer: keep pending but advance the offset so the next
                // sync resumes from exactly where the cloud got to.
                UploadOutcome::Interrupted { uploaded_to } => FileRecord {
                    status: FileStatus::Pending,
                    uploaded_offset: uploaded_to,
                },
                // No progress: keep pending at the existing offset and retry later.
                // The file is not dropped.
                UploadOutcome::Failed => FileRecord {
                    status: FileStatus::Pending,
                    uploaded_offset: start_offset,
                },
            };
            self.manifest.files.insert(key, record);
            // Persist after every file so a crash here resumes correctly: the
            // manifest always reflects exactly what the cloud has confirmed.
            self.manifest.save(&self.manifest_path)?;
        }

        Ok(self.stats())
    }

    /// Compute the current pending/done/bytes snapshot from the manifest.
    fn stats(&self) -> SyncStats {
        let mut pending = 0;
        let mut done = 0;
        let mut bytes_uploaded = 0;
        for record in self.manifest.files.values() {
            match record.status {
                FileStatus::Pending => pending += 1,
                FileStatus::Done => done += 1,
            }
            bytes_uploaded += record.uploaded_offset;
        }
        SyncStats {
            pending,
            done,
            bytes_uploaded,
        }
    }

    /// Current progress snapshot without running a sync, so a caller can poll
    /// counts (e.g. for telemetry) between sync passes.
    #[must_use]
    pub fn stats_now(&self) -> SyncStats {
        self.stats()
    }

    /// The upload keys of every file the cloud has *confirmed* fully received
    /// (manifest status `Done`).
    ///
    /// This exists so the deployable binary can build the gateway's blob claims
    /// from only the bytes that genuinely landed: a blob pointer must describe an
    /// object the gateway can HEAD in the bucket, so it must come from a confirmed
    /// `Done` upload, never from a file that is still pending or only partway up.
    /// Returning just the keys (not bytes) keeps the manifest internals private
    /// while letting the caller re-read the corresponding local file to compute the
    /// checksum/size of exactly the bytes that were uploaded.
    #[must_use]
    pub fn done_keys(&self) -> Vec<String> {
        self.manifest
            .files
            .iter()
            .filter(|(_, record)| record.status == FileStatus::Done)
            .map(|(key, _)| key.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// A test [`Uploader`] that records what it received and can be configured to
    /// interrupt the first attempt after N bytes, or to refuse all uploads until a
    /// flag flips — so resume and retry can be exercised deterministically without
    /// a real network.
    struct FakeUploader {
        /// Received bytes per key, in upload order, so a test can assert the cloud
        /// got the whole file exactly once with no gaps or duplication.
        received: Mutex<BTreeMap<String, Vec<u8>>>,
        /// If set, the first upload attempt for any key delivers only this many
        /// bytes (from its start offset) and reports `Interrupted`, to exercise
        /// resume. Subsequent attempts deliver the rest.
        interrupt_after: Option<u64>,
        /// Keys already attempted once, so `interrupt_after` only fires on the
        /// first attempt and the resume attempt completes.
        attempted: Mutex<std::collections::BTreeSet<String>>,
        /// While true, every upload reports `Failed` (the link is down), to
        /// exercise no-drop retry. A test flips it false to "reconnect".
        offline: Mutex<bool>,
    }

    impl FakeUploader {
        fn new() -> FakeUploader {
            FakeUploader {
                received: Mutex::new(BTreeMap::new()),
                interrupt_after: None,
                attempted: Mutex::new(std::collections::BTreeSet::new()),
                offline: Mutex::new(false),
            }
        }

        fn interrupting(after: u64) -> FakeUploader {
            FakeUploader {
                interrupt_after: Some(after),
                ..FakeUploader::new()
            }
        }

        fn offline() -> FakeUploader {
            let u = FakeUploader::new();
            *u.offline.lock().unwrap() = true;
            u
        }

        /// The full bytes the cloud holds for a key (the concatenation of every
        /// accepted slice), used to assert no gap and no duplication.
        fn bytes_for(&self, key: &str) -> Vec<u8> {
            self.received
                .lock()
                .unwrap()
                .get(key)
                .cloned()
                .unwrap_or_default()
        }
    }

    impl Uploader for FakeUploader {
        fn upload(&self, key: &str, data: &[u8], start_offset: u64) -> UploadOutcome {
            if *self.offline.lock().unwrap() {
                // Link down: no progress, file stays pending and is retried.
                return UploadOutcome::Failed;
            }

            let first_attempt = self.attempted.lock().unwrap().insert(key.to_string());
            let remaining = &data[start_offset as usize..];

            // On the first attempt, if configured to interrupt, accept only a
            // prefix of the remaining bytes and report how far the cloud got.
            if first_attempt && let Some(after) = self.interrupt_after {
                let take = (after as usize).min(remaining.len());
                let slice = &remaining[..take];
                self.received
                    .lock()
                    .unwrap()
                    .entry(key.to_string())
                    .or_default()
                    .extend_from_slice(slice);
                return UploadOutcome::Interrupted {
                    uploaded_to: start_offset + take as u64,
                };
            }

            // Accept the whole remaining slice and confirm done.
            self.received
                .lock()
                .unwrap()
                .entry(key.to_string())
                .or_default()
                .extend_from_slice(remaining);
            UploadOutcome::Done
        }
    }

    fn found(path: &str, class: StreamClass, size: u64) -> FoundFile {
        FoundFile {
            path: PathBuf::from(path),
            class,
            size,
        }
    }

    #[test]
    fn parses_class_and_priority_from_filename() {
        assert_eq!(
            StreamClass::from_filename("safety-3.mcap"),
            Some(StreamClass::Safety)
        );
        assert_eq!(
            StreamClass::from_filename("meta-0.mcap"),
            Some(StreamClass::Meta)
        );
        assert_eq!(
            StreamClass::from_filename("sensor-12.mcap"),
            Some(StreamClass::Sensor)
        );
        assert_eq!(StreamClass::from_filename("notes-1.mcap"), None);
        // Priority order: Safety sorts before Meta before Sensor.
        assert!(StreamClass::Safety < StreamClass::Meta);
        assert!(StreamClass::Meta < StreamClass::Sensor);
    }

    #[test]
    fn uploads_safety_first_then_meta_then_sensor() {
        // Real files on disk so the agent can read their bytes; the source hands
        // them back deliberately out of priority order to prove the agent sorts.
        let data_dir = tempdir().unwrap();
        let state_dir = tempdir().unwrap();
        let make = |name: &str, body: &[u8], class: StreamClass| {
            let p = data_dir.path().join(name);
            fs::write(&p, body).unwrap();
            FoundFile {
                path: p,
                class,
                size: body.len() as u64,
            }
        };
        // Order in the source list is intentionally Sensor, Safety, Meta.
        let files = vec![
            make("sensor-0.mcap", b"SSSS", StreamClass::Sensor),
            make("safety-0.mcap", b"AA", StreamClass::Safety),
            make("meta-0.mcap", b"MMM", StreamClass::Meta),
        ];
        let source = VecSource::new(files);

        // An uploader that records the *order* keys were uploaded in.
        struct OrderUploader {
            order: Mutex<Vec<String>>,
        }
        impl Uploader for OrderUploader {
            fn upload(&self, key: &str, _data: &[u8], _start: u64) -> UploadOutcome {
                self.order.lock().unwrap().push(key.to_string());
                UploadOutcome::Done
            }
        }
        let uploader = OrderUploader {
            order: Mutex::new(Vec::new()),
        };

        let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();
        agent.sync_once().unwrap();

        let order = agent.uploader.order.lock().unwrap().clone();
        assert_eq!(
            order,
            vec![
                "safety-0.mcap".to_string(),
                "meta-0.mcap".to_string(),
                "sensor-0.mcap".to_string(),
            ],
            "files must upload Safety first, then Meta, then Sensor"
        );
    }

    #[test]
    fn resumes_from_interrupted_offset_without_gap_or_duplication() {
        let data_dir = tempdir().unwrap();
        let state_dir = tempdir().unwrap();
        let body = b"0123456789"; // 10 bytes
        let path = data_dir.path().join("meta-0.mcap");
        fs::write(&path, body).unwrap();
        let source = VecSource::new(vec![FoundFile {
            path: path.clone(),
            class: StreamClass::Meta,
            size: body.len() as u64,
        }]);

        // Interrupt after 4 bytes on the first attempt.
        let uploader = FakeUploader::interrupting(4);
        let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();

        // First sync: interrupted at 4, file stays pending with offset 4.
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.done, 0);
        assert_eq!(stats.bytes_uploaded, 4);
        assert_eq!(
            agent.manifest.files["meta-0.mcap"],
            FileRecord {
                status: FileStatus::Pending,
                uploaded_offset: 4,
            }
        );

        // Second sync: resumes from 4 and completes.
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.done, 1);
        assert_eq!(stats.bytes_uploaded, 10);

        // The cloud received the whole file exactly once: no gap, no duplication.
        assert_eq!(agent.uploader.bytes_for("meta-0.mcap"), body.to_vec());
    }

    #[test]
    fn offline_uploader_keeps_file_pending_then_succeeds_after_reconnect() {
        let data_dir = tempdir().unwrap();
        let state_dir = tempdir().unwrap();
        let body = b"payload";
        let path = data_dir.path().join("safety-0.mcap");
        fs::write(&path, body).unwrap();
        let source = VecSource::new(vec![FoundFile {
            path,
            class: StreamClass::Safety,
            size: body.len() as u64,
        }]);

        let uploader = FakeUploader::offline();
        let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();

        // Offline: upload fails, file stays pending, nothing dropped.
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.done, 0);
        assert_eq!(stats.bytes_uploaded, 0);

        // "Reconnect" and sync again: the still-pending file uploads.
        *agent.uploader.offline.lock().unwrap() = false;
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.done, 1);
        assert_eq!(agent.uploader.bytes_for("safety-0.mcap"), body.to_vec());
    }

    #[test]
    fn restart_does_not_reupload_done_files() {
        let data_dir = tempdir().unwrap();
        let state_dir = tempdir().unwrap();
        let body = b"once";
        let path = data_dir.path().join("meta-0.mcap");
        fs::write(&path, body).unwrap();
        let found = FoundFile {
            path,
            class: StreamClass::Meta,
            size: body.len() as u64,
        };

        // First agent uploads the file to completion.
        {
            let uploader = FakeUploader::new();
            let source = VecSource::new(vec![found.clone()]);
            let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();
            let stats = agent.sync_once().unwrap();
            assert_eq!(stats.done, 1);
        }

        // A fresh agent pointed at the same state dir must NOT re-upload the done
        // file: a brand-new uploader should receive nothing.
        let uploader = FakeUploader::new();
        let source = VecSource::new(vec![found]);
        let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.done, 1);
        assert_eq!(stats.pending, 0);
        // The fresh uploader received nothing, proving no re-upload happened.
        assert!(
            agent.uploader.bytes_for("meta-0.mcap").is_empty(),
            "a done file must not be uploaded again after restart"
        );
    }

    #[test]
    fn dirsource_skips_files_still_being_written() {
        let dir = tempdir().unwrap();
        // Two files: one we will treat as quiet (old mtime) and one just touched.
        let quiet = dir.path().join("meta-0.mcap");
        let fresh = dir.path().join("sensor-0.mcap");
        // A finalized file ends with the MCAP magic; the fresh one is skipped by the
        // quiet check before the magic check, so its content is irrelevant.
        write_finished_mcap(&quiet, b"done");
        fs::write(&fresh, b"writing").unwrap();

        // Set the quiet file's mtime well into the past, leave the fresh one as
        // just-now, then ask "now" to be a moment after the writes. With a 60s
        // quiet period only the old file has gone quiet.
        let now = SystemTime::now();
        let old = now - Duration::from_secs(3600);
        let quiet_file = fs::File::open(&quiet).unwrap();
        quiet_file.set_modified(old).unwrap();

        let source = DirSource::with_clock(dir.path(), Duration::from_secs(60), move || now);
        let finished = source.finished_files();

        let names: Vec<String> = finished
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.contains(&"meta-0.mcap".to_string()),
            "a file quiet for longer than the quiet period must be finished"
        );
        assert!(
            !names.contains(&"sensor-0.mcap".to_string()),
            "a freshly-written (still in-progress) file must be skipped"
        );
    }

    #[test]
    fn dirsource_ignores_non_mcap_and_unknown_prefixes() {
        let dir = tempdir().unwrap();
        let now = SystemTime::now();
        let old = now - Duration::from_secs(3600);
        for name in ["safety-0.mcap", "notes.txt", "unknown-0.mcap"] {
            let p = dir.path().join(name);
            write_finished_mcap(&p, b"x");
            fs::File::open(&p).unwrap().set_modified(old).unwrap();
        }
        let source = DirSource::with_clock(dir.path(), Duration::from_secs(1), move || now);
        let finished = source.finished_files();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].class, StreamClass::Safety);
    }

    /// A torn (unfinalized) `.mcap` that is otherwise quiet is NOT uploaded: it lacks the
    /// MCAP end magic, so the agent treats it as still-being-written rather than shipping
    /// a truncated recording.
    #[test]
    fn dirsource_skips_a_quiet_but_torn_mcap_without_end_magic() {
        let dir = tempdir().unwrap();
        let now = SystemTime::now();
        let old = now - Duration::from_secs(3600);
        let torn = dir.path().join("sensor-0.mcap");
        fs::write(&torn, b"no end magic here").unwrap(); // quiet but unfinalized
        fs::File::open(&torn).unwrap().set_modified(old).unwrap();
        let finished_ok = dir.path().join("meta-0.mcap");
        write_finished_mcap(&finished_ok, b"body");
        fs::File::open(&finished_ok)
            .unwrap()
            .set_modified(old)
            .unwrap();

        let source = DirSource::with_clock(dir.path(), Duration::from_secs(1), move || now);
        let names: Vec<String> = source
            .finished_files()
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.contains(&"meta-0.mcap".to_string()),
            "finalized file is uploadable"
        );
        assert!(
            !names.contains(&"sensor-0.mcap".to_string()),
            "a torn file with no MCAP end magic must not be uploaded"
        );
    }

    /// Write `body` followed by the MCAP end magic, so the file looks finalized to the
    /// agent's torn-file guard.
    fn write_finished_mcap(path: &Path, body: &[u8]) {
        let mut bytes = body.to_vec();
        bytes.extend_from_slice(&MCAP_MAGIC);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn never_drops_across_repeated_offline_syncs() {
        // Hammer sync_once while offline many times: the file must remain pending
        // every time (never lost, never silently done), then complete on reconnect.
        let data_dir = tempdir().unwrap();
        let state_dir = tempdir().unwrap();
        let body = b"important";
        let path = data_dir.path().join("safety-0.mcap");
        fs::write(&path, body).unwrap();
        let source = VecSource::new(vec![found(
            path.to_str().unwrap(),
            StreamClass::Safety,
            body.len() as u64,
        )]);
        let uploader = FakeUploader::offline();
        let mut agent = Agent::new(uploader, source, state_dir.path()).unwrap();

        for _ in 0..5 {
            let stats = agent.sync_once().unwrap();
            assert_eq!(stats.pending, 1, "offline file must stay pending");
            assert_eq!(stats.done, 0);
        }
        *agent.uploader.offline.lock().unwrap() = false;
        let stats = agent.sync_once().unwrap();
        assert_eq!(stats.done, 1);
        assert_eq!(agent.uploader.bytes_for("safety-0.mcap"), body.to_vec());
    }

    // ---- RolloutCheckpoint: delta batching / idempotency (P0 #1) -------------

    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, RobotId,
        RobotIdentity, TenantId,
    };

    /// A minimal rollout for the checkpoint tests, serialized one-per-line the way capture
    /// flushes them to `rollouts.jsonl`.
    fn ckpt_rollout(step: u32) -> Rollout {
        Rollout::new(
            RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1")),
            EpisodeId::new(),
            step,
            MonoClock {
                boot_id: BootId::new(),
                mono_ns: u64::from(step) + 1,
                ts_wall_ns: 1,
            },
            PolicyVersion::new("pol@v1+abc123def456"),
            "sha256:w".into(),
            "arm6dof".into(),
            "pick".into(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            1,
        )
    }

    /// Append rollouts to a JSONL file (create-or-append), one canonical line each, the way
    /// the capture drain does.
    fn append_rollouts(path: &Path, rollouts: &[Rollout]) {
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        for r in rollouts {
            writeln!(f, "{}", serde_json::to_string(r).unwrap()).unwrap();
        }
    }

    /// A missing rollouts file is a clean empty delta at offset 0, not an error.
    #[test]
    fn checkpoint_missing_file_is_empty_delta() {
        let dir = tempdir().unwrap();
        let ckpt = RolloutCheckpoint::load(dir.path());
        let (delta, off) = ckpt.read_new(&dir.path().join("rollouts.jsonl")).unwrap();
        assert!(delta.is_empty());
        assert_eq!(off, 0);
    }

    /// The core idempotency property: each pass reads only the rollouts appended since the
    /// committed offset, so a growing file yields disjoint deltas instead of re-sending the
    /// whole file (which defeated the gateway's batch-id dedup).
    #[test]
    fn checkpoint_reads_only_the_delta_after_commit() {
        let dir = tempdir().unwrap();
        let rollouts = dir.path().join("rollouts.jsonl");
        let mut ckpt = RolloutCheckpoint::load(dir.path());

        // First pass: two rollouts appear.
        append_rollouts(&rollouts, &[ckpt_rollout(0), ckpt_rollout(1)]);
        let (delta1, off1) = ckpt.read_new(&rollouts).unwrap();
        assert_eq!(delta1.len(), 2, "first pass sees both new rollouts");
        ckpt.commit(off1).unwrap();

        // Second pass after one MORE rollout is appended: only the new one, not all three.
        append_rollouts(&rollouts, &[ckpt_rollout(2)]);
        let (delta2, off2) = ckpt.read_new(&rollouts).unwrap();
        assert_eq!(delta2.len(), 1, "second pass sees only the delta");
        assert_eq!(delta2[0].step_index, 2);
        ckpt.commit(off2).unwrap();

        // Nothing new -> empty delta.
        let (delta3, _) = ckpt.read_new(&rollouts).unwrap();
        assert!(delta3.is_empty());
    }

    /// Before a commit, re-reading returns the IDENTICAL delta (so a retried POST reuses the
    /// same batch id and the gateway dedups it) — the crash-between-POST-and-commit case.
    #[test]
    fn checkpoint_reread_before_commit_is_identical() {
        let dir = tempdir().unwrap();
        let rollouts = dir.path().join("rollouts.jsonl");
        let ckpt = RolloutCheckpoint::load(dir.path());
        append_rollouts(&rollouts, &[ckpt_rollout(0), ckpt_rollout(1)]);

        let (a, off_a) = ckpt.read_new(&rollouts).unwrap();
        let (b, off_b) = ckpt.read_new(&rollouts).unwrap();
        assert_eq!(off_a, off_b);
        let ids_a: Vec<_> = a.iter().map(|r| r.id).collect();
        let ids_b: Vec<_> = b.iter().map(|r| r.id).collect();
        assert_eq!(
            ids_a, ids_b,
            "re-read before commit must be the identical delta"
        );
    }

    /// A trailing partial line (capture mid-write, no newline yet) is NOT consumed until its
    /// newline lands, so a half-serialized rollout is never parsed or registered.
    #[test]
    fn checkpoint_excludes_a_partial_trailing_line() {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let rollouts = dir.path().join("rollouts.jsonl");
        append_rollouts(&rollouts, &[ckpt_rollout(0)]); // one complete line
        // Append a partial line with no trailing newline.
        let mut f = fs::OpenOptions::new().append(true).open(&rollouts).unwrap();
        write!(f, "{{\"partial\": ").unwrap();
        drop(f);

        let ckpt = RolloutCheckpoint::load(dir.path());
        let (delta, _) = ckpt.read_new(&rollouts).unwrap();
        assert_eq!(delta.len(), 1, "only the one complete line is read");
    }

    /// A malformed COMPLETE line is a hard error, never a silently-dropped rollout —
    /// surfacing the parse failure so the pass retries rather than registering a gap.
    #[test]
    fn checkpoint_malformed_complete_line_is_an_error() {
        let dir = tempdir().unwrap();
        let rollouts = dir.path().join("rollouts.jsonl");
        fs::write(&rollouts, "{not valid json}\n").unwrap();
        let ckpt = RolloutCheckpoint::load(dir.path());
        assert!(ckpt.read_new(&rollouts).is_err());
    }

    /// The committed offset survives a reload (atomic temp+rename persist), so a restarted
    /// agent resumes from where it acked rather than re-registering from zero.
    #[test]
    fn checkpoint_offset_persists_across_reload() {
        let dir = tempdir().unwrap();
        let rollouts = dir.path().join("rollouts.jsonl");
        append_rollouts(&rollouts, &[ckpt_rollout(0), ckpt_rollout(1)]);

        let mut ckpt = RolloutCheckpoint::load(dir.path());
        let (_, off) = ckpt.read_new(&rollouts).unwrap();
        ckpt.commit(off).unwrap();

        // A fresh agent loads the persisted offset and sees no un-registered rollouts.
        let reloaded = RolloutCheckpoint::load(dir.path());
        assert_eq!(reloaded.offset(), off);
        let (delta, _) = reloaded.read_new(&rollouts).unwrap();
        assert!(
            delta.is_empty(),
            "a reloaded checkpoint re-registers nothing already acked"
        );
    }
}
