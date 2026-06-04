//! The monotonic clock source the hot path reads on every step, plus the OS-sourced
//! boot session id that scopes those monotonic readings.
//!
//! Robot wall-clocks drift, so the authoritative "when did this happen" signal is
//! a monotonic counter that only ever moves forward within one boot. The hot path
//! reads two numbers per step: a monotonic nanosecond value (the attribution
//! authority) and an advisory wall-clock estimate (coarse alignment only). Reading
//! the clock must be cheap and non-blocking because it runs inside a ~50Hz control
//! loop, so it never allocates and never takes a lock.
//!
//! A monotonic reading is only meaningful relative to other readings from the same
//! boot session — a reboot resets the monotonic origin to an arbitrary value, so a
//! delta taken across a reboot is garbage. The boot session must therefore be
//! identified by something that actually changes on reboot and stays constant across
//! mere process restarts. A per-process random id cannot tell those two apart: it
//! changes on every restart, so a reboot looks identical to a restart and two clocks
//! in one process would disagree on which boot they belong to. So the boot id here is
//! derived from an OS boot-session fact (Linux boot uuid / boot time, macOS boot
//! time), computed once per process and shared by every clock, so all clocks in one
//! boot agree and a reboot is the only thing that changes the id.

use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fieldloop_types::BootId;
use uuid::Uuid;

/// A source of `(mono_ns, wall_ns)` readings tied to a boot session.
///
/// `mono_ns` is a monotonic (non-decreasing) nanosecond counter whose origin is
/// arbitrary but fixed for the life of the boot — only differences within one boot
/// are meaningful, which matches the "monotonic clock is the attribution authority
/// within a boot" rule. `wall_ns` is nanoseconds since the Unix epoch and is advisory
/// only (it can jump backward when the wall-clock is corrected), so it is never used
/// for ordering. `boot_id` names the boot session those `mono_ns` values are scoped
/// to: two readings are only comparable when their sources report the same `boot_id`.
///
/// The trait exists so tests can inject a deterministic fake in place of the real
/// system clock, making clock-dependent behavior reproducible — including the boot
/// session, so a test can simulate same-boot and cross-boot scenarios.
pub trait ClockSource: Send + Sync {
    /// Read the clock once. Returns `(mono_ns, wall_ns)`. Must be cheap and
    /// non-blocking — it is called on every control-loop step.
    fn read(&self) -> (u64, i64);

    /// The boot session these monotonic readings belong to. Stable for the life of
    /// the source; `mono_ns` deltas are only valid between readings that share it.
    fn boot_id(&self) -> BootId;
}

/// The real clock used on the robot.
///
/// `mono_ns` is nanoseconds elapsed since a process-wide monotonic anchor, derived
/// from [`std::time::Instant`] which is guaranteed non-decreasing. Anchoring all
/// `SystemClock`s in a process to one shared origin (rather than the instant each was
/// constructed) means two clocks built at different times in the same boot agree on
/// the monotonic timeline, so their `mono_ns` values are directly comparable. `wall_ns`
/// is nanoseconds since the Unix epoch from [`std::time::SystemTime`], carried as an
/// advisory estimate only. `boot_id` is the OS-sourced boot session, the same value
/// for every `SystemClock` in this boot.
pub struct SystemClock {
    /// The OS-sourced boot session id, resolved once per process and shared. Constant
    /// across process restarts within one boot; changes only on a real reboot.
    boot_id: BootId,
}

impl SystemClock {
    /// Build a clock anchored to this boot's OS boot session and the process-wide
    /// monotonic origin.
    ///
    /// The boot id is resolved from the OS (and cached), not minted fresh, so every
    /// `SystemClock` in the process reports the same boot session — which is what makes
    /// a reboot (new OS boot session) distinguishable from a mere restart (same OS boot
    /// session).
    #[must_use]
    pub fn new() -> Self {
        Self {
            boot_id: os_boot_id(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockSource for SystemClock {
    fn read(&self) -> (u64, i64) {
        // Measure against the process-wide origin so two SystemClocks share a timeline.
        // Reading the cached origin is a relaxed atomic-ish OnceLock get plus an
        // Instant::elapsed — no allocation, no lock — so it is safe in the 50Hz loop.
        // Elapsed since a fixed origin is non-decreasing by Instant's contract. Saturate
        // the nanosecond count to u64 so an absurdly long uptime cannot panic the control
        // loop; in practice u64 nanoseconds spans ~584 years.
        let mono_ns = u64::try_from(mono_origin().elapsed().as_nanos()).unwrap_or(u64::MAX);

        // Wall time since the Unix epoch. If the system clock is set before the
        // epoch (or unavailable) we fall back to 0 rather than panic — this value
        // is advisory and never an ordering input, so a degraded reading is fine.
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_nanos()).ok())
            .unwrap_or(0);

        (mono_ns, wall_ns)
    }

    fn boot_id(&self) -> BootId {
        self.boot_id
    }
}

/// The process-wide monotonic origin: the instant the first clock anchored against.
///
/// Captured once and shared so every `SystemClock` measures `mono_ns` from the same
/// point. Without a shared origin, a clock built later in the boot would report a
/// smaller `mono_ns` for the same wall moment than one built earlier, and the two
/// could not be compared even though they are the same boot.
fn mono_origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// Resolve this boot's session id from the OS, once per process.
///
/// Cached in a `OnceLock` so the (possibly file-reading / syscall) resolution happens
/// at most once and every later caller — every `SystemClock` — gets the identical id
/// for free. This is what guarantees two clocks in one boot agree and a restart does
/// not look like a reboot.
fn os_boot_id() -> BootId {
    static BOOT_ID: OnceLock<BootId> = OnceLock::new();
    *BOOT_ID.get_or_init(|| BootId::from_uuid(boot_uuid_from_os()))
}

/// Fold an OS boot-session fact into a 16-byte id deterministically.
///
/// The same input bytes always produce the same id, and different boot sessions
/// (different boot uuid / boot time) produce different ids with overwhelming
/// probability. A double-hash fills all 16 bytes from one variable-length input
/// without pulling in a cryptographic-hash dependency: this id only needs to be a
/// stable, well-distributed label for "which boot", not collision-resistant against
/// an adversary.
fn boot_uuid_from_bytes(seed: &[u8]) -> Uuid {
    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h1);
    let hi = h1.finish();

    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    // Salt the second half so the two 64-bit halves are independent even for a
    // short seed (otherwise both halves would be the same hash of the same input).
    0xF1E1_D100_B007_5E55u64.hash(&mut h2);
    seed.hash(&mut h2);
    let lo = h2.finish();

    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&hi.to_le_bytes());
    bytes[8..].copy_from_slice(&lo.to_le_bytes());
    Uuid::from_bytes(bytes)
}

/// Read this boot's session id from the OS, platform by platform.
///
/// Each branch returns a value that is constant for the life of one boot and changes
/// on the next boot, so it is a real boot-session identifier — not a per-process value.
///
/// * Linux: `/proc/sys/kernel/random/boot_id` is a kernel-minted uuid regenerated on
///   every boot; if it is unreadable, `/proc/stat`'s `btime` line (boot wall time in
///   whole seconds) is the fallback. Both are stable within a boot.
/// * macOS / iOS (the dev platform): `sysctl kern.boottime` returns the boot
///   timestamp; that timestamp is fixed for the boot and resets on the next boot.
/// * Any other platform: there is no portable OS boot fact reachable from std, so this
///   falls back to a value that is stable for the life of THIS PROCESS but not across a
///   reboot — meaning on the fallback path a reboot is indistinguishable from a process
///   restart. That is an honest degradation, not a real boot id; it is labeled as such
///   below and is the price of not adding a platform crate for unsupported targets.
fn boot_uuid_from_os() -> Uuid {
    #[cfg(target_os = "linux")]
    {
        if let Some(uuid) = linux_boot_seed() {
            return boot_uuid_from_bytes(&uuid);
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Some(seed) = darwin_boottime_seed() {
            return boot_uuid_from_bytes(&seed);
        }
    }

    // Fallback for platforms with no reachable OS boot fact (or when the OS read
    // failed). HONEST LIMITATION: this seed is derived from a process-lifetime anchor,
    // so it is constant within this process but a fresh process — including one after a
    // reboot — gets a different id. On this path a reboot looks like a restart, which is
    // exactly the ambiguity the OS-sourced branches above remove. It is used only when
    // no real boot fact is available.
    boot_uuid_from_bytes(&process_anchor_seed())
}

/// Linux: the kernel's per-boot uuid, or the boot wall-time as a fallback seed.
///
/// `/proc/sys/kernel/random/boot_id` is regenerated by the kernel on each boot and is
/// the canonical boot-session id; reading it is a tiny one-shot file read done once at
/// process start, never on the hot path. If that read fails, `/proc/stat`'s `btime`
/// line carries the boot wall time in seconds, which is likewise fixed per boot.
#[cfg(target_os = "linux")]
fn linux_boot_seed() -> Option<Vec<u8>> {
    if let Ok(s) = std::fs::read_to_string("/proc/sys/kernel/random/boot_id") {
        let t = s.trim();
        if !t.is_empty() {
            return Some(t.as_bytes().to_vec());
        }
    }
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    for line in stat.lines() {
        if let Some(rest) = line.strip_prefix("btime ") {
            let v = rest.trim();
            if !v.is_empty() {
                return Some(format!("btime:{v}").into_bytes());
            }
        }
    }
    None
}

/// macOS / iOS: the boot timestamp from `sysctl kern.boottime`.
///
/// `kern.boottime` is a `struct timeval` (seconds + microseconds) recording when the
/// kernel booted; it is fixed for the life of a boot and differs across boots, so its
/// raw bytes are a real boot-session seed. The `sysctl(2)` call by MIB name is a single
/// syscall made once at process start (cached afterward), never on the hot path.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn darwin_boottime_seed() -> Option<Vec<u8>> {
    // CTL_KERN / KERN_BOOTTIME from <sys/sysctl.h>. A timeval is two C longs on these
    // 64-bit targets (tv_sec, tv_usec), so 16 bytes of output.
    const CTL_KERN: libc::c_int = 1;
    const KERN_BOOTTIME: libc::c_int = 21;
    let mut mib: [libc::c_int; 2] = [CTL_KERN, KERN_BOOTTIME];
    let mut tv = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    let mut len = std::mem::size_of::<libc::timeval>();
    // SAFETY: `mib` is a valid 2-element MIB, `tv` is a writable timeval of exactly
    // `len` bytes, and the new-value pointer is null (a read). sysctl fills `tv` and
    // updates `len`; we only trust the result when the call returns 0.
    let rc = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            2,
            std::ptr::addr_of_mut!(tv).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    // tv_sec/tv_usec are i64 on these 64-bit Apple targets; their little-endian bytes
    // are the boot-session seed. to_le_bytes pins the byte order so the seed is the same
    // regardless of host endianness.
    let mut seed = Vec::with_capacity(16);
    seed.extend_from_slice(&tv.tv_sec.to_le_bytes());
    seed.extend_from_slice(&tv.tv_usec.to_le_bytes());
    Some(seed)
}

/// A process-lifetime seed for the no-OS-boot-fact fallback.
///
/// Combines the process start wall time with the process id so two processes started
/// at the same coarse instant still differ. This is deliberately NOT a boot id: it is
/// constant within a process but changes on every restart, so callers reaching this
/// path cannot distinguish a reboot from a restart. Used only when no platform boot
/// fact is available.
fn process_anchor_seed() -> Vec<u8> {
    // Anchor captured once: the first call's wall time is what every later call in this
    // process sees, so the fallback id is stable for the process lifetime.
    static ANCHOR_NS: OnceLock<i64> = OnceLock::new();
    let anchor = *ANCHOR_NS.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_nanos()).ok())
            .unwrap_or(0)
    });
    let pid = std::process::id();
    let mut seed = Vec::with_capacity(12);
    seed.extend_from_slice(&anchor.to_le_bytes());
    seed.extend_from_slice(&pid.to_le_bytes());
    seed
}

/// A deterministic clock for tests.
///
/// Each `read` returns the current stored monotonic value and the current stored
/// wall value, and advances the monotonic value by a fixed step so successive reads
/// are strictly increasing. This makes the "monotonic value is non-decreasing across
/// steps" property testable without depending on real time. It also carries a fixed
/// `boot_id` so a test can pin which boot session the readings belong to, and pair two
/// fakes with the same or different boot ids to exercise same-boot vs cross-boot paths.
pub struct FakeClock {
    /// Current monotonic value, advanced atomically on each read so the fake is
    /// safe to share across threads exactly like the real clock.
    mono_ns: AtomicU64,
    /// How much `mono_ns` advances per read.
    step_ns: u64,
    /// Fixed advisory wall value returned by every read.
    wall_ns: i64,
    /// Fixed boot session this fake reports — chosen by the test.
    boot_id: BootId,
}

impl FakeClock {
    /// Build a fake starting at `start_mono_ns`, advancing by `step_ns` each read,
    /// always reporting `wall_ns` as the advisory wall estimate, and reporting a
    /// freshly-minted boot id.
    #[must_use]
    pub fn new(start_mono_ns: u64, step_ns: u64, wall_ns: i64) -> Self {
        Self::with_boot(start_mono_ns, step_ns, wall_ns, BootId::new())
    }

    /// Build a fake with an explicit boot id, so a test can make two fakes share a
    /// boot session (deltas valid) or differ (deltas must be refused).
    #[must_use]
    pub fn with_boot(start_mono_ns: u64, step_ns: u64, wall_ns: i64, boot_id: BootId) -> Self {
        Self {
            mono_ns: AtomicU64::new(start_mono_ns),
            step_ns,
            wall_ns,
            boot_id,
        }
    }
}

impl ClockSource for FakeClock {
    fn read(&self) -> (u64, i64) {
        // fetch_add returns the value before the add, so the sequence of returned
        // monotonic values is start, start+step, start+2*step, ... — strictly
        // increasing and reproducible.
        let mono_ns = self.mono_ns.fetch_add(self.step_ns, Ordering::Relaxed);
        (mono_ns, self.wall_ns)
    }

    fn boot_id(&self) -> BootId {
        self.boot_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_system_clocks_share_one_boot_session() {
        // The audit's core property: a boot id is a BOOT session, not a per-instance
        // id. Two SystemClocks constructed separately in the same process/boot must
        // report the identical boot id — otherwise a mere restart would look like a
        // reboot and two clocks in one process would disagree on the boot.
        let a = SystemClock::new();
        let b = SystemClock::new();
        assert_eq!(
            a.boot_id(),
            b.boot_id(),
            "two SystemClocks in one boot must agree on boot_id"
        );
    }

    #[test]
    fn boot_id_is_stable_across_repeated_reads() {
        // boot_id() must not change reading to reading within a clock's life.
        let c = SystemClock::new();
        let first = c.boot_id();
        let _ = c.read();
        let _ = c.read();
        assert_eq!(first, c.boot_id());
    }

    #[test]
    fn system_clock_mono_is_non_decreasing_and_shares_origin() {
        // Two clocks anchored to the shared process origin produce comparable,
        // non-decreasing mono readings.
        let a = SystemClock::new();
        let b = SystemClock::new();
        let (m0, _) = a.read();
        let (m1, _) = b.read();
        let (m2, _) = a.read();
        assert!(m1 >= m0, "second clock's reading is on the same timeline");
        assert!(m2 >= m1, "later reading is non-decreasing");
    }

    #[test]
    fn os_seed_is_deterministic_for_same_input() {
        // The same boot fact always folds to the same id; different facts differ.
        let one = boot_uuid_from_bytes(b"boot-session-A");
        let again = boot_uuid_from_bytes(b"boot-session-A");
        let other = boot_uuid_from_bytes(b"boot-session-B");
        assert_eq!(one, again);
        assert_ne!(one, other);
        // Both 64-bit halves are filled (not a half-zero uuid), so the id is well
        // distributed rather than concentrated in the low bytes.
        assert_ne!(one.as_bytes()[..8], [0u8; 8]);
        assert_ne!(one.as_bytes()[8..], [0u8; 8]);
    }
}
