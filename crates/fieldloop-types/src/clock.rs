//! The two-clock stamp.
//!
//! Robot wall-clocks drift, so the schema separates uniqueness from time-ordering:
//! ids are UUIDv7 for uniqueness and coarse sort only, and their embedded timestamp
//! is advisory, never an attribution or cooldown input. The authority for "when did
//! this happen relative to that" is a monotonic clock, captured here.
//!
//! Three time sources coexist, with a strict authority order:
//!
//! 1. **`(boot_id, mono_ns)` — the attribution authority.** `CLOCK_MONOTONIC`
//!    nanoseconds, only comparable *within one boot* (a reboot resets the origin).
//!    Skew-free, since it never depends on a wall-clock. The JOIN does temporal
//!    windowing on `mono_ns` deltas inside the same `boot_id`.
//! 2. **`ts_ingest` — server-set, trusted.** Stamped by the ingest gateway when the
//!    row lands. Drives cooldown-elapsed and cross-boot/cross-robot alignment. The
//!    `server_anchor_offset_ns` anchors this boot's monotonic origin to server time
//!    so two boots can be coarsely aligned when same-boot comparison is impossible.
//! 3. **`ts_wall_ns` — robot wall estimate, advisory.** Skewed; coarse alignment
//!    only. Never an attribution input on its own.

use serde::{Deserialize, Serialize};

use crate::ids::BootId;

/// The robot-side monotonic stamp captured at the moment of the event.
///
/// `(boot_id, mono_ns)` is the **attribution authority**: it is skew-free, unlike
/// any wall-clock. `mono_ns` is `CLOCK_MONOTONIC` and is meaningful only relative
/// to other `mono_ns` readings carrying the *same* `boot_id` (a reboot resets the
/// origin). `ts_wall_ns` is the robot's advisory wall-clock estimate, carried for
/// coarse alignment only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonoClock {
    /// Robot boot session. `mono_ns` comparisons are valid only within one boot.
    pub boot_id: BootId,
    /// `CLOCK_MONOTONIC` nanoseconds at the event — the attribution authority.
    /// Cheap, non-allocating to read inside the 50Hz control loop.
    pub mono_ns: u64,
    /// Robot wall-clock estimate (nanoseconds since Unix epoch). **Advisory /
    /// skewed** — coarse alignment only, never a sole attribution input.
    pub ts_wall_ns: i64,
}

impl MonoClock {
    /// Construct a stamp from raw robot-side readings.
    #[must_use]
    pub const fn new(boot_id: BootId, mono_ns: u64, ts_wall_ns: i64) -> Self {
        Self {
            boot_id,
            mono_ns,
            ts_wall_ns,
        }
    }

    /// Monotonic delta to a later stamp **iff both are from the same boot**.
    ///
    /// Returns `None` across boots — a reboot resets `CLOCK_MONOTONIC`, so there is
    /// no valid monotonic comparison across one. Cross-boot alignment must instead
    /// go through server-anchored ingest time ([`ServerAnchor`]). Returning `None`
    /// makes the "can't subtract `mono_ns` across boots" rule impossible to violate
    /// by accident at the call site.
    #[must_use]
    pub fn mono_delta_ns(&self, later: &MonoClock) -> Option<i128> {
        if self.boot_id == later.boot_id {
            Some(i128::from(later.mono_ns) - i128::from(self.mono_ns))
        } else {
            None
        }
    }
}

/// Server-set time, applied by the ingest gateway when a row lands — the *trusted*
/// clock, since it does not depend on any drifting robot wall-clock.
///
/// `ts_ingest_ns` drives cooldown-elapsed and partitioning;
/// `server_anchor_offset_ns` anchors this boot's monotonic origin to server time
/// so that `(boot_id, mono_ns)` events from different boots/robots can be coarsely
/// reconciled when same-boot monotonic comparison is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerAnchor {
    /// Server-set ingest time (nanoseconds since Unix epoch). Trusted; drives
    /// cooldown elapsed and partition selection. Never robot-supplied.
    pub ts_ingest_ns: i64,
    /// Server-computed offset anchoring this boot's `mono_ns` origin to server
    /// time. Lets cross-boot/cross-robot events be aligned on a common timeline.
    pub server_anchor_offset_ns: i64,
}

impl ServerAnchor {
    /// Construct a server anchor (only the gateway should do this).
    #[must_use]
    pub const fn new(ts_ingest_ns: i64, server_anchor_offset_ns: i64) -> Self {
        Self {
            ts_ingest_ns,
            server_anchor_offset_ns,
        }
    }
}
