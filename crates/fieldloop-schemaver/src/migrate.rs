//! Forward migration of a record envelope, with explicit degradation markers.
//!
//! An old robot emits a record missing fields the schema added later. The danger
//! is that filling those fields with zeros would look like real data downstream —
//! e.g. a fabricated all-zero monotonic clock would let the JOIN attribute by a
//! clock that never existed. So migration here does the opposite: it fills absent
//! fields conservatively AND records, in an explicit `Capabilities` descriptor,
//! exactly what was missing or derived. Downstream then degrades deterministically
//! (wider window, lowered trust) instead of silently trusting fabricated data.
//!
//! Migrations form an ordered chain `v_n -> v_{n+1}` applied in sequence. They are
//! idempotent in the sense that the runner only applies steps strictly above the
//! envelope's current version, so re-running the chain on an already-current
//! record changes nothing.

use serde::{Deserialize, Serialize};

use fieldloop_types::{BootId, MonoClock};

use crate::version::{CURRENT, SchemaVersion};

/// What kind of clock authority the record actually carries after migration.
///
/// This is the headline degradation marker. The JOIN's attribution authority is
/// the monotonic `(boot_id, mono_ns)` clock; a record that never had one cannot be
/// attributed by it. Spelling that out as an enum (rather than a possibly-fake
/// clock value) forces the JOIN to handle the wall-only case explicitly instead of
/// subtracting nanoseconds that don't mean anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockCapability {
    /// Real monotonic `(boot_id, mono_ns)` clock present — full attribution.
    Monotonic,
    /// Only a robot wall-clock estimate is available (a pre-monotonic-clock SDK).
    /// Attribution must fall back to a wider, lower-trust wall-clock window.
    WallOnly,
    /// No usable timestamp at all — attribution by time is impossible.
    None,
}

/// An explicit statement of what a migrated record can and cannot do.
///
/// Carried alongside the migrated envelope so downstream never has to guess
/// whether a field is real or back-filled. Every back-filled or derived field
/// adds a human-readable note here, so a degraded record is self-describing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// The clock authority this record actually supports.
    pub clock: ClockCapability,
    /// One note per field that was absent in the source version and either
    /// back-filled or left explicitly unavailable. Empty for a record that was
    /// already current — nothing was derived, so there is nothing to disclose.
    pub degradations: Vec<Degradation>,
}

impl Capabilities {
    /// A clean descriptor for a record that needed no back-fill.
    #[must_use]
    pub fn full() -> Self {
        Self {
            clock: ClockCapability::Monotonic,
            degradations: Vec::new(),
        }
    }
}

/// One named thing that was missing in the source version, and how it was handled.
///
/// Naming the missing field (not just a flag) means an operator reading a degraded
/// record knows precisely which capability is reduced and why, without consulting
/// any external table of version differences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degradation {
    /// The field that was absent in the incoming record, e.g. `"monotonic_clock"`.
    pub field: String,
    /// Plain-words description of the consequence, e.g. that attribution must use a
    /// wider wall-clock window because the skew-free monotonic clock is unavailable.
    pub note: String,
}

/// A versioned Rollout-shaped record as it arrives off the wire.
///
/// This is a small, representative envelope (not the full production `Rollout`)
/// that models the one evolution that actually bites: v1 carried only a wall-clock
/// timestamp; v2 added the monotonic `(boot_id, mono_ns)` clock; v3 added the
/// server-anchor offset. Optional fields are `None` precisely when the emitting
/// SDK predates them — that is what lets migration tell "absent" from "zero".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutEnvelope {
    /// The shape this envelope is currently in. Bumped by each migration step.
    pub version: SchemaVersion,
    /// Robot wall-clock estimate (ns since epoch). Present since v1 — every
    /// version has at least this, so it is non-optional.
    pub ts_wall_ns: i64,
    /// Monotonic `(boot_id, mono_ns)` clock. `None` on a v1 record (the SDK had no
    /// monotonic clock); `Some` from v2 on. Migration never fabricates this.
    pub clock: Option<MonoClock>,
    /// Server-anchor offset (ns) aligning this boot's monotonic origin to server
    /// time. `None` before v3; `Some` from v3 on.
    pub server_anchor_offset_ns: Option<i64>,
}

impl RolloutEnvelope {
    /// Build a v1 record — wall-clock only, no monotonic clock, no anchor.
    #[must_use]
    pub fn v1(ts_wall_ns: i64) -> Self {
        Self {
            version: SchemaVersion::new(1),
            ts_wall_ns,
            clock: None,
            server_anchor_offset_ns: None,
        }
    }

    /// Build a current (v3) record — fully populated, nothing to back-fill.
    #[must_use]
    pub fn current(clock: MonoClock, server_anchor_offset_ns: i64) -> Self {
        Self {
            version: CURRENT,
            ts_wall_ns: clock.ts_wall_ns,
            clock: Some(clock),
            server_anchor_offset_ns: Some(server_anchor_offset_ns),
        }
    }
}

/// One ordered forward step: bring a record at `from` up to `from + 1`.
///
/// A migration both mutates the envelope and appends any degradation it introduced
/// to `caps`, so the capability descriptor is built incrementally by the same code
/// that does the back-fill — they can never drift out of sync.
struct Migration {
    /// The version this step upgrades *from*. The runner applies it only when the
    /// envelope is exactly at this version, which is what makes the chain ordered.
    from: SchemaVersion,
    /// The transform. Takes the envelope at `from` and the running capability
    /// descriptor; leaves the envelope at `from + 1`.
    apply: fn(&mut RolloutEnvelope, &mut Capabilities),
}

/// The full ordered chain, lowest version first. Adding a future v4 means adding
/// one entry here; the runner needs no other change.
fn chain() -> Vec<Migration> {
    vec![
        Migration {
            from: SchemaVersion::new(1),
            apply: migrate_v1_to_v2,
        },
        Migration {
            from: SchemaVersion::new(2),
            apply: migrate_v2_to_v3,
        },
    ]
}

/// v1 -> v2: the monotonic clock was added in v2, so a v1 record never had one.
/// We do NOT fabricate a zero `(boot_id, mono_ns)` — that would look like a real
/// skew-free reading to the JOIN. Instead we leave `clock` absent, mark the clock
/// capability `WallOnly`, and record the degradation naming the missing field.
fn migrate_v1_to_v2(env: &mut RolloutEnvelope, caps: &mut Capabilities) {
    caps.clock = ClockCapability::WallOnly;
    caps.degradations.push(Degradation {
        field: "monotonic_clock".to_string(),
        note: "v1 SDK had no (boot_id, mono_ns) clock; attribution must fall back to \
               a wider, lower-trust wall-clock window instead of skew-free monotonic deltas"
            .to_string(),
    });
    env.version = SchemaVersion::new(2);
}

/// v2 -> v3: the server-anchor offset was added in v3. It is computed server-side
/// at ingest, not carried by the robot, so its absence on a v2 record is not a
/// degradation of the record's data — the gateway will stamp it. We mark the field
/// as derived-at-ingest (not fabricated as real robot data) and bump the version.
fn migrate_v2_to_v3(env: &mut RolloutEnvelope, caps: &mut Capabilities) {
    if env.server_anchor_offset_ns.is_none() {
        caps.degradations.push(Degradation {
            field: "server_anchor_offset_ns".to_string(),
            note: "added in v3; absent on this record, to be derived by the ingest \
                   gateway at land time rather than read from the robot"
                .to_string(),
        });
    }
    env.version = SchemaVersion::new(3);
}

/// Run the ordered chain, bringing `env` up to [`CURRENT`] and producing the
/// capability descriptor that records what had to be derived along the way.
///
/// Idempotent by construction: each step fires only when the envelope sits exactly
/// at that step's `from` version, and steps are ordered ascending. So an
/// already-current record matches no step and comes back unchanged with a clean
/// `Capabilities::full()`, while a v1 record walks v1->v2->v3 applying each step
/// exactly once.
#[must_use]
pub fn migrate_to_current(mut env: RolloutEnvelope) -> (RolloutEnvelope, Capabilities) {
    let mut caps = Capabilities::full();
    for step in chain() {
        if env.version == step.from {
            (step.apply)(&mut env, &mut caps);
        }
    }
    (env, caps)
}

/// Re-derive a record's clock capability from its final shape, independent of the
/// migration path. Used so a record that arrived *already* current (a v3 record
/// with a real clock) reports `Monotonic` without having gone through any step.
#[must_use]
pub fn clock_capability_of(env: &RolloutEnvelope) -> ClockCapability {
    match &env.clock {
        Some(_) => ClockCapability::Monotonic,
        None => ClockCapability::WallOnly,
    }
}

/// A throwaway `BootId` is never minted by migration: the absence of a monotonic
/// clock is preserved as `None`, not papered over. This free function exists only
/// so callers constructing test/current envelopes have a clear place to get one.
#[must_use]
pub fn fresh_boot_id() -> BootId {
    BootId::new()
}
