//! Type-safe id newtypes.
//!
//! Every id in the canonical schema is a UUIDv7 wrapped in a *distinct* newtype.
//! Reusing one bare UUID type for every id and guarding mix-ups with a runtime
//! check leaves room for a swapped id to slip through; Fieldloop's attribution is
//! higher-ambiguity, so a wrong id must fail loudly and early. These newtypes make
//! a swap a **compile error**: you cannot pass a [`RolloutId`] where an
//! [`EpisodeId`] is expected.
//!
//! ## The embedded v7 timestamp is ADVISORY ONLY
//! The embedded v7 timestamp gives uniqueness and a coarse, time-ordered sort, and
//! nothing more. Never use it for attribution, cooldown, or any ordering that must
//! be correct: robot wall-clocks drift, so the timestamp baked into an id minted
//! on a robot is unreliable. Time-ordering authority is the monotonic
//! `(boot_id, mono_ns)` clock within a boot, and server-anchored ingest time across
//! boots/robots (see [`crate::clock`]). Do not read the timestamp back out of these
//! ids to make any such decision.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Error returned when parsing an id newtype from a string fails.
#[derive(Debug, thiserror::Error)]
#[error("invalid {kind} (expected a UUID): {source}")]
pub struct IdParseError {
    /// The newtype that failed to parse (e.g. `"RolloutId"`).
    pub kind: &'static str,
    /// The underlying uuid parse error.
    #[source]
    pub source: uuid::Error,
}

/// Defines a transparent UUIDv7 newtype with the full id surface:
/// `new()` (= `Uuid::now_v7()`), `Display`, `FromStr`, serde transparency, and
/// the standard equality/hash/ordering derives so ids work as map/set keys and
/// in `ORDER BY`-style coarse sorts.
///
/// `#[serde(transparent)]` means the wire form is exactly the inner UUID string,
/// so these newtypes are drop-in compatible with the bare-`UUID` columns in the
/// ClickHouse/Postgres storage schemas — the type safety is a Rust-side guarantee
/// that costs nothing on the wire.
macro_rules! uuid_v7_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Mint a fresh id. Uses `Uuid::now_v7()` so ids are globally unique
            /// and coarsely time-sortable — but the embedded timestamp is advisory
            /// (robot wall-clocks drift) and must never drive attribution or
            /// cooldown; that is the monotonic clock's job.
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }

            /// Wrap an already-existing UUID (e.g. read back from storage or the
            /// wire). No validation that it is v7 — storage is the source of an
            /// id we previously minted.
            #[must_use]
            pub const fn from_uuid(id: Uuid) -> Self {
                Self(id)
            }

            /// The underlying UUID. Use for storage encoding (e.g. `toUInt128`)
            /// only — never to extract the advisory, drift-prone embedded
            /// timestamp for attribution.
            #[must_use]
            pub const fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            /// Defaults to a freshly-minted id (not nil), so a `..Default::default()`
            /// never silently produces a colliding all-zero id.
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl FromStr for $name {
            type Err = IdParseError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Uuid::from_str(s)
                    .map(Self)
                    .map_err(|source| IdParseError { kind: stringify!($name), source })
            }
        }

        impl From<Uuid> for $name {
            fn from(id: Uuid) -> Self {
                Self(id)
            }
        }

        impl From<$name> for Uuid {
            fn from(id: $name) -> Uuid {
                id.0
            }
        }
    };
}

uuid_v7_newtype! {
    /// Identifies a single deployed-policy inference step — the [`crate::Rollout`]
    /// atom. Minted on the robot at inference start, inside the 50Hz control loop
    /// (so minting must stay lock-free and allocation-free), and threaded onto
    /// outbound teleop/robot messages so a later outcome can be attributed back to
    /// it.
    RolloutId
}

uuid_v7_newtype! {
    /// Identifies a trajectory/run: the set of rollouts sharing it form one
    /// episode. The episode is *implicit* — derived by grouping rollouts on this
    /// id, never written up-front. This avoids any "open an episode row" step in
    /// the hot path; the episode rollup is computed later by the JOIN layer.
    EpisodeId
}

uuid_v7_newtype! {
    /// Identifies an immutable raw [`crate::OutcomeEvent`]: the observed or
    /// synthesized signal *before* it has been attributed to any rollout.
    OutcomeId
}

uuid_v7_newtype! {
    /// Identifies one [`crate::Feedback`] row. Minted as a **fresh**
    /// `Uuid::now_v7()` per row in real wall time — never hash-derived. A hash-seeded
    /// v7 would fix the timestamp, breaking "latest write wins" ordering; instead a
    /// re-attribution mints a genuinely-newer id so latest-wins correctly picks it.
    /// Idempotency lives in the separate `dedup_key` string, not in this id.
    FeedbackId
}

uuid_v7_newtype! {
    /// Robot boot session id. `mono_ns` readings are only comparable **within the
    /// same `BootId`** — a reboot resets `CLOCK_MONOTONIC` back to an arbitrary
    /// origin, so subtracting `mono_ns` across boots is meaningless. Carried in the
    /// [`crate::clock::MonoClock`] stamp on every Rollout/OutcomeEvent.
    BootId
}

uuid_v7_newtype! {
    /// Identifies one eval run window. Present only during an A/B eval; pairs with
    /// an opaque `arm_label` so the candidate's `policy_version` stays blinded
    /// while the deploy gate compares arms.
    EvalRunId
}
