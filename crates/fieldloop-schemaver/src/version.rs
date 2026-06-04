//! The schema version number itself.
//!
//! A fleet runs SDK versions years apart, so the very first thing the ingest
//! gateway must do with a record is decide *which shape* it is in. That decision
//! needs a single, totally-ordered version number — a plain `u32` major. We use a
//! u32 (not a semver triple) because schema evolution here is append-only and
//! linear: each bump adds fields, never branches, so one ordered integer is
//! enough and keeps the comparison/migration logic trivial and unambiguous.

use serde::{Deserialize, Serialize};

/// A schema version: a single, totally-ordered major number.
///
/// Ordering is the whole point — classification and migration both work by
/// comparing this against the supported range, so deriving `Ord` here is what
/// makes "is this record older than current?" a one-line, unambiguous check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    /// Construct a version from its raw major number.
    #[must_use]
    pub const fn new(major: u32) -> Self {
        Self(major)
    }

    /// The raw major number, for arithmetic when stepping a migration chain.
    #[must_use]
    pub const fn major(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The version this build of Fieldloop emits and migrates everything *up to*.
///
/// Pinned as a constant so there is exactly one source of truth for "current":
/// the policy, the migration chain, and every test all compare against this same
/// value, so they can never disagree about what the current shape is. v1 had only
/// a wall-clock; v2 added the monotonic `(boot_id, mono_ns)` clock; v3 (current)
/// added a server-anchored ingest offset.
pub const CURRENT: SchemaVersion = SchemaVersion(3);
