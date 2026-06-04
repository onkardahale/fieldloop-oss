//! The public schema deprecation policy — the bands nobody used to own.
//!
//! External parties pin the open schema, and the fleet emits years-old versions,
//! so "which versions do we still accept, and which are on notice?" must be a
//! written-down, testable policy rather than scattered `if` checks. This module
//! is that policy: a `SupportPolicy` defines the accepted range and a deprecation
//! band, and `classify` maps any incoming version into exactly one explicit
//! class. Making the bands a struct (not constants) means a deployment can widen
//! or tighten support without code changes, and tests can pin specific bands.

use crate::version::SchemaVersion;

/// How a declared version sits relative to the policy. Exactly one variant
/// applies to any version, so downstream handling is total: there is no
/// unclassified case that could "silently degrade to what?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionClass {
    /// Exactly the current shape — no migration needed.
    Current,
    /// In the supported range and older than current — accepted, will be migrated
    /// forward. Not in the deprecation band, so no upgrade nag is warranted yet.
    Supported,
    /// In the deprecation band just above `min_supported` — still accepted and
    /// migrated, but the sender is on notice that this version will stop being
    /// accepted, so the gateway must surface a warning.
    Deprecated,
    /// Below `min_supported` (too old to safely interpret) OR above `current` (a
    /// future/unknown version this build cannot understand). Either way it must be
    /// rejected rather than guessed at.
    Unsupported,
}

/// The accepted version range plus the deprecation band — the deprecation policy.
///
/// `min_supported..=current` is the accepted range. The lowest `deprecation_band`
/// versions of that range (starting at `min_supported`) are *deprecated*: still
/// accepted, but flagged for upgrade. Keeping the band as an explicit width lets
/// operators say "the oldest N still-supported versions are on notice" in one
/// number, which is the whole point of owning the policy in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportPolicy {
    /// Oldest version still accepted. Anything below this is `Unsupported`.
    pub min_supported: SchemaVersion,
    /// The version this build emits. Anything above this is `Unsupported` (a
    /// future version we cannot interpret).
    pub current: SchemaVersion,
    /// How many versions, counting up from `min_supported`, are `Deprecated`
    /// (accepted-with-warning). A band of 1 deprecates only `min_supported`
    /// itself; the current version is never deprecated.
    pub deprecation_band: u32,
}

impl SupportPolicy {
    /// Classify a declared version into exactly one [`VersionClass`].
    ///
    /// The order of checks matters and is deliberate: a future/unknown version
    /// (above current) is rejected before anything else, then exact-current, then
    /// the too-old floor, then the deprecation band, leaving `Supported` as the
    /// in-range remainder. Written this way so each class has one obvious cause.
    #[must_use]
    pub fn classify(&self, v: SchemaVersion) -> VersionClass {
        if v > self.current || v < self.min_supported {
            // Future/unknown, or below the floor: cannot be safely interpreted.
            return VersionClass::Unsupported;
        }
        if v == self.current {
            return VersionClass::Current;
        }
        // In range and older than current. Is it inside the deprecation band?
        let band_top = self.min_supported.major() + self.deprecation_band; // exclusive
        if v.major() < band_top {
            VersionClass::Deprecated
        } else {
            VersionClass::Supported
        }
    }
}
