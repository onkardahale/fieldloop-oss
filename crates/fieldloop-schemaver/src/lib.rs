//! Schema-version negotiation, forward-migration, and the deprecation policy for
//! records arriving from robots on old SDKs.
//!
//! A fleet runs SDK versions years apart. An old robot emits a record missing
//! fields the schema added later (notably the monotonic `(boot_id, mono_ns)`
//! clock). Without a policy, that record "silently degrades to *what?*" — and
//! external parties pinning the open schema have no versioning contract to rely on.
//! This crate makes the handling explicit end to end:
//!
//! 1. [`SchemaVersion`] / [`CURRENT`] — the ordered version number this build emits.
//! 2. [`SupportPolicy`] / [`SupportPolicy::classify`] — the public deprecation
//!    policy: which versions are current, supported, deprecated, or unsupported.
//! 3. [`negotiate`] — turns a declared version into accept / accept-with-warning /
//!    reject, with the exact migration chain.
//! 4. [`migrate_to_current`] — runs the ordered, idempotent forward chain, filling
//!    absent fields with *explicit capability/degradation markers* rather than fake
//!    zeros, so a missing monotonic clock surfaces as [`ClockCapability::WallOnly`].
//! 5. [`ingest_negotiate`] — the single call an ingest gateway makes per record:
//!    classify, then either reject or migrate-with-capabilities.

pub mod migrate;
pub mod negotiate;
pub mod policy;
pub mod version;

pub use migrate::{
    Capabilities, ClockCapability, Degradation, RolloutEnvelope, clock_capability_of,
    migrate_to_current,
};
pub use negotiate::{MigrationPlan, Negotiation, RejectReason, negotiate};
pub use policy::{SupportPolicy, VersionClass};
pub use version::{CURRENT, SchemaVersion};

/// A record that has been accepted and brought up to the current schema shape,
/// together with the explicit statement of what it can and cannot do.
///
/// `warnings` is non-empty when the record came in on a deprecated version: the
/// data is usable but the sender must upgrade, and the gateway should log/surface
/// these. `capabilities` is the load-bearing field — it tells the JOIN, per record,
/// whether it may rely on the skew-free monotonic clock or must degrade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigratedRecord {
    /// The record migrated to [`CURRENT`].
    pub envelope_at_current: RolloutEnvelope,
    /// What the migrated record actually supports (clock authority + any
    /// derived/missing fields).
    pub capabilities: Capabilities,
    /// Upgrade notices to surface (empty unless the source version was deprecated).
    pub warnings: Vec<String>,
}

/// The one call an ingest gateway makes per incoming record.
///
/// It negotiates `declared` against `policy`, and on accept runs the forward
/// migration on `raw_envelope`, returning the current-shape record plus its
/// capability descriptor; on reject it returns the precise [`RejectReason`]. This
/// is deterministic and side-effect-free, so the same record always yields the same
/// decision — there is no hidden "silently degrade" branch.
///
/// # Errors
/// Returns [`RejectReason`] when `declared` is below the supported floor or above
/// the current version (future/unknown), i.e. too old or too new to interpret.
pub fn ingest_negotiate(
    declared: SchemaVersion,
    raw_envelope: RolloutEnvelope,
    policy: &SupportPolicy,
) -> Result<MigratedRecord, RejectReason> {
    match negotiate(declared, policy) {
        Negotiation::Reject { reason } => Err(reason),
        Negotiation::Accept { .. } => {
            let (envelope_at_current, capabilities) = migrate_to_current(raw_envelope);
            Ok(MigratedRecord {
                envelope_at_current,
                capabilities,
                warnings: Vec::new(),
            })
        }
        Negotiation::AcceptWithWarning { warning, .. } => {
            let (envelope_at_current, capabilities) = migrate_to_current(raw_envelope);
            Ok(MigratedRecord {
                envelope_at_current,
                capabilities,
                warnings: vec![warning],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_types::MonoClock;

    /// A policy with current=v3, floor=v1, and a 1-wide deprecation band: v1 is
    /// deprecated, v2 is plainly supported, v3 is current.
    fn policy() -> SupportPolicy {
        SupportPolicy {
            min_supported: SchemaVersion::new(1),
            current: SchemaVersion::new(3),
            deprecation_band: 1,
        }
    }

    fn current_envelope() -> RolloutEnvelope {
        let clock = MonoClock::new(migrate::fresh_boot_id(), 42, 1_000);
        RolloutEnvelope::current(clock, 7)
    }

    // ---- classification -----------------------------------------------------

    #[test]
    fn classify_current_is_current() {
        assert_eq!(
            policy().classify(SchemaVersion::new(3)),
            VersionClass::Current
        );
    }

    #[test]
    fn classify_one_below_current_in_range_is_supported() {
        assert_eq!(
            policy().classify(SchemaVersion::new(2)),
            VersionClass::Supported
        );
    }

    #[test]
    fn classify_in_deprecation_band_is_deprecated() {
        assert_eq!(
            policy().classify(SchemaVersion::new(1)),
            VersionClass::Deprecated
        );
    }

    #[test]
    fn classify_below_min_is_unsupported() {
        assert_eq!(
            policy().classify(SchemaVersion::new(0)),
            VersionClass::Unsupported
        );
    }

    #[test]
    fn classify_future_unknown_is_unsupported() {
        assert_eq!(
            policy().classify(SchemaVersion::new(9)),
            VersionClass::Unsupported
        );
    }

    // ---- negotiate ----------------------------------------------------------

    #[test]
    fn negotiate_current_accepts_no_migrations() {
        assert_eq!(
            negotiate(SchemaVersion::new(3), &policy()),
            Negotiation::Accept { migrations: vec![] }
        );
    }

    #[test]
    fn negotiate_supported_accepts_with_chain() {
        // v2 -> v3 is a single step.
        assert_eq!(
            negotiate(SchemaVersion::new(2), &policy()),
            Negotiation::Accept {
                migrations: vec![(SchemaVersion::new(2), SchemaVersion::new(3))]
            }
        );
    }

    #[test]
    fn negotiate_deprecated_accepts_with_warning_and_chain() {
        match negotiate(SchemaVersion::new(1), &policy()) {
            Negotiation::AcceptWithWarning {
                warning,
                migrations,
            } => {
                assert!(warning.contains("deprecated"));
                assert_eq!(
                    migrations,
                    vec![
                        (SchemaVersion::new(1), SchemaVersion::new(2)),
                        (SchemaVersion::new(2), SchemaVersion::new(3)),
                    ]
                );
            }
            other => panic!("expected AcceptWithWarning, got {other:?}"),
        }
    }

    #[test]
    fn negotiate_too_old_rejects() {
        match negotiate(SchemaVersion::new(0), &policy()) {
            Negotiation::Reject {
                reason: RejectReason::BelowMinSupported { .. },
            } => {}
            other => panic!("expected BelowMinSupported reject, got {other:?}"),
        }
    }

    #[test]
    fn negotiate_future_rejects() {
        match negotiate(SchemaVersion::new(9), &policy()) {
            Negotiation::Reject {
                reason: RejectReason::FutureOrUnknown { .. },
            } => {}
            other => panic!("expected FutureOrUnknown reject, got {other:?}"),
        }
    }

    // ---- explicit degradation (the headline) --------------------------------

    #[test]
    fn v1_record_migrates_to_wall_only_not_a_fake_zero_clock() {
        let (env, caps) = migrate_to_current(RolloutEnvelope::v1(123));
        // Landed on the current shape.
        assert_eq!(env.version, CURRENT);
        // The monotonic clock was NOT fabricated — it is still absent, preserved
        // honestly rather than back-filled with a fake (boot_id, mono_ns) zero.
        assert!(
            env.clock.is_none(),
            "v1 migration must NOT fabricate a zero monotonic clock"
        );
        // Capability is explicitly WallOnly.
        assert_eq!(caps.clock, ClockCapability::WallOnly);
        // And there is a degradation entry naming the missing monotonic clock.
        assert!(
            caps.degradations
                .iter()
                .any(|d| d.field == "monotonic_clock"),
            "must record a degradation naming the missing monotonic clock"
        );
    }

    #[test]
    fn current_record_has_monotonic_capability_no_degradation() {
        let (env, caps) = migrate_to_current(current_envelope());
        assert_eq!(env.version, CURRENT);
        assert_eq!(caps.clock, ClockCapability::Monotonic);
        assert!(
            caps.degradations.is_empty(),
            "an already-current record derives nothing, so discloses no degradation"
        );
        assert_eq!(clock_capability_of(&env), ClockCapability::Monotonic);
    }

    // ---- ordering + idempotence ---------------------------------------------

    #[test]
    fn migration_is_idempotent_on_current_record() {
        let start = current_envelope();
        let (once, _) = migrate_to_current(start.clone());
        let (twice, caps_twice) = migrate_to_current(once.clone());
        // Re-running the chain on an already-current record is a no-op.
        assert_eq!(start, once);
        assert_eq!(once, twice);
        assert!(caps_twice.degradations.is_empty());
    }

    #[test]
    fn v1_to_current_applies_each_step_once_and_lands_current() {
        let (env, caps) = migrate_to_current(RolloutEnvelope::v1(123));
        assert_eq!(env.version, CURRENT);
        // Wall-clock carried through from v1 untouched.
        assert_eq!(env.ts_wall_ns, 123);
        // Two steps ran (v1->v2 missing clock, v2->v3 missing anchor): two
        // distinct degradation fields, each recorded exactly once.
        let fields: Vec<&str> = caps.degradations.iter().map(|d| d.field.as_str()).collect();
        assert!(fields.contains(&"monotonic_clock"));
        assert!(fields.contains(&"server_anchor_offset_ns"));
        assert_eq!(fields.len(), 2, "each step contributes exactly once");
    }

    // ---- ingest_negotiate entry point ---------------------------------------

    #[test]
    fn ingest_negotiate_accepts_supported_old_record_with_capabilities() {
        // A v1 (deprecated) record is accepted and migrated, with a warning.
        let rec = ingest_negotiate(SchemaVersion::new(1), RolloutEnvelope::v1(50), &policy())
            .expect("v1 is in the supported range");
        assert_eq!(rec.envelope_at_current.version, CURRENT);
        assert_eq!(rec.capabilities.clock, ClockCapability::WallOnly);
        assert!(!rec.warnings.is_empty(), "deprecated source must warn");
    }

    #[test]
    fn ingest_negotiate_rejects_unsupported_record() {
        let err = ingest_negotiate(SchemaVersion::new(9), current_envelope(), &policy())
            .expect_err("v9 is in the future/unknown");
        assert!(matches!(err, RejectReason::FutureOrUnknown { .. }));
    }
}
