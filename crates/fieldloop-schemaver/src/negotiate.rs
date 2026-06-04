//! Negotiation: turn a classified version into an accept/reject decision.
//!
//! Classification says *what* a version is; negotiation says *what to do* about
//! it, and is the layer an ingest gateway talks to. Keeping the two separate means
//! the deprecation policy (the bands) can change without touching the accept/reject
//! contract, and the contract is exhaustive: every version produces exactly one of
//! Accept / AcceptWithWarning / Reject, so there is no undefined "silently degrade"
//! path.

use crate::policy::{SupportPolicy, VersionClass};
use crate::version::SchemaVersion;

/// Why a record was refused. Carried in `Reject` so the gateway can log a precise
/// cause (too old vs. future/unknown) instead of a generic failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Below the policy floor — too old for this build to interpret safely.
    BelowMinSupported {
        /// The version the record declared.
        declared: SchemaVersion,
        /// The oldest version the policy still accepts.
        min_supported: SchemaVersion,
    },
    /// Above current — a future or unknown version this build does not understand.
    /// Refusing (rather than guessing) is the safe default for an open schema that
    /// external parties may extend ahead of us.
    FutureOrUnknown {
        /// The version the record declared.
        declared: SchemaVersion,
        /// The newest version this build can produce/understand.
        current: SchemaVersion,
    },
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::BelowMinSupported {
                declared,
                min_supported,
            } => write!(
                f,
                "schema {declared} is below the minimum supported {min_supported}; too old to interpret"
            ),
            RejectReason::FutureOrUnknown { declared, current } => write!(
                f,
                "schema {declared} is newer than current {current}; future/unknown version refused"
            ),
        }
    }
}

/// The migration chain to run as a list of version steps `[v_n -> v_{n+1}, ...]`.
///
/// Exposed in the negotiation result (not just executed silently) so a caller or a
/// test can see exactly which steps a record will go through before its data is
/// touched — the chain is part of the negotiated contract, not a hidden detail.
pub type MigrationPlan = Vec<(SchemaVersion, SchemaVersion)>;

/// The outcome of negotiating one declared version against the policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Negotiation {
    /// Accept and run `migrations` (empty when the record is already current).
    Accept {
        /// The ordered steps to bring the record to current.
        migrations: MigrationPlan,
    },
    /// Accept and run `migrations`, but the sender is in the deprecation band and
    /// must upgrade — `warning` is the message the gateway should surface/log.
    AcceptWithWarning {
        /// Human-readable upgrade notice.
        warning: String,
        /// The ordered steps to bring the record to current.
        migrations: MigrationPlan,
    },
    /// Refuse the record; `reason` says precisely why.
    Reject {
        /// The precise cause of refusal.
        reason: RejectReason,
    },
}

/// Build the ordered step list from `declared` up to `policy.current`.
fn plan_from(declared: SchemaVersion, current: SchemaVersion) -> MigrationPlan {
    (declared.major()..current.major())
        .map(|n| (SchemaVersion::new(n), SchemaVersion::new(n + 1)))
        .collect()
}

/// Decide what to do with a declared version under `policy`.
///
/// The mapping is fixed: Current accepts with no migrations; Supported accepts with
/// the full forward chain; Deprecated accepts the same chain but attaches an upgrade
/// warning so the still-working old sender is put on notice; Unsupported rejects
/// with the specific cause. This total mapping is what replaces the old undefined
/// "silently degrades to what?" behavior with one explicit decision per version.
#[must_use]
pub fn negotiate(declared: SchemaVersion, policy: &SupportPolicy) -> Negotiation {
    match policy.classify(declared) {
        VersionClass::Current => Negotiation::Accept {
            migrations: Vec::new(),
        },
        VersionClass::Supported => Negotiation::Accept {
            migrations: plan_from(declared, policy.current),
        },
        VersionClass::Deprecated => Negotiation::AcceptWithWarning {
            warning: format!(
                "schema {declared} is deprecated (still accepted, will be migrated to \
                 {current}); upgrade the emitting SDK before {min} is retired",
                current = policy.current,
                min = policy.min_supported
            ),
            migrations: plan_from(declared, policy.current),
        },
        VersionClass::Unsupported => {
            let reason = if declared > policy.current {
                RejectReason::FutureOrUnknown {
                    declared,
                    current: policy.current,
                }
            } else {
                RejectReason::BelowMinSupported {
                    declared,
                    min_supported: policy.min_supported,
                }
            };
            Negotiation::Reject { reason }
        }
    }
}
