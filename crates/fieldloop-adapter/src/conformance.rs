//! The conformance harness — the headline of this crate.
//!
//! [`check_conformance`] takes any [`EmbodimentAdapter`] and mechanically asserts the
//! contract's invariants, returning a worst-first [`ConformanceReport`]. This is what
//! turns "onboard a new robot type" into a self-checking deliverable: an OEM
//! implements the trait, runs this harness, and a [`ConformanceReport::Conformant`]
//! result is the objective signal that the adapter is correctly built — no eng-team
//! review of bespoke behavior required.
//!
//! The harness is pure: it builds its own fixtures and calls the adapter's pure
//! methods, so it can run anywhere with no I/O.

use std::collections::BTreeMap;

use fieldloop_types::{
    BootId, BoundedBlob, MonoClock, OutcomeEvent, OutcomeKind, RobotId, RobotIdentity, TenantId,
};

use crate::adapter::{ALL_OUTCOME_KINDS, EmbodimentAdapter, RawAction};

/// How bad a conformance failure is. Ordered worst-first when reported.
///
/// A closed, ordered enum so the report can sort failures by severity and a caller
/// can gate on "any Critical" cheaply. `Critical` denotes a violation that would
/// produce a *wrong* result in the live pipeline (a false attribution); `Major`
/// denotes a contract breach that breaks a downstream consumer (an unusable export,
/// a non-total method).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// A violation that would silently produce a wrong result in the live pipeline,
    /// such as binding a tight-timing outcome on a drifting wall-clock.
    Critical,
    /// A contract breach that breaks a downstream consumer but does not by itself
    /// fabricate a wrong attribution.
    Major,
}

impl Severity {
    /// Sort rank, smaller = worse, so failures sort worst-first. Defined explicitly
    /// (rather than relying on the derived enum order) so the intended worst-first
    /// ordering is stated in one place and survives any reordering of the variants.
    fn rank(self) -> u8 {
        match self {
            Severity::Critical => 0,
            Severity::Major => 1,
        }
    }
}

/// One failed conformance check: which check, a human-readable detail, and how bad.
///
/// Carries the detail string so a failure is actionable on its own — the OEM sees
/// exactly which invariant broke and on what input, without needing to re-derive it
/// from a bare pass/fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceFailure {
    /// A short, stable name for the violated check (for example
    /// `collision_colocation`), so failures can be grouped and referenced.
    pub check: &'static str,
    /// A human-readable explanation of what was wrong, including the offending input.
    pub detail: String,
    /// How severe the violation is.
    pub severity: Severity,
}

/// The result of running the conformance harness on an adapter.
///
/// Either fully [`ConformanceReport::Conformant`], or a non-empty, worst-first list
/// of failures. Modeled as an enum (not a `Vec` that happens to be empty) so a caller
/// must explicitly handle the conformant case and cannot mistake "no failures yet
/// collected" for "passed".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformanceReport {
    /// The adapter satisfied every checked invariant.
    Conformant,
    /// The adapter violated one or more invariants, listed worst (Critical) first.
    NonConformant(Vec<ConformanceFailure>),
}

impl ConformanceReport {
    /// Whether the adapter passed every check. A convenience for tests and callers
    /// that only need the boolean verdict.
    #[must_use]
    pub fn is_conformant(&self) -> bool {
        matches!(self, ConformanceReport::Conformant)
    }

    /// The failures, or an empty slice if conformant. Lets a caller inspect specific
    /// failures without matching the enum by hand.
    #[must_use]
    pub fn failures(&self) -> &[ConformanceFailure] {
        match self {
            ConformanceReport::Conformant => &[],
            ConformanceReport::NonConformant(failures) => failures,
        }
    }
}

/// Build a deterministic [`OutcomeEvent`] fixture for a given outcome kind.
///
/// The harness needs an outcome of each kind to exercise [`EmbodimentAdapter::classify`]
/// totality. Fixed field values keep the fixture deterministic so a re-run of the
/// harness produces the same report.
fn fixture_outcome(kind: OutcomeKind) -> OutcomeEvent {
    OutcomeEvent::new(
        RobotIdentity::new(TenantId::new("conformance"), RobotId::new("fixture-robot")),
        MonoClock::new(BootId::new(), 1_000_000, 1_700_000_000_000_000_000),
        kind,
        BoundedBlob::empty(),
    )
}

/// Run the full conformance suite against an adapter.
///
/// Returns [`ConformanceReport::Conformant`] if every invariant holds, else a
/// worst-first list of [`ConformanceFailure`]s. The checks, in the order run:
///   * `embodiment` is non-empty (an unnamed adapter cannot be selected);
///   * every declared [`fieldloop_config::AttributionWindow`] has `window_ms > 0`
///     (a zero window can never bind anything);
///   * the collision (tight-timing) kind sets `requires_monotonic_colocation = true`
///     — **Critical**, because binding a collision on a wall-clock would be a false
///     attribution;
///   * `classify` returns for every outcome kind without panicking (totality);
///   * `detect_success` is total and deterministic on fixtures (same input twice →
///     same verdict, no panic);
///   * `normalize_action` is deterministic and idempotent — **the determinism failure
///     is Critical**, because a non-deterministic normalization would split one
///     action into divergent training samples;
///   * `lerobot_features` is non-empty and every [`crate::FeatureSpec`] is well-formed.
#[must_use]
pub fn check_conformance(adapter: &dyn EmbodimentAdapter) -> ConformanceReport {
    let mut failures: Vec<ConformanceFailure> = Vec::new();

    // --- embodiment() non-empty ---------------------------------------------
    if adapter.embodiment().trim().is_empty() {
        failures.push(ConformanceFailure {
            check: "embodiment_non_empty",
            detail: "embodiment() returned an empty name; an unnamed adapter cannot be selected"
                .to_string(),
            severity: Severity::Major,
        });
    }

    // --- declared attribution windows are positive --------------------------
    for kind in ALL_OUTCOME_KINDS {
        if let Some(window) = adapter.attribution_window(kind)
            && window.window_ms == 0
        {
            failures.push(ConformanceFailure {
                    check: "attribution_window_positive",
                    detail: format!(
                        "attribution_window({kind:?}) has window_ms == 0; a zero window can never bind an outcome"
                    ),
                    severity: Severity::Major,
                });
        }
    }

    // --- collision must be clock-colocated (Critical) -----------------------
    // Binding a collision on a drifting wall-clock would be a false attribution, so a
    // collision that is attributed at all must demand monotonic colocation.
    if adapter.attribution_window(OutcomeKind::Collision).is_some()
        && !adapter.requires_monotonic_colocation(OutcomeKind::Collision)
    {
        failures.push(ConformanceFailure {
            check: "collision_colocation",
            detail:
                "Collision is attributed but requires_monotonic_colocation(Collision) is false; \
                 binding a collision on a wall-clock estimate would be a false attribution"
                    .to_string(),
            severity: Severity::Critical,
        });
    }

    // --- classify totality --------------------------------------------------
    // Calling classify for every outcome kind proves it returns a valid FailureClass
    // for each (the type system guarantees the return is a real class; this proves it
    // does not panic on any kind).
    for kind in ALL_OUTCOME_KINDS {
        let outcome = fixture_outcome(kind);
        let _class = adapter.classify(&outcome);
    }

    // --- detect_success totality + determinism ------------------------------
    // A spread of signals (including extremes and a likely-undecidable mid value)
    // exercises the success path; calling twice proves determinism.
    for signal in [f64::MIN, -1.0, 0.0, 0.5, 1.0, f64::MAX] {
        let first = adapter.detect_success(signal);
        let second = adapter.detect_success(signal);
        if first != second {
            failures.push(ConformanceFailure {
                check: "detect_success_deterministic",
                detail: format!(
                    "detect_success({signal}) returned {first:?} then {second:?}; the success \
                     decision must be deterministic"
                ),
                severity: Severity::Major,
            });
        }
    }

    // --- normalize_action determinism + idempotence (Critical) --------------
    let raw = RawAction::from_pairs([("a", 0.5_f64), ("b", -0.25_f64), ("c", 1.0_f64)]);
    let once = adapter.normalize_action(&raw);
    let twice = adapter.normalize_action(&raw);
    if once != twice {
        failures.push(ConformanceFailure {
            check: "normalize_action_deterministic",
            detail: "normalize_action produced different output for the same raw action; a \
                 non-deterministic normalization splits one action into divergent training samples"
                .to_string(),
            severity: Severity::Critical,
        });
    }
    // Idempotence: feeding the normalized channels back in as a raw action must
    // normalize to the same canonical form, so re-normalizing an already-canonical
    // action is stable.
    let re_raw = RawAction {
        channels: once.channels.clone(),
    };
    let re_normalized = adapter.normalize_action(&re_raw);
    if re_normalized.channels != once.channels || re_normalized.action_space != once.action_space {
        failures.push(ConformanceFailure {
            check: "normalize_action_idempotent",
            detail:
                "normalizing an already-normalized action changed it; normalization must be stable \
                 so a canonical action stays canonical"
                    .to_string(),
            severity: Severity::Major,
        });
    }

    // --- lerobot_features non-empty + well-formed ---------------------------
    let features = adapter.lerobot_features();
    if features.is_empty() {
        failures.push(ConformanceFailure {
            check: "lerobot_features_non_empty",
            detail: "lerobot_features() is empty; an export with no features cannot train a policy"
                .to_string(),
            severity: Severity::Major,
        });
    } else {
        let malformed: BTreeMap<&String, _> = features
            .features
            .iter()
            .filter(|(_, spec)| !spec.is_well_formed())
            .collect();
        for (name, spec) in malformed {
            failures.push(ConformanceFailure {
                check: "lerobot_features_well_formed",
                detail: format!(
                    "feature {name:?} is malformed (dtype={:?}, shape={:?}); a feature needs a \
                     non-empty dtype and no zero-length axis",
                    spec.dtype, spec.shape
                ),
                severity: Severity::Major,
            });
        }
    }

    if failures.is_empty() {
        ConformanceReport::Conformant
    } else {
        // Worst-first: Critical before Major. Stable within a severity so the order of
        // checks above is preserved, keeping reports reproducible.
        failures.sort_by_key(|f| f.severity.rank());
        ConformanceReport::NonConformant(failures)
    }
}
