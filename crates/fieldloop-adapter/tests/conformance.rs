//! Conformance suite tests: the reference adapter passes, and deliberately broken
//! adapters each fail the specific check that catches their defect — proving the
//! harness detects real errors, not just that it accepts a good adapter.

use std::cell::Cell;

use fieldloop_adapter::{
    EmbodimentAdapter, FeatureSpec, LeRobotFeatures, NormalizedAction, RawAction, Severity,
    SixDofArmAdapter, SuccessVerdict, check_conformance, select_adapter,
};
use fieldloop_config::{ActionSpace, AttributionWindow};
use fieldloop_types::{FailureClass, OutcomeEvent, OutcomeKind};

// ---- the reference adapter is fully conformant -----------------------------

#[test]
fn reference_adapter_is_conformant() {
    let report = check_conformance(&SixDofArmAdapter);
    assert!(
        report.is_conformant(),
        "reference adapter must pass the full suite, got: {report:?}"
    );
}

// ---- a builder so each broken adapter overrides one behavior at a time -----

/// A configurable adapter that starts identical to the reference adapter and lets a
/// single behavior be broken, so each test isolates exactly one defect and proves the
/// harness flags that specific check.
struct BrokenAdapter {
    collision_colocated: bool,
    collision_window_ms: u64,
    deterministic_normalize: bool,
    features: LeRobotFeatures,
}

impl BrokenAdapter {
    fn good() -> Self {
        Self {
            collision_colocated: true,
            collision_window_ms: 200,
            deterministic_normalize: true,
            features: SixDofArmAdapter.lerobot_features(),
        }
    }
}

impl EmbodimentAdapter for BrokenAdapter {
    fn embodiment(&self) -> &str {
        "broken"
    }
    fn action_space(&self) -> ActionSpace {
        ActionSpace::JointPosition
    }
    fn attribution_window(&self, kind: OutcomeKind) -> Option<AttributionWindow> {
        match kind {
            OutcomeKind::Collision => Some(AttributionWindow {
                window_ms: self.collision_window_ms,
                requires_monotonic_colocation: self.collision_colocated,
            }),
            OutcomeKind::Heartbeat => None,
            _ => Some(AttributionWindow {
                window_ms: 1_000,
                requires_monotonic_colocation: false,
            }),
        }
    }
    fn requires_monotonic_colocation(&self, kind: OutcomeKind) -> bool {
        match kind {
            OutcomeKind::Collision => self.collision_colocated,
            _ => false,
        }
    }
    fn classify(&self, _outcome: &OutcomeEvent) -> FailureClass {
        FailureClass::Hardware
    }
    fn detect_success(&self, signal: f64) -> SuccessVerdict {
        if signal <= 0.1 {
            SuccessVerdict::Success
        } else {
            SuccessVerdict::Unknown
        }
    }
    fn normalize_action(&self, raw: &RawAction) -> NormalizedAction {
        let mut channels = raw.channels.clone();
        if !self.deterministic_normalize {
            // A non-deterministic mutation: each call perturbs the output, so two
            // calls on the same input diverge. Uses interior state via a thread-local
            // counter so the value changes call-to-call without needing &mut self.
            thread_local!(static COUNTER: Cell<u64> = const { Cell::new(0) });
            let n = COUNTER.with(|c| {
                let v = c.get();
                c.set(v + 1);
                v
            });
            channels.insert("nondet".to_string(), n as f64);
        }
        NormalizedAction {
            action_space: self.action_space(),
            channels,
        }
    }
    fn lerobot_features(&self) -> LeRobotFeatures {
        self.features.clone()
    }
}

fn has_check(report: &fieldloop_adapter::ConformanceReport, check: &str, sev: Severity) -> bool {
    report
        .failures()
        .iter()
        .any(|f| f.check == check && f.severity == sev)
}

// (a) collision window with colocation = false -> Critical conformance failure

#[test]
fn collision_without_colocation_is_critical() {
    let mut a = BrokenAdapter::good();
    a.collision_colocated = false;
    let report = check_conformance(&a);
    assert!(!report.is_conformant());
    assert!(
        has_check(&report, "collision_colocation", Severity::Critical),
        "expected a Critical collision_colocation failure, got: {report:?}"
    );
    // Worst-first: the Critical failure must sort ahead of any Major one.
    assert_eq!(report.failures()[0].severity, Severity::Critical);
}

// (b) zero window_ms -> failure

#[test]
fn zero_window_ms_fails() {
    let mut a = BrokenAdapter::good();
    a.collision_window_ms = 0;
    let report = check_conformance(&a);
    assert!(has_check(
        &report,
        "attribution_window_positive",
        Severity::Major
    ));
}

// (c) non-deterministic normalize_action -> determinism failure (Critical)

#[test]
fn non_deterministic_normalize_is_critical() {
    let mut a = BrokenAdapter::good();
    a.deterministic_normalize = false;
    let report = check_conformance(&a);
    assert!(
        has_check(
            &report,
            "normalize_action_deterministic",
            Severity::Critical
        ),
        "expected a Critical determinism failure, got: {report:?}"
    );
}

// (d) empty lerobot_features() -> failure

#[test]
fn empty_features_fails() {
    let mut a = BrokenAdapter::good();
    a.features = LeRobotFeatures::from_pairs::<_, String>([]);
    let report = check_conformance(&a);
    assert!(has_check(
        &report,
        "lerobot_features_non_empty",
        Severity::Major
    ));
}

#[test]
fn malformed_feature_fails() {
    let mut a = BrokenAdapter::good();
    // A zero-length axis is an untrainable feature.
    a.features = LeRobotFeatures::from_pairs([("action", FeatureSpec::new("float32", vec![0]))]);
    let report = check_conformance(&a);
    assert!(has_check(
        &report,
        "lerobot_features_well_formed",
        Severity::Major
    ));
}

// ---- classify / detect_success totality ------------------------------------

#[test]
fn classify_is_total_over_all_outcome_kinds() {
    use fieldloop_types::{BootId, BoundedBlob, MonoClock, RobotId, RobotIdentity, TenantId};
    let adapter = SixDofArmAdapter;
    for kind in [
        OutcomeKind::TeleopTakeover,
        OutcomeKind::EStop,
        OutcomeKind::Collision,
        OutcomeKind::DownstreamFailure,
        OutcomeKind::Heartbeat,
    ] {
        let outcome = OutcomeEvent::new(
            RobotIdentity::new(TenantId::new("t"), RobotId::new("r")),
            MonoClock::new(BootId::new(), 1, 1),
            kind,
            BoundedBlob::empty(),
        );
        // Returns a valid FailureClass for every kind without panicking.
        let _class: FailureClass = adapter.classify(&outcome);
    }
}

#[test]
fn detect_success_is_total_and_stable() {
    let adapter = SixDofArmAdapter;
    for signal in [
        f64::NAN,
        f64::MIN,
        -5.0,
        0.0,
        0.05,
        0.5,
        0.95,
        1.0,
        f64::MAX,
    ] {
        let first = adapter.detect_success(signal);
        let second = adapter.detect_success(signal);
        assert_eq!(first, second, "detect_success must be stable for {signal}");
    }
    // The documented decision boundaries hold.
    assert_eq!(adapter.detect_success(0.05), SuccessVerdict::Success);
    assert_eq!(adapter.detect_success(0.95), SuccessVerdict::Failure);
    assert_eq!(adapter.detect_success(0.5), SuccessVerdict::Unknown);
    assert_eq!(adapter.detect_success(f64::NAN), SuccessVerdict::Unknown);
}

// ---- registry --------------------------------------------------------------

#[test]
fn registry_selects_reference_by_name_and_rejects_unknown() {
    let selected = select_adapter("six_dof_arm").expect("reference adapter must be selectable");
    assert_eq!(selected.embodiment(), "six_dof_arm");
    // A selected adapter is itself conformant.
    assert!(check_conformance(selected.as_ref()).is_conformant());

    assert!(
        select_adapter("no_such_robot").is_none(),
        "an unknown embodiment must not resolve to an adapter"
    );
}
