//! A reference adapter — a simple, correct 6-DOF joint-position arm.
//!
//! This is both a template a new OEM can copy and a fixture the conformance tests run
//! against. It is intentionally minimal and obviously correct so that "what a
//! conformant adapter looks like" is concrete, and so the harness has a known-good
//! input to prove it accepts a valid adapter (not just that it rejects broken ones).

use fieldloop_config::{ActionSpace, AttributionWindow};
use fieldloop_types::{FailureClass, OutcomeEvent, OutcomeKind};

use crate::adapter::{
    EmbodimentAdapter, FeatureSpec, LeRobotFeatures, NormalizedAction, RawAction, SuccessVerdict,
};

/// A reference 6-DOF joint-position arm adapter.
///
/// A zero-sized type: this adapter holds no state because all of its behavior is a
/// pure function of its inputs, which is exactly the determinism the contract
/// requires. A real OEM adapter might hold calibration constants, but it must remain
/// equally pure.
#[derive(Debug, Clone, Copy, Default)]
pub struct SixDofArmAdapter;

impl EmbodimentAdapter for SixDofArmAdapter {
    fn embodiment(&self) -> &str {
        "six_dof_arm"
    }

    fn action_space(&self) -> ActionSpace {
        // A position-controlled arm: commands target absolute joint angles.
        ActionSpace::JointPosition
    }

    fn attribution_window(&self, kind: OutcomeKind) -> Option<AttributionWindow> {
        // Window lengths reflect how quickly each kind follows the rollout that caused
        // it: a collision is near-instant and tight, a downstream failure can lag.
        match kind {
            OutcomeKind::Collision => Some(AttributionWindow {
                window_ms: 200,
                // A collision is tight-timing: only bind it when clock-colocated.
                requires_monotonic_colocation: true,
            }),
            OutcomeKind::EStop => Some(AttributionWindow {
                window_ms: 500,
                requires_monotonic_colocation: true,
            }),
            OutcomeKind::TeleopTakeover => Some(AttributionWindow {
                window_ms: 2_000,
                requires_monotonic_colocation: false,
            }),
            OutcomeKind::DownstreamFailure => Some(AttributionWindow {
                window_ms: 10_000,
                requires_monotonic_colocation: false,
            }),
            // Heartbeat is not attributed to a rollout: it is a coverage signal whose
            // presence proves a window was observed, so it has no attribution window.
            OutcomeKind::Heartbeat => None,
        }
    }

    fn requires_monotonic_colocation(&self, kind: OutcomeKind) -> bool {
        // Mirrors the declared windows: the tight-timing kinds (collision, e-stop)
        // demand a clock-safe comparison; the looser kinds may match on the
        // wall-clock estimate.
        matches!(kind, OutcomeKind::Collision | OutcomeKind::EStop)
    }

    fn classify(&self, outcome: &OutcomeEvent) -> FailureClass {
        // A coarse, total hint from the outcome kind alone — a seed for triage that a
        // human label can override. Total over every kind so failure analysis always
        // has a class.
        match outcome.outcome_kind {
            OutcomeKind::Collision => FailureClass::Manipulation,
            OutcomeKind::EStop => FailureClass::Hardware,
            OutcomeKind::TeleopTakeover => FailureClass::Operator,
            OutcomeKind::DownstreamFailure => FailureClass::Environment,
            // A heartbeat is not itself a failure; absent a better signal it maps to
            // the environment bucket as a neutral default rather than panicking.
            OutcomeKind::Heartbeat => FailureClass::Environment,
        }
    }

    fn detect_success(&self, signal: f64) -> SuccessVerdict {
        // The signal is a goal-distance in [0, 1]: small means at goal (success),
        // large means far (failure), and a NaN or out-of-range reading is undecidable
        // rather than forced into a verdict. Total and panic-free for every f64.
        if signal.is_nan() || !(0.0..=1.0).contains(&signal) {
            SuccessVerdict::Unknown
        } else if signal <= 0.1 {
            SuccessVerdict::Success
        } else if signal >= 0.9 {
            SuccessVerdict::Failure
        } else {
            SuccessVerdict::Unknown
        }
    }

    fn normalize_action(&self, raw: &RawAction) -> NormalizedAction {
        // Canonicalize by clamping each joint command to the arm's normalized [-1, 1]
        // range and re-keying into the sorted channel map. This is a pure function of
        // the input and is idempotent: an already-clamped value clamps to itself, so
        // re-normalizing a canonical action yields the same action.
        let channels = raw
            .channels
            .iter()
            .map(|(name, &value)| (name.clone(), value.clamp(-1.0, 1.0)))
            .collect();
        NormalizedAction {
            action_space: self.action_space(),
            channels,
        }
    }

    fn lerobot_features(&self) -> LeRobotFeatures {
        // The minimal training schema for a 6-DOF arm: the 6-joint observation and
        // action vectors plus the scalar success label. Non-empty and well-formed, so
        // the export can materialize against it.
        LeRobotFeatures::from_pairs([
            ("observation.state", FeatureSpec::new("float32", vec![6])),
            ("action", FeatureSpec::new("float32", vec![6])),
            ("next.success", FeatureSpec::scalar("bool")),
        ])
    }
}
