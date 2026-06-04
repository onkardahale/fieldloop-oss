//! Runnable example: a small adapter for a warehouse pick arm.
//!
//! Run with:
//! `cargo run -p fieldloop-adapter --example onboard_your_robot`

use std::process::ExitCode;

use fieldloop_adapter::{
    ConformanceReport, EmbodimentAdapter, FeatureSpec, LeRobotFeatures, NormalizedAction,
    RawAction, SuccessVerdict, check_conformance,
};
use fieldloop_config::{ActionSpace, AttributionWindow};
use fieldloop_types::{FailureClass, OutcomeEvent, OutcomeKind};

/// A toy but realistic warehouse pick-arm adapter.
///
/// The success signal this example expects is a single scalar composed by the caller:
/// normalized goal distance plus a penalty when the gripper is not holding the part.
/// Small is good, large is bad, and the adapter turns that scalar into the closed
/// three-way `SuccessVerdict`.
#[derive(Debug, Clone, Copy, Default)]
struct WarehousePickArmAdapter;

impl EmbodimentAdapter for WarehousePickArmAdapter {
    fn embodiment(&self) -> &str {
        // The stable name this robot type is selected by everywhere else.
        "warehouse_pick_arm"
    }

    fn action_space(&self) -> ActionSpace {
        // This arm is position-controlled: each command targets joint positions.
        ActionSpace::JointPosition
    }

    fn attribution_window(&self, kind: OutcomeKind) -> Option<AttributionWindow> {
        // Each outcome kind gets the time window in which it may bind back to the
        // rollout that likely caused it.
        match kind {
            OutcomeKind::Collision => Some(AttributionWindow {
                window_ms: 150,
                requires_monotonic_colocation: true,
            }),
            OutcomeKind::EStop => Some(AttributionWindow {
                window_ms: 500,
                requires_monotonic_colocation: true,
            }),
            OutcomeKind::TeleopTakeover => Some(AttributionWindow {
                window_ms: 1_500,
                requires_monotonic_colocation: false,
            }),
            OutcomeKind::DownstreamFailure => Some(AttributionWindow {
                window_ms: 5_000,
                requires_monotonic_colocation: false,
            }),
            // Heartbeats prove coverage rather than binding to a rollout.
            OutcomeKind::Heartbeat => None,
        }
    }

    fn requires_monotonic_colocation(&self, kind: OutcomeKind) -> bool {
        // Tight-timing safety events only bind when rollout and outcome share a boot
        // and can be compared on the monotonic clock.
        matches!(kind, OutcomeKind::Collision | OutcomeKind::EStop)
    }

    fn classify(&self, outcome: &OutcomeEvent) -> FailureClass {
        // This is a triage hint, not the final human-reviewed label.
        match outcome.outcome_kind {
            OutcomeKind::Collision => FailureClass::Manipulation,
            OutcomeKind::EStop => FailureClass::Hardware,
            OutcomeKind::TeleopTakeover => FailureClass::Operator,
            OutcomeKind::DownstreamFailure => FailureClass::Environment,
            OutcomeKind::Heartbeat => FailureClass::Environment,
        }
    }

    fn detect_success(&self, signal: f64) -> SuccessVerdict {
        // The caller computes `signal` from robot telemetry before calling us.
        // In this example:
        //   signal = normalized_goal_distance + gripper_not_holding_penalty
        // so the good range is near 0.0 and the bad range is near 1.0.
        if signal.is_nan() || !(0.0..=1.0).contains(&signal) {
            SuccessVerdict::Unknown
        } else if signal <= 0.15 {
            SuccessVerdict::Success
        } else if signal >= 0.85 {
            SuccessVerdict::Failure
        } else {
            SuccessVerdict::Unknown
        }
    }

    fn normalize_action(&self, raw: &RawAction) -> NormalizedAction {
        // The adapter is where robot-local channel names and units become the
        // canonical action vector the rest of Fieldloop understands.
        let channels = raw
            .channels
            .iter()
            .map(|(name, value)| (name.clone(), value.clamp(-1.0, 1.0)))
            .collect();
        NormalizedAction {
            action_space: self.action_space(),
            channels,
        }
    }

    fn lerobot_features(&self) -> LeRobotFeatures {
        // This is the feature schema the downstream training export materializes.
        LeRobotFeatures::from_pairs([
            (
                "observation.joint_position",
                FeatureSpec::new("float32", vec![7]),
            ),
            ("observation.gripper_closed", FeatureSpec::scalar("bool")),
            ("action", FeatureSpec::new("float32", vec![7])),
            ("next.success", FeatureSpec::scalar("bool")),
        ])
    }
}

fn main() -> ExitCode {
    let adapter = WarehousePickArmAdapter;
    let report = check_conformance(&adapter);
    print_report(&report, adapter.embodiment());

    if report.is_conformant() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn print_report(report: &ConformanceReport, embodiment: &str) {
    match report {
        ConformanceReport::Conformant => {
            println!("adapter `{embodiment}` is conformant");
        }
        ConformanceReport::NonConformant(failures) => {
            println!(
                "adapter `{embodiment}` is NOT conformant ({} failures):",
                failures.len()
            );
            for failure in failures {
                println!(
                    "- [{:?}] {}: {}",
                    failure.severity, failure.check, failure.detail
                );
            }
        }
    }
}
