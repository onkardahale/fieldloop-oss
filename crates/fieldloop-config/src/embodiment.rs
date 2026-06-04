//! Embodiment adapters: one declared robot type and how its outcomes are attributed.
//!
//! Each robot type a customer runs is its own embodiment with a fixed action space
//! and an attribution table. The attribution table answers, per outcome kind, "how
//! far back in time may we look to bind this outcome to a rollout, and may we only
//! do so when the two share a boot on the monotonic clock?". Holding this in config
//! (not code literals) lets a heterogeneous fleet of different robot types each tune
//! their own windows without a code change.

use std::collections::HashMap;

use fieldloop_types::OutcomeKind;
use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// The control interface a robot type exposes, as a closed set.
///
/// A closed enum rather than a free string so dispatch code can match exhaustively
/// and a typo in the config is rejected at load instead of producing an
/// unrecognized action space at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ActionSpace {
    /// Commands target absolute joint angles.
    JointPosition,
    /// Commands target joint angular velocities.
    JointVelocity,
    /// Commands target an end-effector pose in Cartesian space.
    EndEffectorPose,
    /// Commands pick from a finite, discrete set of actions.
    Discrete,
}

/// How far back, and how strictly, one outcome kind may be attributed to a rollout.
///
/// `window_ms` bounds the temporal search; `requires_monotonic_colocation` forces
/// the rollout and outcome to share a boot and be comparable on the skew-free
/// monotonic clock before they may bind. The strict flag exists because robot
/// wall-clocks drift: for tight-timing kinds, a match on a drifting wall estimate
/// would be a false attribution, so those kinds demand a clock-safe comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributionWindow {
    /// The maximum outcome-arrival delay, in milliseconds, still eligible to bind.
    pub window_ms: u64,
    /// When true, the rollout and outcome must share a boot and be compared on the
    /// monotonic clock; a wall-clock-only match is rejected.
    pub requires_monotonic_colocation: bool,
}

/// The raw, unvalidated form of one embodiment as written in TOML.
///
/// The attribution table is keyed by the *string* an operator typed, so an
/// unrecognized outcome kind can be reported with the bad key intact before it is
/// parsed into the typed [`OutcomeKind`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedEmbodimentConfig {
    /// The control interface this robot type exposes.
    pub action_space: ActionSpace,
    /// Per-outcome-kind attribution windows, keyed by the outcome kind's config
    /// name (for example `collision`, `teleop_takeover`).
    #[serde(default)]
    pub attribution: HashMap<String, AttributionWindow>,
}

/// A validated embodiment: its attribution table is keyed by the typed
/// [`OutcomeKind`], so every key is guaranteed to be a real kind and lookups are a
/// total enum match rather than a string compare.
#[derive(Debug, Clone)]
pub struct EmbodimentConfig {
    /// The declared name of this robot type (the lookup key in the parent config).
    pub name: String,
    /// The control interface this robot type exposes.
    pub action_space: ActionSpace,
    /// Per-kind attribution windows, now keyed by the typed outcome kind.
    pub attribution: HashMap<OutcomeKind, AttributionWindow>,
}

impl UninitializedEmbodimentConfig {
    /// Parse each attribution key into a real [`OutcomeKind`] and enforce the
    /// clock-safety rule, producing the validated [`EmbodimentConfig`].
    ///
    /// `name` is threaded in (rather than stored on the raw struct) because in TOML
    /// the name is the table key, not a field, and the validated form needs it both
    /// as its identity and to name itself in any error message.
    pub(crate) fn load(self, name: String) -> Result<EmbodimentConfig, ConfigError> {
        let mut attribution = HashMap::with_capacity(self.attribution.len());
        for (raw_kind, window) in self.attribution {
            let kind =
                parse_outcome_kind(&raw_kind).ok_or_else(|| ConfigError::UnknownOutcomeKind {
                    embodiment: name.clone(),
                    kind: raw_kind.clone(),
                })?;

            // The collision kind is tight-timing: it must compare on the monotonic
            // clock or not bind at all, so a non-colocated collision window is a
            // configuration mistake rather than a permitted looser policy.
            if kind == OutcomeKind::Collision && !window.requires_monotonic_colocation {
                return Err(ConfigError::CollisionNotColocated {
                    embodiment: name.clone(),
                });
            }

            attribution.insert(kind, window);
        }

        Ok(EmbodimentConfig {
            name,
            action_space: self.action_space,
            attribution,
        })
    }
}

/// Map an attribution key string to its [`OutcomeKind`], matching the snake_case
/// names the schema crate serializes to.
///
/// Done with an explicit match (not a serde round-trip) so the accepted spellings
/// are visible in one place and stay in lock-step with the canonical kinds even if
/// the schema's serialization details change.
fn parse_outcome_kind(s: &str) -> Option<OutcomeKind> {
    match s {
        "teleop_takeover" => Some(OutcomeKind::TeleopTakeover),
        "e_stop" => Some(OutcomeKind::EStop),
        "collision" => Some(OutcomeKind::Collision),
        "downstream_failure" => Some(OutcomeKind::DownstreamFailure),
        "heartbeat" => Some(OutcomeKind::Heartbeat),
        _ => None,
    }
}
