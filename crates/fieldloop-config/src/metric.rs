//! Outcome / metric definitions: what a customer wants scored, and how.
//!
//! A metric declares a name, the shape of its value (boolean, float, or a
//! categorical failure class), and the grain it is measured at (one inference, or a
//! whole episode). Only a float score has a direction worth optimizing, so the
//! optimization direction is carried *inside* the float arm of the value enum —
//! making "a boolean metric with an optimize direction" unrepresentable rather than
//! merely invalid.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// The grain a metric is measured at, as a closed set.
///
/// A closed enum so read-side code can match exhaustively on grain, and a new grain
/// forces every consumer to decide how to handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricLevel {
    /// Scored per single inference / rollout step.
    Inference,
    /// Scored per whole episode / trajectory.
    Episode,
}

/// For a float metric, which direction is "better".
///
/// A closed enum so the optimizer cannot be handed a free-text direction it does
/// not understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimize {
    /// Higher is better (for example a reward).
    Max,
    /// Lower is better (for example a distance-to-goal).
    Min,
}

/// The bare value-shape tag of a metric, as written under the TOML `kind` key.
///
/// A closed enum so a typo'd kind fails at load. This is only the discriminant; the
/// validated [`MetricKind`] pairs it with the optimization direction where that is
/// meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKindTag {
    /// A true/false outcome (for example success, grasped).
    Boolean,
    /// A real-valued score.
    Float,
    /// A categorical failure class drawn from the shared failure taxonomy.
    FailureClass,
}

/// The validated shape of a metric's value.
///
/// The optimization direction lives on the `Float` arm and nowhere else, so a
/// validated boolean or failure-class metric simply has no place to put one — the
/// illegal "non-float with an optimize direction" state cannot be represented once a
/// metric is loaded. (Deserialization accepts a looser shape so it can produce a
/// clear, named error; this is the tightened form it is checked into.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// A true/false outcome (for example success, grasped).
    Boolean,
    /// A real-valued score; it carries which direction is better.
    Float {
        /// Whether higher or lower values are the goal.
        optimize: Optimize,
    },
    /// A categorical failure class drawn from the shared failure taxonomy.
    FailureClass,
}

/// The raw, unvalidated form of one metric as written in TOML.
///
/// `optimize` is an [`Option`] here rather than being bound to a float-only arm,
/// because the goal is a *named* error: a flattened tagged enum would let serde
/// silently swallow a stray `optimize` on a boolean metric, so the consistency
/// between `kind` and `optimize` is checked explicitly in [`Self::load`] instead.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedMetricConfig {
    /// The grain this metric is measured at.
    pub level: MetricLevel,
    /// The bare value-shape tag.
    pub kind: MetricKindTag,
    /// The optimization direction. Required for a float, forbidden otherwise.
    #[serde(default)]
    pub optimize: Option<Optimize>,
}

/// A validated metric definition, carrying its own name.
#[derive(Debug, Clone)]
pub struct MetricConfig {
    /// The declared name of this metric (the lookup key in the parent config).
    pub name: String,
    /// The grain this metric is measured at.
    pub level: MetricLevel,
    /// The value shape and, for floats, the optimization direction.
    pub kind: MetricKind,
}

/// The reserved namespace the runtime uses for its own built-in metrics; user
/// config may not declare names in it.
pub(crate) const RESERVED_PREFIX: &str = "fieldloop";

impl UninitializedMetricConfig {
    /// Check the name and the kind/optimize consistency, producing the tightened
    /// [`MetricConfig`].
    ///
    /// A float must declare a direction (there is nothing to optimize toward without
    /// one) and a non-float must not (a boolean or categorical value has no
    /// direction, so an `optimize` there is a mistake that would otherwise do
    /// nothing). The name must also stay out of the runtime's reserved namespace.
    pub(crate) fn load(self, name: String) -> Result<MetricConfig, ConfigError> {
        if name == RESERVED_PREFIX || name.starts_with(&format!("{RESERVED_PREFIX}_")) {
            return Err(ConfigError::ReservedMetricName { name });
        }

        let kind = match (self.kind, self.optimize) {
            (MetricKindTag::Float, Some(optimize)) => MetricKind::Float { optimize },
            (MetricKindTag::Float, None) => {
                return Err(ConfigError::OptimizeMismatch {
                    name,
                    reason: "a float metric must declare an `optimize` direction",
                });
            }
            (MetricKindTag::Boolean, None) => MetricKind::Boolean,
            (MetricKindTag::FailureClass, None) => MetricKind::FailureClass,
            (MetricKindTag::Boolean | MetricKindTag::FailureClass, Some(_)) => {
                return Err(ConfigError::OptimizeMismatch {
                    name,
                    reason: "only a float metric may declare an `optimize` direction",
                });
            }
        };

        Ok(MetricConfig {
            name,
            level: self.level,
            kind,
        })
    }
}
