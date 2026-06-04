//! The single error type returned when configuration fails to load.
//!
//! Every variant names the offending item (an embodiment name, a metric name, a
//! TOML key path) so a human reading the message can jump straight to the line in
//! their config that is wrong. Configuration is validated once, at load time, so
//! these errors surface at startup rather than as a surprise mid-request after a
//! robot is already deployed.

use thiserror::Error;

/// What went wrong while turning raw TOML into a validated [`crate::Config`].
///
/// A closed enum rather than a string so callers can match on the specific failure
/// (tests assert the exact variant) and so adding a new failure mode forces every
/// consumer to acknowledge it.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The TOML did not parse or did not match the expected shape (a typo'd key, a
    /// wrong type, an unknown field). The message carries the key path that serde
    /// rejected, so the operator can find the exact offending line.
    #[error("config does not parse or match the schema at `{path}`: {message}")]
    Deserialize {
        /// The dotted TOML key path that failed, e.g. `embodiments.ur5e.action_space`.
        path: String,
        /// The underlying serde message explaining why that key was rejected.
        message: String,
    },

    /// Two embodiments (or two attribution keys / two metrics) share a name. Names
    /// are the lookup key the runtime joins on, so a duplicate would make a lookup
    /// ambiguous — it is rejected up front rather than silently letting one shadow
    /// the other.
    #[error("duplicate {scope} name `{name}` — names must be unique")]
    DuplicateName {
        /// Which table the duplicate is in (`embodiment` or `metric`).
        scope: &'static str,
        /// The name that appears more than once.
        name: String,
    },

    /// An attribution table used a key that is not a real outcome kind. The
    /// attribution table is keyed by outcome kind, and a key the runtime cannot map
    /// to a known kind would never be consulted — so an unrecognized key is a typo
    /// worth failing on rather than ignoring.
    #[error("embodiment `{embodiment}` has an attribution entry for unknown outcome kind `{kind}`")]
    UnknownOutcomeKind {
        /// The embodiment whose attribution table holds the bad key.
        embodiment: String,
        /// The unrecognized key as written in the config.
        kind: String,
    },

    /// The tight-window collision kind was declared without requiring monotonic
    /// co-location. A collision can only be attributed when the rollout and the
    /// outcome share a boot and can be compared on the skew-free monotonic clock;
    /// allowing it to bind on a drifting wall-clock estimate would manufacture false
    /// collision attributions, so the stricter flag is required for that kind.
    #[error(
        "embodiment `{embodiment}` declares a `collision` attribution window without \
         `requires_monotonic_colocation = true` — collision attribution must be clock-safe"
    )]
    CollisionNotColocated {
        /// The embodiment with the unsafe collision window.
        embodiment: String,
    },

    /// A metric name is reserved or uses the reserved prefix. The runtime mints its
    /// own internal metric names under a `fieldloop` prefix; letting user config
    /// reuse that namespace would let a user definition collide with or impersonate
    /// a built-in one.
    #[error("metric name `{name}` is reserved — names must not use the `fieldloop` prefix")]
    ReservedMetricName {
        /// The offending metric name.
        name: String,
    },

    /// A float metric was declared without an optimization direction, or a
    /// non-float metric carried one. Only a float score has a direction to optimize
    /// (maximize reward, minimize distance); a boolean or categorical metric has no
    /// meaningful direction, so attaching one would be a config mistake that silently
    /// does nothing.
    #[error("metric `{name}`: {reason}")]
    OptimizeMismatch {
        /// The metric whose optimize field is inconsistent with its kind.
        name: String,
        /// A plain-English description of the mismatch.
        reason: &'static str,
    },

    /// The calibration version was empty. Confidence numbers are traced back to the
    /// calibration version that produced them; an empty version would make that
    /// provenance untraceable, so a non-empty version is required.
    #[error("calibration version must be a non-empty string")]
    EmptyCalibrationVersion,

    /// A customer label definition is internally inconsistent (e.g. a zero attribution
    /// window). The illegal *bindings* (an outcome kind or failure class outside the
    /// closed taxonomy) are already rejected by deserialization; this covers the
    /// remaining per-label invariants so a label can never be silently malformed.
    #[error("label `{name}`: {reason}")]
    InvalidLabel {
        /// The offending label name.
        name: String,
        /// A plain-English description of what is wrong.
        reason: &'static str,
    },
}
