//! # `fieldloop-config` — typed, load-time-validated configuration
//!
//! Fieldloop runs a heterogeneous fleet of robot types whose outcomes must be bound
//! back to the rollouts that caused them. That binding is tuned per robot type and
//! per outcome kind; this crate is where those tunings, the metrics a customer wants
//! scored, and the calibration dials are declared in TOML and turned into a typed,
//! validated [`Config`].
//!
//! ## Validate once, at load, with a located error
//! A bad config should fail at startup with a message naming the offending item, not
//! as a surprise after a robot is already deployed and shipping outcomes. So loading
//! is a two-phase move:
//!
//! 1. [`UninitializedConfig`] is the raw deserialized form — string keys, unresolved
//!    cross-references, unknown fields rejected outright.
//! 2. [`Config`] is the validated form — every attribution key resolved to a real
//!    outcome kind, every name unique, every invariant checked. By the time runtime
//!    code holds a [`Config`], the embodiment, metric, or window it looks up is
//!    guaranteed to exist and be well-formed.
//!
//! Holding the two apart means the validated type can make the runtime's life easy
//! (typed map lookups, exhaustive matches) while all the fallible work happens once,
//! in [`UninitializedConfig::load`].

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::collections::HashMap;

use serde::Deserialize;

mod calibration;
mod embodiment;
mod error;
mod labels;
mod metric;

pub use calibration::{CalibrationConfig, UninitializedCalibrationConfig};
pub use embodiment::{
    ActionSpace, AttributionWindow, EmbodimentConfig, UninitializedEmbodimentConfig,
};
pub use error::ConfigError;
pub use labels::{LabelConfig, Severity, UninitializedLabelConfig};
pub use metric::{MetricConfig, MetricKind, MetricLevel, Optimize, UninitializedMetricConfig};

use fieldloop_types::{FailureClass, OutcomeKind};

/// The raw, deserialized configuration, before any cross-reference is resolved or
/// invariant checked.
///
/// `deny_unknown_fields` so a typo'd top-level section is a hard error rather than
/// silently ignored config. The embodiment and metric sections default to empty
/// maps, so a minimal config need not declare every section.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedConfig {
    /// Declared robot types, keyed by name.
    #[serde(default)]
    pub embodiments: HashMap<String, UninitializedEmbodimentConfig>,
    /// Declared metrics, keyed by name.
    #[serde(default)]
    pub metrics: HashMap<String, UninitializedMetricConfig>,
    /// Customer-defined failure labels mapped onto the controlled taxonomy, keyed by
    /// label name (e.g. `[labels.failed_grasp]`). Defaults to empty.
    #[serde(default)]
    pub labels: HashMap<String, UninitializedLabelConfig>,
    /// The calibration knob set.
    pub calibration: UninitializedCalibrationConfig,
}

/// The validated, ready-to-serve configuration.
///
/// Name-keyed maps whose values are fully resolved: an embodiment's attribution
/// table is keyed by the typed [`OutcomeKind`], a metric's value shape and optimize
/// direction are consistent by construction, and the calibration version is
/// guaranteed non-empty.
#[derive(Debug, Clone)]
pub struct Config {
    /// Declared robot types, keyed by name, each with a resolved attribution table.
    pub embodiments: HashMap<String, EmbodimentConfig>,
    /// Declared metrics, keyed by name.
    pub metrics: HashMap<String, MetricConfig>,
    /// Customer-defined failure labels, keyed by canonicalized name. Each binds the
    /// custom name to a real outcome kind + failure class in the closed taxonomy.
    pub labels: HashMap<String, LabelConfig>,
    /// The validated calibration knob set.
    pub calibration: CalibrationConfig,
}

impl UninitializedConfig {
    /// Resolve cross-references and check every invariant, producing a [`Config`].
    ///
    /// Note that some invariants are already enforced by deserialization (unknown
    /// fields, a misspelled kind or action space); this method covers the rest:
    /// unique names, attribution keys that resolve to real outcome kinds, the
    /// collision clock-safety rule, the float/optimize consistency, reserved metric
    /// names, and a non-empty calibration version. Names are canonicalized to lowercase for lookup, and two
    /// names that collide once canonicalized are rejected so a case-folding lookup is
    /// never ambiguous. (A literally repeated TOML table key is caught even earlier,
    /// by the TOML parser, and surfaces as a deserialize error.)
    pub fn load(self) -> Result<Config, ConfigError> {
        // Names are looked up case-insensitively at runtime, so they are
        // canonicalized to lowercase here and a collision after canonicalization is
        // rejected — otherwise `ur5e` and `UR5e` would be two entries that a
        // case-folding lookup could not tell apart.
        let mut embodiments = HashMap::with_capacity(self.embodiments.len());
        for (name, raw) in self.embodiments {
            let key = name.to_ascii_lowercase();
            if embodiments.contains_key(&key) {
                return Err(ConfigError::DuplicateName {
                    scope: "embodiment",
                    name: key,
                });
            }
            let loaded = raw.load(name)?;
            embodiments.insert(key, loaded);
        }

        let mut metrics = HashMap::with_capacity(self.metrics.len());
        for (name, raw) in self.metrics {
            let key = name.to_ascii_lowercase();
            if metrics.contains_key(&key) {
                return Err(ConfigError::DuplicateName {
                    scope: "metric",
                    name: key,
                });
            }
            let loaded = raw.load(name)?;
            metrics.insert(key, loaded);
        }

        // Customer labels: canonicalize the name, reject a post-canonicalization
        // collision (so a case-folding lookup is unambiguous), and validate each.
        let mut labels = HashMap::with_capacity(self.labels.len());
        for (name, raw) in self.labels {
            let key = name.to_ascii_lowercase();
            if labels.contains_key(&key) {
                return Err(ConfigError::DuplicateName {
                    scope: "label",
                    name: key,
                });
            }
            let loaded = raw.load(key.clone())?;
            labels.insert(key, loaded);
        }

        let calibration = self.calibration.load()?;

        Ok(Config {
            embodiments,
            metrics,
            labels,
            calibration,
        })
    }
}

impl Config {
    /// Parse a TOML string into a validated [`Config`].
    ///
    /// Deserialization errors are wrapped with the exact key path that failed (via a
    /// path-tracking deserializer) so a malformed config points the operator at the
    /// offending line rather than reporting a generic "invalid data".
    pub fn from_toml_str(input: &str) -> Result<Config, ConfigError> {
        let de = toml::Deserializer::new(input);
        let raw: UninitializedConfig =
            serde_path_to_error::deserialize(de).map_err(|e| ConfigError::Deserialize {
                path: e.path().to_string(),
                message: e.inner().to_string(),
            })?;
        raw.load()
    }

    /// The attribution window for one outcome kind on one embodiment, or `None` if
    /// the embodiment is unknown or has no window declared for that kind.
    ///
    /// This is the lookup later stages of the pipeline call to decide how far back to
    /// search when binding an outcome to a rollout. The embodiment name is matched
    /// case-insensitively (names are canonicalized at load). It returns `None` rather
    /// than a default so a caller must consciously decide what to do for an
    /// unconfigured pair instead of silently inheriting a window that was never
    /// declared for it.
    #[must_use]
    pub fn attribution_window(
        &self,
        embodiment: &str,
        kind: OutcomeKind,
    ) -> Option<AttributionWindow> {
        self.embodiments
            .get(&embodiment.to_ascii_lowercase())
            .and_then(|e| e.attribution.get(&kind).copied())
    }

    /// The customer label definition for a name, or `None` if no such label is declared.
    /// Matched case-insensitively (names are canonicalized at load), so a metric/label
    /// string seen at runtime resolves to its taxonomy binding without case fuss.
    #[must_use]
    pub fn label(&self, name: &str) -> Option<&LabelConfig> {
        self.labels.get(&name.to_ascii_lowercase())
    }

    /// Roll a runtime label/metric string up into the controlled taxonomy: the
    /// [`FailureClass`] a custom label maps to, or `None` if the string is not a
    /// declared label. This is the bridge that lets a custom `failed_grasp` be counted
    /// both as itself and within `manipulation` — without the closed class ever being
    /// extended by a free string.
    #[must_use]
    pub fn rollup_class(&self, label_or_metric: &str) -> Option<FailureClass> {
        self.label(label_or_metric).map(|l| l.failure_class)
    }

    /// A small, self-contained example configuration used by tests and as a
    /// runnable reference for the TOML shape this crate accepts.
    #[must_use]
    pub fn example() -> Config {
        Config::from_toml_str(EXAMPLE_TOML).expect("the embedded example config must be valid")
    }
}

/// A minimal but representative configuration: one arm embodiment with a clock-safe
/// collision window and a takeover window, two metrics (a boolean and a float that
/// declares its optimization direction), and a calibration section.
pub const EXAMPLE_TOML: &str = r#"
[embodiments.ur5e]
action_space = "joint_position"

[embodiments.ur5e.attribution.collision]
window_ms = 250
requires_monotonic_colocation = true

[embodiments.ur5e.attribution.teleop_takeover]
window_ms = 5000
requires_monotonic_colocation = false

[metrics.task_success]
kind = "boolean"
level = "episode"

[metrics.episode_return]
kind = "float"
optimize = "max"
level = "episode"

[labels.failed_grasp]
outcome_kind = "downstream_failure"
failure_class = "manipulation"
task = "pick_can"
severity = "standard"
attribution_window_ms = 2000

[labels.bad_dock_alignment]
outcome_kind = "downstream_failure"
failure_class = "planning"
severity = "critical"

[calibration]
version = "calib-2026-01"
temporal_window_default_ms = 2000
heartbeat_coverage_k = 0.9
"#;

/// A copyable sample config file shipped under `examples/`, kept parse-tested here
/// so the onboarding docs do not silently rot away from the real loader.
pub const EMBODIMENT_SAMPLE_TOML: &str = include_str!("../examples/embodiment.sample.toml");

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid config loads and the attribution accessor returns the exact window
    /// declared for the tight-timing collision kind — the contract later stages
    /// depend on.
    #[test]
    fn valid_config_loads_and_collision_window_resolves() {
        let cfg = Config::from_toml_str(EXAMPLE_TOML).expect("example config must load");

        let window = cfg
            .attribution_window("ur5e", OutcomeKind::Collision)
            .expect("ur5e must have a collision window");
        assert_eq!(window.window_ms, 250);
        assert!(window.requires_monotonic_colocation);

        // A takeover window is also present and is allowed to be wall-clock-loose.
        let takeover = cfg
            .attribution_window("ur5e", OutcomeKind::TeleopTakeover)
            .expect("ur5e must have a takeover window");
        assert_eq!(takeover.window_ms, 5000);
        assert!(!takeover.requires_monotonic_colocation);

        // An undeclared (embodiment, kind) pair returns None rather than a default,
        // and an unknown embodiment also returns None.
        assert!(cfg.attribution_window("ur5e", OutcomeKind::EStop).is_none());
        assert!(
            cfg.attribution_window("does-not-exist", OutcomeKind::Collision)
                .is_none()
        );
    }

    /// An unknown top-level field is rejected at deserialize time, and the error
    /// carries the key path so the operator can find it.
    #[test]
    fn unknown_field_is_rejected() {
        let toml = r#"
[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
bogus_field = 7
"#;
        let err = Config::from_toml_str(toml).expect_err("unknown field must fail");
        match err {
            ConfigError::Deserialize { message, .. } => {
                assert!(
                    message.contains("bogus_field") || message.contains("unknown field"),
                    "message should mention the offending key: {message}"
                );
            }
            other => panic!("expected a deserialize error, got {other:?}"),
        }
    }

    /// An attribution table keyed by a string that is not a real outcome kind fails,
    /// and the error names both the embodiment and the bad key.
    #[test]
    fn bogus_outcome_kind_key_is_rejected() {
        let toml = r#"
[embodiments.ur5e]
action_space = "joint_position"

[embodiments.ur5e.attribution.not_a_real_kind]
window_ms = 100
requires_monotonic_colocation = true

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("bogus outcome kind must fail");
        match err {
            ConfigError::UnknownOutcomeKind { embodiment, kind } => {
                assert_eq!(embodiment, "ur5e");
                assert_eq!(kind, "not_a_real_kind");
            }
            other => panic!("expected UnknownOutcomeKind, got {other:?}"),
        }
    }

    /// A collision window without monotonic co-location is rejected, because a
    /// collision must only bind on the skew-free monotonic clock.
    #[test]
    fn collision_without_colocation_is_rejected() {
        let toml = r#"
[embodiments.ur5e]
action_space = "joint_position"

[embodiments.ur5e.attribution.collision]
window_ms = 100
requires_monotonic_colocation = false

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("unsafe collision window must fail");
        assert!(matches!(
            err,
            ConfigError::CollisionNotColocated { embodiment } if embodiment == "ur5e"
        ));
    }

    /// A float metric declared without its optimization direction fails, and the
    /// error names the offending metric — there is nothing to optimize toward
    /// without a direction.
    #[test]
    fn float_metric_missing_optimize_is_rejected() {
        let toml = r#"
[metrics.episode_return]
kind = "float"
level = "episode"

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("float without optimize must fail");
        assert!(matches!(
            err,
            ConfigError::OptimizeMismatch { ref name, .. } if name == "episode_return"
        ));
    }

    /// A non-float metric carrying an optimize direction is rejected, because only a
    /// float score has a direction worth optimizing.
    #[test]
    fn boolean_metric_with_optimize_is_rejected() {
        let toml = r#"
[metrics.task_success]
kind = "boolean"
level = "episode"
optimize = "max"

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("boolean with optimize must fail");
        assert!(matches!(
            err,
            ConfigError::OptimizeMismatch { ref name, .. } if name == "task_success"
        ));
    }

    /// Two metric names that collide once canonicalized to lowercase are rejected,
    /// so a case-insensitive lookup is never ambiguous.
    #[test]
    fn duplicate_metric_name_is_rejected() {
        let toml = r#"
[metrics.task_success]
kind = "boolean"
level = "episode"

[metrics.Task_Success]
kind = "boolean"
level = "inference"

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("duplicate metric name must fail");
        assert!(matches!(
            err,
            ConfigError::DuplicateName { scope: "metric", name } if name == "task_success"
        ));
    }

    /// A metric using the reserved namespace is rejected so user config cannot
    /// impersonate a built-in metric.
    #[test]
    fn reserved_metric_name_is_rejected() {
        let toml = r#"
[metrics.fieldloop_internal]
kind = "boolean"
level = "inference"

[calibration]
version = "v1"
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("reserved name must fail");
        assert!(matches!(
            err,
            ConfigError::ReservedMetricName { name } if name == "fieldloop_internal"
        ));
    }

    /// An empty calibration version fails, because confidence numbers are traced
    /// back to the version that produced them.
    #[test]
    fn empty_calibration_version_is_rejected() {
        let toml = r#"
[calibration]
version = ""
temporal_window_default_ms = 1000
heartbeat_coverage_k = 0.5
"#;
        let err = Config::from_toml_str(toml).expect_err("empty version must fail");
        assert!(matches!(err, ConfigError::EmptyCalibrationVersion));
    }

    /// The embedded example is a valid config and exposes the metrics it declares
    /// with the right value shapes.
    #[test]
    fn example_exposes_typed_metrics() {
        let cfg = Config::example();
        assert_eq!(cfg.calibration.version, "calib-2026-01");

        let success = cfg
            .metrics
            .get("task_success")
            .expect("task_success metric");
        assert_eq!(success.kind, MetricKind::Boolean);
        assert_eq!(success.level, MetricLevel::Episode);

        let ret = cfg
            .metrics
            .get("episode_return")
            .expect("episode_return metric");
        assert_eq!(
            ret.kind,
            MetricKind::Float {
                optimize: Optimize::Max
            }
        );
    }

    /// The copyable embodiment sample stays in lock-step with the real loader, so a
    /// newcomer can paste it into place without discovering later that the example
    /// drifted out of date.
    #[test]
    fn embodiment_sample_toml_loads() {
        let cfg = Config::from_toml_str(EMBODIMENT_SAMPLE_TOML)
            .expect("the sample embodiment config must stay valid");

        let collision = cfg
            .attribution_window("warehouse_pick_arm", OutcomeKind::Collision)
            .expect("sample must declare a collision window");
        assert_eq!(collision.window_ms, 150);
        assert!(collision.requires_monotonic_colocation);

        let downstream = cfg
            .attribution_window("warehouse_pick_arm", OutcomeKind::DownstreamFailure)
            .expect("sample must declare a downstream-failure window");
        assert_eq!(downstream.window_ms, 5000);
        assert!(!downstream.requires_monotonic_colocation);
    }

    /// A calibration section every label test can append to (Config requires one).
    const CALIB: &str = "\n[calibration]\nversion = \"v1\"\ntemporal_window_default_ms = 1000\nheartbeat_coverage_k = 0.5\n";

    /// A customer label loads, carries its full binding, and rolls up into the closed
    /// failure class — queryable as itself AND within the controlled taxonomy.
    #[test]
    fn labels_load_and_roll_up_into_the_taxonomy() {
        let cfg = Config::from_toml_str(EXAMPLE_TOML).expect("example config must load");

        let l = cfg
            .label("failed_grasp")
            .expect("failed_grasp must be declared");
        assert_eq!(l.outcome_kind, OutcomeKind::DownstreamFailure);
        assert_eq!(l.failure_class, FailureClass::Manipulation);
        assert_eq!(l.task.as_deref(), Some("pick_can"));
        assert_eq!(l.severity, Severity::Standard);
        assert_eq!(l.attribution_window_ms, Some(2000));

        // The bridge: a custom label rolls up into a real failure class.
        assert_eq!(
            cfg.rollup_class("failed_grasp"),
            Some(FailureClass::Manipulation)
        );
        // Critical severity + an absent window default through.
        let d = cfg.label("bad_dock_alignment").expect("declared");
        assert_eq!(d.severity, Severity::Critical);
        assert_eq!(d.attribution_window_ms, None);
        // Case-insensitive; an undeclared name resolves to nothing (no invented class).
        assert!(cfg.label("FAILED_GRASP").is_some());
        assert!(cfg.label("never_defined").is_none());
        assert!(cfg.rollup_class("never_defined").is_none());
    }

    /// Binding a label to a failure class OUTSIDE the closed taxonomy is unrepresentable:
    /// it fails to deserialize. The controlled vocabulary can never be polluted.
    #[test]
    fn label_class_outside_taxonomy_is_rejected() {
        let toml = format!(
            "[labels.weird]\noutcome_kind = \"downstream_failure\"\nfailure_class = \"grasping\"\n{CALIB}"
        );
        let err = Config::from_toml_str(&toml).expect_err("an unknown failure class must fail");
        assert!(
            matches!(err, ConfigError::Deserialize { .. }),
            "got {err:?}"
        );
    }

    /// Likewise an outcome kind outside the taxonomy is rejected at load.
    #[test]
    fn label_outcome_kind_outside_taxonomy_is_rejected() {
        let toml = format!(
            "[labels.weird]\noutcome_kind = \"exploded\"\nfailure_class = \"hardware\"\n{CALIB}"
        );
        let err = Config::from_toml_str(&toml).expect_err("an unknown outcome kind must fail");
        assert!(
            matches!(err, ConfigError::Deserialize { .. }),
            "got {err:?}"
        );
    }

    /// Two label names that collide once canonicalized are rejected so a case-folding
    /// lookup is never ambiguous.
    #[test]
    fn duplicate_label_name_is_rejected() {
        let toml = format!(
            "[labels.Foo]\noutcome_kind = \"e_stop\"\nfailure_class = \"hardware\"\n\
             [labels.foo]\noutcome_kind = \"e_stop\"\nfailure_class = \"operator\"\n{CALIB}"
        );
        let err =
            Config::from_toml_str(&toml).expect_err("duplicate canonical label name must fail");
        assert!(
            matches!(err, ConfigError::DuplicateName { scope: "label", .. }),
            "got {err:?}"
        );
    }

    /// A zero attribution window is a load-time error (a window of zero would bind
    /// nothing — a config mistake, not a valid setting).
    #[test]
    fn zero_label_window_is_rejected() {
        let toml = format!(
            "[labels.x]\noutcome_kind = \"collision\"\nfailure_class = \"hardware\"\n\
             attribution_window_ms = 0\n{CALIB}"
        );
        let err = Config::from_toml_str(&toml).expect_err("a zero window must fail");
        assert!(
            matches!(err, ConfigError::InvalidLabel { .. }),
            "got {err:?}"
        );
    }
}
