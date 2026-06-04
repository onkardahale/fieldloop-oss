//! Customer-defined failure labels mapped onto the controlled taxonomy.
//!
//! A robotics team's failures are domain-specific — `failed_grasp`, `missed_bin`,
//! `barcode_scan_failed`, `bad_dock_alignment` — but cross-fleet analytics needs a
//! controlled vocabulary or it degrades into unqueryable free strings. This module is
//! the bridge: a customer declares a label by name and BINDS it to a real
//! [`OutcomeKind`] and [`FailureClass`] from the closed taxonomy (plus an optional task,
//! a triage severity, and a per-label attribution-window override). The custom label is
//! then both queryable as itself AND rolls up into the shared classes.
//!
//! The closed taxonomy is the feature, not a limitation: binding a label to an
//! `outcome_kind` or `failure_class` outside the taxonomy fails to deserialize, so an
//! illegal binding is unrepresentable — the controlled vocabulary can never be
//! polluted by an invented class. What is open is the label *name* and its *mapping*.

use serde::{Deserialize, Serialize};

use fieldloop_types::{FailureClass, OutcomeKind};

use crate::error::ConfigError;

/// How serious a failure mode is, for triage. Closed on purpose: a custom severity would
/// defeat the point of a shared triage vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// The default: a failure worth logging and curating, but not an escalation.
    #[default]
    Standard,
    /// A safety- or mission-critical failure that should be escalated.
    Critical,
}

/// The raw, deserialized label definition (before name + invariant resolution).
///
/// `deny_unknown_fields` so a mistyped key is a hard error, and the `outcome_kind` /
/// `failure_class` fields deserialize into the closed enums — a value outside the
/// taxonomy is a load-time error, never a silently-accepted string.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedLabelConfig {
    /// The closed outcome kind this label is a flavor of (e.g. `downstream_failure`).
    pub outcome_kind: OutcomeKind,
    /// The closed failure class this label rolls up into (e.g. `manipulation`).
    pub failure_class: FailureClass,
    /// The task this label is scoped to, if any (a free-text task id like `pick_can`).
    #[serde(default)]
    pub task: Option<String>,
    /// Triage severity; defaults to `standard`.
    #[serde(default)]
    pub severity: Severity,
    /// An optional per-label attribution-window override in milliseconds. When absent,
    /// the binding falls back to the embodiment/outcome-kind window. Must be > 0.
    #[serde(default)]
    pub attribution_window_ms: Option<u64>,
}

/// A validated label definition, carrying its own name.
///
/// By the time runtime code holds one of these, the name is set, the bindings resolve
/// to real taxonomy members, and any declared window is positive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelConfig {
    /// The customer's label name (e.g. `failed_grasp`), canonicalized to lowercase.
    pub name: String,
    /// The closed outcome kind this label is a flavor of.
    pub outcome_kind: OutcomeKind,
    /// The closed failure class this label rolls up into.
    pub failure_class: FailureClass,
    /// The task this label is scoped to, if any.
    pub task: Option<String>,
    /// Triage severity.
    pub severity: Severity,
    /// An optional per-label attribution-window override (ms).
    pub attribution_window_ms: Option<u64>,
}

impl UninitializedLabelConfig {
    /// Resolve into a validated [`LabelConfig`] under `name`. The taxonomy bindings are
    /// already validated by deserialization; this carries the name through and checks
    /// the remaining per-label invariants (a declared window must be positive).
    pub fn load(self, name: String) -> Result<LabelConfig, ConfigError> {
        if self.attribution_window_ms == Some(0) {
            return Err(ConfigError::InvalidLabel {
                name,
                reason: "attribution_window_ms must be greater than zero when set",
            });
        }
        Ok(LabelConfig {
            name,
            outcome_kind: self.outcome_kind,
            failure_class: self.failure_class,
            task: self.task,
            severity: self.severity,
            attribution_window_ms: self.attribution_window_ms,
        })
    }
}
