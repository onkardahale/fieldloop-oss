//! Python binding over the `fieldloop-join` decision→outcome attribution engine.
//!
//! This is the loop's analytic core surfaced to Python: given the rollouts a policy
//! produced (each a decision) and the outcomes the world reported back (each usually
//! WITHOUT a reference to which decision caused it), it binds outcome→rollout and emits
//! calibrated [`fieldloop_types::Feedback`] rows. That inference — attributing a
//! delayed, unlabeled outcome to the decision that caused it — is the problem Fieldloop
//! exists to solve, and `attribute` runs the real engine, not a reimplementation.
//!
//! Like the rest of this crate it is a THIN marshaling layer over [`crate::marshal`]:
//! it converts Python dicts to the Rust schema, calls
//! [`fieldloop_join::attribute_report`], and converts the result back. Every
//! attribution decision (the explicit→temporal→absence cascade, cross-tenant rejection,
//! confidence calibration) lives in the engine, never here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fieldloop_config::Config;
use fieldloop_join::{AttributeOptions, SkipReason, Skipped, attribute_report};

use crate::marshal::{
    Routed, dict_to_outcome, dict_to_rollout, feedback_to_dict, outcome_kind_str,
};

/// Snake_case spelling of a [`SkipReason`], so a non-binding is auditable from Python
/// rather than opaque.
fn skip_reason_str(reason: &SkipReason) -> &'static str {
    match reason {
        SkipReason::CrossTenant => "cross_tenant",
        SkipReason::NoCandidateInWindow => "no_candidate_in_window",
        SkipReason::NoWindowConfigured => "no_window_configured",
        SkipReason::AmbiguousMonotonicColocation => "ambiguous_monotonic_colocation",
        SkipReason::HeartbeatSample => "heartbeat_sample",
        SkipReason::NoSpatialOrCausalCandidate => "no_spatial_or_causal_candidate",
    }
}

/// Convert one non-binding into a flat dict: which outcome, of what kind, and why it
/// did not bind — so a dropped outcome is visible to Python, never silently lost.
fn skipped_to_dict<'py>(py: Python<'py>, sk: &Skipped) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("outcome_id", sk.outcome.id.to_string())?;
    d.set_item("outcome_kind", outcome_kind_str(sk.outcome.outcome_kind))?;
    d.set_item("reason", skip_reason_str(&sk.reason))?;
    Ok(d)
}

/// Attribute outcomes to the rollouts that caused them, returning the bindings and the
/// auditable non-bindings.
///
/// `config_toml` is the embodiment/attribution-window config parsed into a real
/// `Config`; `rollouts` and `outcomes` are lists of the flat dicts described on
/// [`crate::marshal`]. A dict with `outcome_kind == "heartbeat"` is routed to the
/// coverage set, not the failure cascade.
///
/// Returns `{"feedbacks": [...], "skipped": [...]}`: each feedback is one attributed
/// binding (target, value, join method + calibrated confidence); each skipped entry is
/// an outcome that did not bind, with the reason. Raises `ValueError` on an invalid
/// config, a malformed id, an unknown outcome kind, or a missing required field —
/// before any partial result is produced.
#[pyfunction]
#[pyo3(signature = (config_toml, rollouts, outcomes, *, join_version=None, heartbeat_period_ns=None, absence_metric_name=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn attribute<'py>(
    py: Python<'py>,
    config_toml: &str,
    rollouts: Vec<Bound<'py, PyDict>>,
    outcomes: Vec<Bound<'py, PyDict>>,
    join_version: Option<String>,
    heartbeat_period_ns: Option<u64>,
    absence_metric_name: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = Config::from_toml_str(config_toml)
        .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;

    let mut roll = Vec::with_capacity(rollouts.len());
    for d in &rollouts {
        roll.push(dict_to_rollout(d)?);
    }

    let mut events = Vec::new();
    let mut heartbeats = Vec::new();
    for d in &outcomes {
        match dict_to_outcome(d)? {
            Routed::Outcome(o) => events.push(*o),
            Routed::Heartbeat(h) => heartbeats.push(h),
        }
    }

    // Start from the engine defaults and override only what the caller pinned, so a call
    // that passes nothing gets the same well-chosen defaults the Rust API does.
    let mut opts = AttributeOptions::default();
    if let Some(version) = join_version {
        opts.join_version = version;
    }
    opts.heartbeat_period_ns = heartbeat_period_ns;
    if let Some(metric) = absence_metric_name {
        opts.absence_metric_name = metric;
    }

    let report = attribute_report(&config, &roll, &events, &heartbeats, &opts);

    let result = PyDict::new(py);
    let feedbacks = PyList::empty(py);
    for fb in &report.feedbacks {
        feedbacks.append(feedback_to_dict(py, fb)?)?;
    }
    result.set_item("feedbacks", feedbacks)?;

    let skipped = PyList::empty(py);
    for sk in &report.skipped {
        skipped.append(skipped_to_dict(py, sk)?)?;
    }
    result.set_item("skipped", skipped)?;
    Ok(result)
}
