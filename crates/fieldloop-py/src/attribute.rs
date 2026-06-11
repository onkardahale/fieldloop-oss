//! Python binding over the `fieldloop-join` decision→outcome attribution engine.
//!
//! Given the rollouts a policy produced (each a decision) and the outcomes the world
//! reported back (each usually WITHOUT a reference to which decision caused it), it binds
//! outcome→rollout and emits calibrated [`fieldloop_types::Feedback`] rows. `attribute`
//! runs the real engine, not a reimplementation.
//!
//! Thin marshaling layer over [`crate::marshal`]: converts Python dicts to the Rust
//! schema, calls [`fieldloop_join::attribute_report`], and converts the result back.
//! Every attribution decision (the explicit→temporal→absence cascade, cross-tenant
//! rejection, confidence calibration) lives in the engine, never here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fieldloop_config::Config;
use fieldloop_import::{MappingConfig, import_mcap};
use fieldloop_join::{
    AttributeOptions, AttributionReport, FittedCalibrator, Heartbeat, SkipReason, Skipped,
    attribute_report, attribute_with,
};

use crate::marshal::{
    Routed, dict_to_feedback, dict_to_outcome, dict_to_rollout, feedback_to_dict, outcome_kind_str,
    route_outcome,
};

/// A fitted confidence calibrator, built from a team's own curator-confirmed labels and
/// passed back into [`attribute`] so a binding's confidence reflects observed accuracy
/// rather than the raw recency/coverage score.
///
/// Opaque on purpose: the fitted curves are an internal of the Rust engine. A caller
/// builds one with [`fit_calibrator`] and hands it to `attribute(..., calibrator=cal)`;
/// without one, attribution uses the honest identity default (confidence = raw score).
#[pyclass(name = "Calibrator", frozen)]
pub struct PyCalibrator {
    pub(crate) inner: FittedCalibrator,
}

/// Fit a [`Calibrator`](PyCalibrator) from feedback history: the curator's confirmed
/// `manual` labels are the ground truth an inferred binding's score is calibrated
/// against, bucketed per `(join_method, embodiment)` and fit with isotonic regression.
///
/// `feedbacks` and `rollouts` are the same dict shapes `attribute`/`curate` use.
/// `min_labels` is the per-bucket sample floor below which a bucket stays uncalibrated
/// (identity), so a thin bucket never fabricates a curve. The result is opaque; pass it
/// to `attribute(..., calibrator=...)`.
#[pyfunction]
#[pyo3(signature = (feedbacks, rollouts, min_labels=4))]
#[allow(clippy::needless_pass_by_value)]
pub fn fit_calibrator(
    feedbacks: Vec<Bound<'_, PyDict>>,
    rollouts: Vec<Bound<'_, PyDict>>,
    min_labels: usize,
) -> PyResult<PyCalibrator> {
    let mut fbs = Vec::with_capacity(feedbacks.len());
    for d in &feedbacks {
        fbs.push(dict_to_feedback(d)?);
    }
    let mut roll = Vec::with_capacity(rollouts.len());
    for d in &rollouts {
        roll.push(dict_to_rollout(d)?);
    }
    Ok(PyCalibrator {
        inner: FittedCalibrator::fit(&fbs, &roll, min_labels),
    })
}

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
/// binding (target, value, join method + a scored confidence); each skipped entry is
/// an outcome that did not bind, with the reason. The default confidence is the raw
/// recency/coverage score surfaced unchanged (the identity calibrator) — it is not yet
/// fit to field outcomes, so a `0.9` means "close and well-covered", not "90% of such
/// bindings proved correct". Raises `ValueError` on an invalid
/// config, a malformed id, an unknown outcome kind, or a missing required field —
/// before any partial result is produced.
#[pyfunction]
#[pyo3(signature = (config_toml, rollouts, outcomes, *, join_version=None, heartbeat_period_ns=None, absence_metric_name=None, calibrator=None))]
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
pub fn attribute<'py>(
    py: Python<'py>,
    config_toml: &str,
    rollouts: Vec<Bound<'py, PyDict>>,
    outcomes: Vec<Bound<'py, PyDict>>,
    join_version: Option<String>,
    heartbeat_period_ns: Option<u64>,
    absence_metric_name: Option<String>,
    calibrator: Option<PyRef<'py, PyCalibrator>>,
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

    let opts = build_options(join_version, heartbeat_period_ns, absence_metric_name);
    let report = run_engine(
        &config,
        calibrator.as_deref(),
        &roll,
        &events,
        &heartbeats,
        &opts,
    );
    report_to_pydict(py, &report)
}

/// Build engine options from the optional per-call overrides, starting from the same
/// defaults the Rust API uses so a call that pins nothing behaves identically.
fn build_options(
    join_version: Option<String>,
    heartbeat_period_ns: Option<u64>,
    absence_metric_name: Option<String>,
) -> AttributeOptions {
    let mut opts = AttributeOptions::default();
    if let Some(version) = join_version {
        opts.join_version = version;
    }
    opts.heartbeat_period_ns = heartbeat_period_ns;
    if let Some(metric) = absence_metric_name {
        opts.absence_metric_name = metric;
    }
    opts
}

/// Run the attribution engine, with the fitted calibrator if one was passed. With a
/// calibrator the binding confidences reflect observed accuracy; without one, the
/// identity default applies (confidence = raw recency/coverage score).
fn run_engine(
    config: &Config,
    calibrator: Option<&PyCalibrator>,
    rollouts: &[fieldloop_types::Rollout],
    events: &[fieldloop_types::OutcomeEvent],
    heartbeats: &[Heartbeat],
    opts: &AttributeOptions,
) -> AttributionReport {
    match calibrator {
        Some(cal) => attribute_with(config, &cal.inner, rollouts, events, heartbeats, opts),
        None => attribute_report(config, rollouts, events, heartbeats, opts),
    }
}

/// Project an [`AttributionReport`] onto the `{"feedbacks": [...], "skipped": [...]}` dict
/// both `attribute` and `attribute_mcap` return, so the two entry points emit an identical
/// shape regardless of whether the inputs came from dicts or an MCAP file.
fn report_to_pydict<'py>(
    py: Python<'py>,
    report: &AttributionReport,
) -> PyResult<Bound<'py, PyDict>> {
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

/// Attribute the outcomes in an MCAP file to the decisions that caused them.
///
/// The file-import counterpart to [`attribute`]: instead of rollout/outcome dicts, it
/// takes the raw bytes of an MCAP recording plus a topic-mapping TOML (`mapping_toml`,
/// the [`MappingConfig`] format — identity constants, a `[clock]` source, and the
/// decision/outcome topic lists). It decodes the file into typed rollouts and outcomes,
/// routes any `heartbeat`-kind outcome to the coverage path, runs the same engine, and
/// returns the same `{"feedbacks": [...], "skipped": [...]}` shape. `config_toml` is the
/// embodiment/attribution-window config, exactly as for [`attribute`]; the mapping's
/// `embodiment` must name one present in it. Raises `ValueError` on an invalid config, an
/// invalid mapping, or an undecodable MCAP, before any partial result.
#[pyfunction]
#[pyo3(signature = (config_toml, mcap_bytes, mapping_toml, *, join_version=None, heartbeat_period_ns=None, absence_metric_name=None, calibrator=None))]
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
pub fn attribute_mcap<'py>(
    py: Python<'py>,
    config_toml: &str,
    mcap_bytes: &[u8],
    mapping_toml: &str,
    join_version: Option<String>,
    heartbeat_period_ns: Option<u64>,
    absence_metric_name: Option<String>,
    calibrator: Option<PyRef<'py, PyCalibrator>>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = Config::from_toml_str(config_toml)
        .map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))?;
    let mapping = MappingConfig::from_toml(mapping_toml)
        .map_err(|e| PyValueError::new_err(format!("invalid mapping: {e}")))?;
    let imported = import_mcap(mcap_bytes, &mapping)
        .map_err(|e| PyValueError::new_err(format!("could not import MCAP: {e}")))?;

    let mut events = Vec::new();
    let mut heartbeats = Vec::new();
    for outcome in imported.outcomes {
        match route_outcome(outcome) {
            Routed::Outcome(o) => events.push(*o),
            Routed::Heartbeat(h) => heartbeats.push(h),
        }
    }

    let opts = build_options(join_version, heartbeat_period_ns, absence_metric_name);
    let report = run_engine(
        &config,
        calibrator.as_deref(),
        &imported.rollouts,
        &events,
        &heartbeats,
        &opts,
    );
    report_to_pydict(py, &report)
}
