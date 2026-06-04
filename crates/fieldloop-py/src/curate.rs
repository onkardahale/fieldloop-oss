//! Python binding over the `fieldloop-curation` training-slice compiler.
//!
//! Curation is the loop stage after attribution: given the rollouts and the feedback
//! that scored them, it pins exactly which `(rollout, authoritative-feedback)` pairs
//! become training data. The judgement it encodes is the point — a binding below the
//! confidence floor, or a retracted one, is HELD OUT and flagged needs-review rather
//! than silently trained on, because a doubtful label poisons the model. `curate` runs
//! the real [`fieldloop_curation::compile_slice`]; nothing is reimplemented here.
//!
//! It closes the loop with attribution: `curate` reads back exactly the feedback dicts
//! `attribute` returned (via [`crate::marshal::dict_to_feedback`]), so a Python caller
//! pipes `attribute(...)["feedbacks"]` straight into `curate`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fieldloop_curation::{
    DEFAULT_MIN_CONFIDENCE, Grain, OutcomeTsWindow, ResolvedSlice, ReviewReason, SliceSpec,
    compile_slice,
};
use fieldloop_types::{PolicyVersion, Provenance};

use crate::marshal::{
    dict_to_feedback, dict_to_rollout, field, opt_bool, opt_f32, opt_i64_opt, opt_str,
    parse_failure_class, req_str,
};

/// Marshal a spec dict into a [`SliceSpec`].
///
/// `grain` is the one required choice (per-step vs per-trajectory datasets are genuinely
/// different shapes, so there is no safe default). Every other field is an optional,
/// additive filter that can only narrow the slice; `min_confidence` defaults to the
/// conservative [`DEFAULT_MIN_CONFIDENCE`] and `include_synthetic` to `false` (synthetic
/// data is opt-in). `outcome_ts` is an optional nested `{start_ns, end_ns}` window.
fn dict_to_slice_spec(d: &Bound<'_, PyDict>) -> PyResult<SliceSpec> {
    let grain = match req_str(d, "grain")?.as_str() {
        "rollout" => Grain::Rollout,
        "episode" => Grain::Episode,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown grain `{other}` (expected `rollout` or `episode`)"
            )));
        }
    };
    let outcome_ts = match field(d, "outcome_ts")? {
        Some(v) => {
            let w = v.downcast_into::<PyDict>().map_err(|_| {
                PyValueError::new_err("field `outcome_ts` must be a dict with start_ns/end_ns")
            })?;
            OutcomeTsWindow {
                start_ns: opt_i64_opt(&w, "start_ns")?,
                end_ns: opt_i64_opt(&w, "end_ns")?,
            }
        }
        None => OutcomeTsWindow::default(),
    };
    Ok(SliceSpec {
        policy_version: opt_str(d, "policy_version")?.map(PolicyVersion::new),
        failure_class: match opt_str(d, "failure_class")? {
            Some(s) => Some(parse_failure_class(&s)?),
            None => None,
        },
        task_id: opt_str(d, "task_id")?,
        site: opt_str(d, "site")?,
        outcome_ts,
        min_confidence: opt_f32(d, "min_confidence")?.unwrap_or(DEFAULT_MIN_CONFIDENCE),
        include_synthetic: opt_bool(d, "include_synthetic", false)?,
        grain,
    })
}

/// Whether a pinned item's rollout was synthetic or real field evidence — carried so a
/// consumer can separate or down-weight synthetic data and it is never silently treated
/// as real.
fn provenance_str(provenance: &Provenance) -> &'static str {
    if provenance.is_synthetic() {
        "synthetic"
    } else {
        "real"
    }
}

/// Why a candidate was held out of training data.
fn review_reason_str(reason: &ReviewReason) -> &'static str {
    match reason {
        ReviewReason::BelowMinConfidence => "below_min_confidence",
        ReviewReason::Retracted => "retracted",
        ReviewReason::NoSurvivingBinding => "no_surviving_binding",
    }
}

/// Convert the compiled, content-hashed slice into a dict: the pinned training items
/// (ids only — a rebuild re-fetches by id), the rolled-up episode set, the held-out
/// needs-review candidates with reasons, and the stable content hash.
fn resolved_slice_to_dict<'py>(
    py: Python<'py>,
    slice: &ResolvedSlice,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("tenant_id", slice.tenant_id.to_string())?;
    d.set_item("content_hash", slice.content_hash.as_str())?;

    let items = PyList::empty(py);
    for item in &slice.items {
        let row = PyDict::new(py);
        row.set_item("episode_id", item.episode_id.to_string())?;
        row.set_item("rollout_id", item.rollout_id.to_string())?;
        row.set_item("feedback_id", item.feedback_id.to_string())?;
        row.set_item("provenance", provenance_str(&item.provenance))?;
        items.append(row)?;
    }
    d.set_item("items", items)?;

    let episodes = PyList::empty(py);
    for episode in &slice.episodes {
        episodes.append(episode.to_string())?;
    }
    d.set_item("episodes", episodes)?;

    let needs_review = PyList::empty(py);
    for nr in &slice.needs_review {
        let row = PyDict::new(py);
        row.set_item("rollout_id", nr.rollout_id.to_string())?;
        row.set_item("feedback_id", nr.feedback_id.map(|f| f.to_string()))?;
        row.set_item("reason", review_reason_str(&nr.reason))?;
        needs_review.append(row)?;
    }
    d.set_item("needs_review", needs_review)?;
    Ok(d)
}

/// Compile a training slice: pin which `(rollout, authoritative-feedback)` pairs become
/// training data, holding out and flagging the doubtful ones.
///
/// `spec` is a slice-spec dict (see [`dict_to_slice_spec`]); `rollouts` and `feedbacks`
/// are the same dicts `attribute` consumes and returns. For each matching rollout the
/// compiler picks the authoritative binding (manual outranks automated, else newest),
/// then applies the confidence gate: a retracted binding, or one below
/// `min_confidence`, is excluded and surfaced under `needs_review`.
///
/// Returns `{"tenant_id", "content_hash", "items": [...], "episodes": [...],
/// "needs_review": [...]}`. Raises `ValueError` on a malformed spec, rollout, or
/// feedback dict.
#[pyfunction]
#[pyo3(signature = (spec, rollouts, feedbacks))]
#[allow(clippy::needless_pass_by_value)]
pub fn curate<'py>(
    py: Python<'py>,
    spec: Bound<'py, PyDict>,
    rollouts: Vec<Bound<'py, PyDict>>,
    feedbacks: Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = dict_to_slice_spec(&spec)?;

    let mut roll = Vec::with_capacity(rollouts.len());
    for d in &rollouts {
        roll.push(dict_to_rollout(d)?);
    }
    let mut feedback = Vec::with_capacity(feedbacks.len());
    for d in &feedbacks {
        feedback.push(dict_to_feedback(d)?);
    }

    let slice = compile_slice(&spec, &roll, &feedback);
    resolved_slice_to_dict(py, &slice)
}
