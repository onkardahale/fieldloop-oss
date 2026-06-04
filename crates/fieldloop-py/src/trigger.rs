//! Python binding over the `fieldloop-trigger` detectors + selective-upload broker.
//!
//! The full sensor stream is terabytes/day — Fieldloop cannot pull every robot's
//! payload, so this stage decides which rollouts are worth the heavy upload.
//! `select_uploads` runs the two built-in pure detectors over the rollouts/outcomes and
//! feeds their triggers to the real budget broker:
//! - the **reflex** detector fires on an unambiguous "it went wrong" outcome (e-stop,
//!   collision, takeover, downstream failure) — a hard-safety signal;
//! - the **low-confidence** pre-filter fires on a rollout whose `policy_confidence` tag
//!   is below threshold (a "this looks risky" guess, before any outcome).
//!
//! The broker then keeps the highest-priority triggers under a budget and, critically,
//! never silently drops the rest: every drop is returned as data. Safety (reflex)
//! triggers bypass the budget and are always kept — dropping a collision's payload
//! because a budget filled up would lose exactly the evidence a safety review needs.
//! All of that logic lives in the engine; this binding only marshals.
//!
//! An outcome is paired to the rollout it fired on via its `explicit_rollout_id`, so a
//! reflex trigger knows which rollout's payload to pull.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fieldloop_trigger::{
    Detector, DetectorTier, LowConfidenceDetector, ReflexDetector, SelectiveUploadBroker,
    UploadBudget,
};
use fieldloop_types::{OutcomeEvent, RolloutId};

use crate::marshal::{Routed, dict_to_outcome, dict_to_rollout};

/// Snake_case spelling of a [`DetectorTier`], so a request/drop names the tier that
/// produced it (and whether it was a safety pull or a budgeted one).
fn tier_str(tier: DetectorTier) -> &'static str {
    match tier {
        DetectorTier::ReflexA => "reflex_a",
        DetectorTier::PreFilterAPrime => "pre_filter_a_prime",
        DetectorTier::LearnedB => "learned_b",
    }
}

/// Decide which rollouts' heavy sensor payloads to pull, under a budget.
///
/// `rollouts` and `outcomes` are the same flat dicts the other bindings use (an outcome
/// pairs to its rollout via `explicit_rollout_id`; a `policy_confidence` rollout field
/// arms the low-confidence pre-filter). `max_requests` caps how many budgeted pulls are
/// kept; `max_bytes` (with `bytes_per_request`) optionally caps total bytes. Safety
/// (reflex) triggers are kept even past the budget.
///
/// Returns `{"requests": [...], "dropped": [...]}`: each request is a payload pull
/// (rollout, optional outcome, detector, tier, priority); each dropped entry is a
/// budgeted trigger that lost out, surfaced so an operator sees the sacrificed coverage.
/// Raises `ValueError` on a malformed rollout/outcome dict.
#[pyfunction]
#[pyo3(signature = (rollouts, outcomes, *, max_requests, max_bytes=None, bytes_per_request=0))]
#[allow(clippy::needless_pass_by_value)]
pub fn select_uploads<'py>(
    py: Python<'py>,
    rollouts: Vec<Bound<'py, PyDict>>,
    outcomes: Vec<Bound<'py, PyDict>>,
    max_requests: usize,
    max_bytes: Option<u64>,
    bytes_per_request: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut roll = Vec::with_capacity(rollouts.len());
    for d in &rollouts {
        roll.push(dict_to_rollout(d)?);
    }

    // Pair each outcome to the rollout it fired on by its explicit id, so the reflex
    // detector inspects a rollout alongside its own outcome. Heartbeats are coverage,
    // not failures, so they are dropped here (a reflex never fires on one).
    let mut by_rollout: HashMap<RolloutId, OutcomeEvent> = HashMap::new();
    for d in &outcomes {
        if let Routed::Outcome(o) = dict_to_outcome(d)?
            && let Some(rid) = o.explicit_rollout_id
        {
            by_rollout.insert(rid, *o);
        }
    }

    // Run the two built-in detectors over every rollout. Reflex sees the rollout's
    // paired outcome (if any); the low-confidence pre-filter fires on the rollout alone.
    let reflex = ReflexDetector;
    let low_confidence = LowConfidenceDetector;
    let mut events = Vec::new();
    for rollout in &roll {
        let paired = by_rollout.get(&rollout.id);
        if let Some(event) = reflex.inspect(rollout, paired) {
            events.push(event);
        }
        if let Some(event) = low_confidence.inspect(rollout, None) {
            events.push(event);
        }
    }

    // The byte budget is unbounded by default (only the request count binds) so a caller
    // can use a pure count budget without computing per-request sizes.
    let budget = UploadBudget {
        max_requests,
        max_bytes: max_bytes.unwrap_or(u64::MAX),
    };
    let decision = SelectiveUploadBroker::new(budget, bytes_per_request).select(&events);

    let result = PyDict::new(py);
    let requests = PyList::empty(py);
    for request in &decision.requests {
        let row = PyDict::new(py);
        row.set_item("rollout_id", request.rollout_id.to_string())?;
        row.set_item("outcome_id", request.outcome_id.map(|i| i.to_string()))?;
        row.set_item("detector_id", request.detector_id)?;
        row.set_item("tier", tier_str(request.tier))?;
        row.set_item("priority", request.priority)?;
        requests.append(row)?;
    }
    result.set_item("requests", requests)?;

    let dropped = PyList::empty(py);
    for drop in &decision.dropped {
        let row = PyDict::new(py);
        row.set_item("rollout_id", drop.rollout_id.to_string())?;
        row.set_item("detector_id", drop.detector_id)?;
        row.set_item("tier", tier_str(drop.tier))?;
        row.set_item("priority", drop.priority)?;
        dropped.append(row)?;
    }
    result.set_item("dropped", dropped)?;
    Ok(result)
}
