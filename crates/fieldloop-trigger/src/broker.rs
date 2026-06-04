//! The selective-upload broker: turns a window's worth of trigger events into a
//! bounded set of payload-pull requests under a budget.
//!
//! Fieldloop cannot upload every robot's full sensor payload — it is terabytes a day.
//! The detectors flag far more events than the system can afford to pull, so this
//! broker is the gate that picks the affordable subset. It ranks events by priority,
//! keeps the highest under a budget, and — critically — never silently discards the
//! rest: everything dropped is counted and returned as a [`DroppedEvent`] in the
//! [`UploadDecision`] so an operator can see what coverage was sacrificed and tune
//! the budget.
//!
//! ## Safety events bypass the budget
//! A [`crate::DetectorTier::ReflexA`] trigger (a collision, an e-stop, a takeover) is
//! a hard-safety signal. Dropping its payload because a budget filled up would lose
//! exactly the evidence a safety review needs, which is never an acceptable trade. So
//! safety events are always included, even past the budget — the budget only governs
//! the *droppable* (risky-looking / learned) tiers.

use fieldloop_types::{OutcomeId, RolloutId};

use crate::detector::{DetectorTier, TriggerEvent};

/// One request to pull a rollout/outcome's heavy sensor payload from the customer's
/// store into Fieldloop's curation path. Pure data — the broker decides *which* to
/// emit; an out-of-crate worker does the actual byte movement.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveUploadRequest {
    /// The rollout whose payload to pull.
    pub rollout_id: RolloutId,
    /// The originating outcome, when the trigger had one. `None` for a pre-filter
    /// trigger that fired on the rollout alone.
    pub outcome_id: Option<OutcomeId>,
    /// The detector that asked for this pull, for provenance.
    pub detector_id: &'static str,
    /// The tier the request came from, so the worker can see whether it was a
    /// budgeted pull or an unconditional safety pull.
    pub tier: DetectorTier,
    /// The priority the broker selected on, carried through for audit/ordering.
    pub priority: f32,
}

impl SelectiveUploadRequest {
    fn from_event(event: &TriggerEvent) -> Self {
        Self {
            rollout_id: event.rollout_id,
            outcome_id: event.outcome_id,
            detector_id: event.detector_id,
            tier: event.tier,
            priority: event.raw_priority,
        }
    }
}

/// The budget that bounds one selective-upload window. Both limits apply; an event is
/// admitted only if it fits under *both*. A safety event bypasses both (see
/// [`SelectiveUploadBroker::select`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UploadBudget {
    /// Maximum number of budgeted (non-safety) requests this window may emit. The
    /// coarse limit on how many payload pulls the curation path can absorb.
    pub max_requests: usize,
    /// Maximum total estimated bytes across budgeted requests. The finer limit on the
    /// actual data volume, so a window of large payloads fills the budget sooner than
    /// a window of small ones.
    pub max_bytes: u64,
}

impl UploadBudget {
    /// A budget bounded only by a request count, with no byte ceiling (the byte limit
    /// is set to the maximum so it never binds). Useful when payload sizes are
    /// unknown and only the count matters.
    #[must_use]
    pub fn by_count(max_requests: usize) -> Self {
        Self {
            max_requests,
            max_bytes: u64::MAX,
        }
    }
}

/// A dropped event, kept and surfaced rather than silently discarded, so an operator
/// can see exactly what coverage the budget cost and which detector lost out.
#[derive(Debug, Clone, PartialEq)]
pub struct DroppedEvent {
    /// The rollout whose payload was *not* pulled.
    pub rollout_id: RolloutId,
    /// The detector whose request was dropped.
    pub detector_id: &'static str,
    /// The tier of the dropped request (always a droppable tier — safety events are
    /// never dropped).
    pub tier: DetectorTier,
    /// The priority it lost out at, so an operator can see how close it came.
    pub priority: f32,
}

/// The result of a selective-upload decision: what was requested and what was
/// dropped. Returning the drops as first-class data (not a log line that may be
/// missed) is what makes "dropped-and-counted, never silent" testable and auditable.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UploadDecision {
    /// The payload pulls to perform, safety events first, then the highest-priority
    /// budgeted events that fit.
    pub requests: Vec<SelectiveUploadRequest>,
    /// Every event the budget forced out, counted and described.
    pub dropped: Vec<DroppedEvent>,
}

impl UploadDecision {
    /// How many requests will be pulled.
    #[must_use]
    pub fn requested_count(&self) -> usize {
        self.requests.len()
    }

    /// How many events were dropped by the budget — the coverage cost of the window.
    #[must_use]
    pub fn dropped_count(&self) -> usize {
        self.dropped.len()
    }
}

/// The selective-upload broker. Holds the budget and the per-event byte estimate it
/// charges against [`UploadBudget::max_bytes`]; stateless beyond that, so a window is
/// just one [`SelectiveUploadBroker::select`] call.
#[derive(Debug, Clone)]
pub struct SelectiveUploadBroker {
    budget: UploadBudget,
    /// Bytes charged per budgeted request against the byte ceiling. A flat estimate
    /// keeps the broker pure and deterministic without needing to size every payload;
    /// a real deployment can refine this per request, but the selection logic is the
    /// same.
    bytes_per_request: u64,
}

impl SelectiveUploadBroker {
    /// Build a broker with a budget and a flat per-request byte estimate.
    #[must_use]
    pub fn new(budget: UploadBudget, bytes_per_request: u64) -> Self {
        Self {
            budget,
            bytes_per_request,
        }
    }

    /// Decide which of `events` to pull under the budget.
    ///
    /// The rule, in order:
    /// 1. Every [`DetectorTier::ReflexA`] safety event is included unconditionally —
    ///    it bypasses both the count and the byte limits, because losing a safety
    ///    payload to a full budget is never acceptable.
    /// 2. The remaining (droppable) events are sorted by priority, highest first, and
    ///    admitted greedily until either limit would be exceeded.
    /// 3. Every droppable event that did not fit is recorded in
    ///    [`UploadDecision::dropped`] — counted and surfaced, never silently lost.
    ///
    /// Determinism: ties in priority are broken by `rollout_id` so the selection is
    /// reproducible on identical input regardless of the order events arrived in.
    #[must_use]
    pub fn select(&self, events: &[TriggerEvent]) -> UploadDecision {
        let mut decision = UploadDecision::default();

        // 1. Safety events are pulled unconditionally and do not consume the budget.
        for event in events.iter().filter(|e| e.tier.is_safety_critical()) {
            decision
                .requests
                .push(SelectiveUploadRequest::from_event(event));
        }

        // 2. Rank the droppable events: priority desc, then rollout_id for a stable,
        //    reproducible tiebreak.
        let mut droppable: Vec<&TriggerEvent> = events
            .iter()
            .filter(|e| !e.tier.is_safety_critical())
            .collect();
        droppable.sort_by(|a, b| {
            b.raw_priority
                .total_cmp(&a.raw_priority)
                .then_with(|| a.rollout_id.as_uuid().cmp(&b.rollout_id.as_uuid()))
        });

        // 3. Admit greedily under both limits; record the rest as dropped.
        let mut admitted = 0usize;
        let mut bytes_used = 0u64;
        for event in droppable {
            let next_bytes = bytes_used.saturating_add(self.bytes_per_request);
            let fits = admitted < self.budget.max_requests && next_bytes <= self.budget.max_bytes;
            if fits {
                admitted += 1;
                bytes_used = next_bytes;
                decision
                    .requests
                    .push(SelectiveUploadRequest::from_event(event));
            } else {
                decision.dropped.push(DroppedEvent {
                    rollout_id: event.rollout_id,
                    detector_id: event.detector_id,
                    tier: event.tier,
                    priority: event.raw_priority,
                });
            }
        }

        decision
    }
}
