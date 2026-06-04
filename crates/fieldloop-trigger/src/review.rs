//! Human-in-the-loop review: a bounded, prioritized queue of trigger events for a
//! human to label, and the path that turns a human's label into ground truth.
//!
//! Human review time is the scarcest resource in the loop — far scarcer than upload
//! bandwidth. Even among the events worth pulling, only a fraction can ever get a
//! human's eyes. So this queue does active-learning prioritization: it ranks events
//! by where a human label is worth the *most* and keeps only the top-K.
//!
//! ## The priority function
//! The value of getting a human label on an event is highest where the model is most
//! *uncertain* and the event is most *valuable* to resolve. So
//! [`review_priority`] scores an event as `uncertainty * value`:
//! * **uncertainty** peaks where the detector's `raw_priority` is near `0.5` — a
//!   confident fire (near `1.0`) or a confident pass (near `0.0`) teaches the model
//!   little, while a coin-flip is exactly where a human label resolves the most
//!   ambiguity. Modeled as `1 - 2*|raw_priority - 0.5|`, a triangle peaking at `0.5`.
//! * **value** is higher for a safety-tier event: confirming the class of a
//!   collision/e-stop is worth more than confirming a routine low-confidence
//!   pre-filter hit.
//!
//! Ranking by `uncertainty * value` (not by raw detector priority) deliberately
//! prefers the events a human can *teach* the model the most from, which is the whole
//! point of active learning.
//!
//! ## The human label is ground truth
//! Accepting a human label produces a [`LabelRecord`] carrying the human-assigned
//! [`fieldloop_types::FailureClass`] (or a success/fail flag). This is the ground
//! truth: it overrides the detector's mere *hint*, and it is the signal the
//! confidence calibration downstream is fit against — automated attribution
//! confidences are measured against these human labels, never asserted. So a
//! [`LabelRecord`] keeps the human's class and the detector's original hint
//! side-by-side, making a human override of the hint explicit and auditable.

use fieldloop_types::{FailureClass, OutcomeId, RolloutId};

use crate::detector::{DetectorTier, TriggerEvent};

/// The value weight given to a safety-tier event in the review priority. A safety
/// event (collision/e-stop/takeover) is worth more to label than a routine
/// pre-filter hit, so its uncertainty is scaled up by this factor. A modest multiplier
/// (not a hard override) so a genuinely uncertain non-safety event can still outrank a
/// near-certain safety one.
const SAFETY_VALUE_WEIGHT: f32 = 2.0;

/// The value weight for a non-safety (droppable-tier) event.
const ROUTINE_VALUE_WEIGHT: f32 = 1.0;

/// The active-learning priority of getting a human label on `event`: `uncertainty *
/// value`. Higher = a human's time is better spent here.
///
/// `uncertainty` is a triangle peaking at a `raw_priority` of `0.5` (a coin-flip,
/// where a label resolves the most ambiguity) and falling to `0` at a confident `0.0`
/// or `1.0` (where the detector is already sure and a label teaches little). `value`
/// up-weights safety-tier events. See the module docs for why this beats ranking by
/// the raw detector priority.
#[must_use]
pub fn review_priority(event: &TriggerEvent) -> f32 {
    let uncertainty = 1.0 - 2.0 * (event.raw_priority.clamp(0.0, 1.0) - 0.5).abs();
    let value = if event.tier == DetectorTier::ReflexA {
        SAFETY_VALUE_WEIGHT
    } else {
        ROUTINE_VALUE_WEIGHT
    };
    uncertainty * value
}

/// A bounded review queue that keeps the top-K trigger events by [`review_priority`].
///
/// Bounded because human review time is finite: past `capacity`, the lowest-priority
/// event is evicted so the queue always holds the `capacity` events most worth a
/// human's time. The queue is kept sorted highest-priority-first so a reviewer pulls
/// the most valuable label next.
#[derive(Debug, Clone)]
pub struct ReviewQueue {
    capacity: usize,
    /// Events with their precomputed review priority, kept sorted highest-first.
    entries: Vec<(f32, TriggerEvent)>,
}

impl ReviewQueue {
    /// A queue holding at most `capacity` events. A zero capacity is a queue that
    /// accepts nothing (review is disabled), which is a valid configured state.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
        }
    }

    /// Offer an event to the queue. It is admitted if there is room, or if it
    /// outranks the current lowest-priority entry (which is then evicted). An event
    /// that is not better than the weakest of a full queue is rejected, so the queue
    /// always converges to the top-K by priority regardless of insertion order. Ties
    /// in priority are broken by `rollout_id` so the ordering is deterministic.
    pub fn offer(&mut self, event: TriggerEvent) {
        let priority = review_priority(&event);
        self.entries.push((priority, event));
        // Sort highest-priority first; break ties on the rollout id so identical
        // inputs always produce the same ordering.
        self.entries.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| a.1.rollout_id.as_uuid().cmp(&b.1.rollout_id.as_uuid()))
        });
        // Trim to capacity: anything past the top-K is evicted (it was the least
        // worth a human's time).
        self.entries.truncate(self.capacity);
    }

    /// The events currently held, highest-priority first.
    #[must_use]
    pub fn ranked(&self) -> Vec<&TriggerEvent> {
        self.entries.iter().map(|(_, e)| e).collect()
    }

    /// How many events the queue holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the queue holds no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Accept a human's label for `event`, producing the ground-truth
    /// [`LabelRecord`]. The human's verdict overrides the detector's `class_hint`;
    /// both are recorded so the override is auditable, and this record is the truth
    /// the downstream confidence calibration is fit against.
    #[must_use]
    pub fn accept_label(event: &TriggerEvent, verdict: HumanVerdict) -> LabelRecord {
        LabelRecord {
            rollout_id: event.rollout_id,
            outcome_id: event.outcome_id,
            detector_id: event.detector_id,
            detector_class_hint: event.class_hint,
            verdict,
        }
    }
}

/// A human reviewer's verdict on a triggered event — the ground truth. Either the
/// event was a genuine failure of a specific [`fieldloop_types::FailureClass`], or it
/// was a success (the detector fired but nothing actually went wrong — a false
/// positive worth recording so the detector can be improved).
///
/// A closed enum so every consumer of a human label handles both the failure and the
/// success case; a success verdict is first-class, not the absence of a class, because
/// "a human confirmed this was fine" is itself valuable training signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanVerdict {
    /// The human confirmed a failure and assigned its class. This class is ground
    /// truth and overrides any detector hint.
    Failure(FailureClass),
    /// The human confirmed the event was a success / non-event (the detector fired
    /// but nothing went wrong).
    Success,
}

/// The ground-truth record produced when a human labels a triggered event. Downstream
/// calibration and the JOIN treat the human's verdict here as truth: automated
/// attribution confidences are measured against these labels, and a curator's class
/// outranks every automated guess. Keeps the detector's original hint alongside the
/// human verdict so a human overriding the hint is explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelRecord {
    /// The rollout this label applies to.
    pub rollout_id: RolloutId,
    /// The originating outcome, when the trigger had one.
    pub outcome_id: Option<OutcomeId>,
    /// The detector whose trigger surfaced this event for review.
    pub detector_id: &'static str,
    /// What the detector *hinted* the class was — kept for the record so the human's
    /// override of it is visible, never silently discarded.
    pub detector_class_hint: FailureClass,
    /// The human's verdict — the ground truth that overrides the hint.
    pub verdict: HumanVerdict,
}

impl LabelRecord {
    /// True iff the human's verdict disagreed with the detector's class hint — either
    /// a different failure class, or a success where the detector hinted a failure. A
    /// quick check that proves the human label can and does override the mere hint.
    #[must_use]
    pub fn human_overrode_hint(&self) -> bool {
        match self.verdict {
            HumanVerdict::Failure(class) => class != self.detector_class_hint,
            HumanVerdict::Success => true,
        }
    }
}
