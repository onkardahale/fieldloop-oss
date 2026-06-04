//! # `fieldloop-trigger` — failure-trigger, labeling, selective-upload
//!
//! Decides which deployed-policy rollouts and their delayed outcomes are worth pulling the heavy
//! sensor payload for (the full stream is terabytes/day), tags them so a detector can't train on its
//! own output, and prioritizes the scarce human-review budget.
//!
//! Pure and deterministic: logic only, no I/O — given landed rollouts and outcomes it returns what
//! to upload, what was dropped, what to send a human, and how each event is tagged.
//!
//! The pieces:
//! - **Detectors** ([`detector`]) — a [`Detector`] trait, tiered by cost/location: a near-free
//!   reflex ([`ReflexDetector`]), a cheap on-robot pre-filter ([`LowConfidenceDetector`]), and a
//!   stub for a heavier learned model ([`LearnedDetector`]). Each emits a [`TriggerEvent`] carrying a
//!   [`fieldloop_types::FailureClass`] *hint* — never a verdict.
//! - **Selective-upload broker** ([`broker`]) — turns a window's triggers into a budgeted set of
//!   [`SelectiveUploadRequest`]s, keeping the highest-priority events (and always the safety events,
//!   even past budget) and surfacing every drop.
//! - **Two-flag cycle-break** ([`cycle`]) — `seed_detection` + `training_eligible` + a generation
//!   counter keep a detector from training on the very events it flagged.
//! - **Human review** ([`review`]) — a bounded queue ranked by active-learning priority; a human's
//!   label becomes a ground-truth [`LabelRecord`] that overrides the hint and that confidence
//!   calibration is fit against.
//!
//! A detector only HINTS: the real failure class comes from the observed outcome or a human, never
//! from a heuristic's guess.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod broker;
pub mod cycle;
pub mod detector;
pub mod review;

pub use broker::{
    DroppedEvent, SelectiveUploadBroker, SelectiveUploadRequest, UploadBudget, UploadDecision,
};
pub use cycle::{TriggerRecord, training_eligible_for};
pub use detector::{
    Detector, DetectorTier, LOW_CONFIDENCE_THRESHOLD, LearnedDetector, LearnedDetectorAdapter,
    LowConfidenceDetector, POLICY_CONFIDENCE_TAG, ReflexDetector, TriggerEvent, TriggerSource,
};
pub use review::{HumanVerdict, LabelRecord, ReviewQueue, review_priority};

// ---------------------------------------------------------------------------
// Self-contained verification. These tests exercise only this crate's public API
// plus the frozen schema types, with fixed inputs — so the whole subsystem is a
// closed loop: detection, budgeting, the cycle-break, active-learning ranking, and
// the human-label path all run deterministically with no DB, network, or harness.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, FailureClass, MonoClock, OutcomeEvent, OutcomeKind,
        PayloadRef, PolicyVersion, RobotId, RobotIdentity, Rollout, RolloutId, TenantId,
    };

    fn robot() -> RobotIdentity {
        RobotIdentity::new(TenantId::new("acme"), RobotId::new("robot-7"))
    }

    /// Build a plain, unremarkable rollout. `confidence` is written into the open
    /// `tags` map under the key the pre-filter reads; `None` means no confidence tag.
    fn rollout_with_confidence(confidence: Option<f32>) -> Rollout {
        let mut tags = BTreeMap::new();
        if let Some(c) = confidence {
            tags.insert(POLICY_CONFIDENCE_TAG.to_string(), c.to_string());
        }
        let mut r = Rollout::new(
            robot(),
            EpisodeId::new(),
            0,
            MonoClock::new(BootId::new(), 1_000_000, 1_700_000_000_000_000_000),
            PolicyVersion::new("pick@v1.2.0+abc"),
            "sha256:dead".to_string(),
            "ur5e".to_string(),
            "bin_pick".to_string(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            850,
        );
        r.tags = tags;
        r
    }

    fn outcome(kind: OutcomeKind) -> OutcomeEvent {
        OutcomeEvent::new(
            robot(),
            MonoClock::new(BootId::new(), 2_000_000, 1_700_000_000_000_000_000),
            kind,
            BoundedBlob::empty(),
        )
    }

    /// A reflex detector fires a Tier-A event on each unambiguous "it went wrong"
    /// outcome (e-stop / collision / takeover / downstream failure), with a
    /// failure-class *hint* — and stays silent on a heartbeat (coverage, not failure).
    #[test]
    fn reflex_detector_fires_tier_a_on_bad_outcomes() {
        let det = ReflexDetector;
        let r = rollout_with_confidence(None);

        for (kind, expect_hint) in [
            (OutcomeKind::EStop, FailureClass::Operator),
            (OutcomeKind::TeleopTakeover, FailureClass::Operator),
            (OutcomeKind::Collision, FailureClass::Manipulation),
            (OutcomeKind::DownstreamFailure, FailureClass::Environment),
        ] {
            let o = outcome(kind);
            let ev = det
                .inspect(&r, Some(&o))
                .expect("a bad outcome must produce a reflex trigger");
            assert_eq!(ev.tier, DetectorTier::ReflexA);
            assert_eq!(ev.class_hint, expect_hint, "the hint maps from the kind");
            assert_eq!(ev.outcome_id, Some(o.id));
        }

        // A heartbeat is coverage, not a failure — no reflex trigger.
        let hb = outcome(OutcomeKind::Heartbeat);
        assert!(det.inspect(&r, Some(&hb)).is_none());
        // And no outcome at all means nothing to react to.
        assert!(det.inspect(&r, None).is_none());
    }

    /// The on-robot pre-filter fires a Tier-A′ event on a low-confidence rollout, and
    /// stays silent on a confident one (and when there is no confidence signal).
    #[test]
    fn low_confidence_detector_fires_tier_a_prime() {
        let det = LowConfidenceDetector;

        // Below threshold → fires, Tier-A′, perception hint, no outcome yet.
        let risky = rollout_with_confidence(Some(0.10));
        let ev = det
            .inspect(&risky, None)
            .expect("a low-confidence rollout must trigger the pre-filter");
        assert_eq!(ev.tier, DetectorTier::PreFilterAPrime);
        assert_eq!(ev.class_hint, FailureClass::Perception);
        assert_eq!(ev.outcome_id, None);
        assert!(ev.raw_priority > 0.0);

        // At/above threshold → no trigger.
        let confident = rollout_with_confidence(Some(0.95));
        assert!(det.inspect(&confident, None).is_none());

        // No confidence tag → no signal → no trigger (does not fire on noise).
        let untagged = rollout_with_confidence(None);
        assert!(det.inspect(&untagged, None).is_none());
    }

    /// The learned-detector stub fires nothing by default (no fabricated score),
    /// while an overriding implementation drives a Tier-B trigger through the adapter.
    #[test]
    fn learned_detector_stub_is_silent_until_overridden() {
        struct Stub;
        impl LearnedDetector for Stub {
            fn id(&self) -> &'static str {
                "learned.stub"
            }
        }
        let adapter = LearnedDetectorAdapter {
            inner: Stub,
            detector_generation: 5,
        };
        let r = rollout_with_confidence(None);
        // Default score 0.0 < threshold 0.5 → silent.
        assert!(adapter.inspect(&r, None).is_none());

        struct Anomaly;
        impl LearnedDetector for Anomaly {
            fn id(&self) -> &'static str {
                "learned.anomaly"
            }
            fn score(&self, _r: &Rollout, _o: Option<&OutcomeEvent>) -> f32 {
                0.9
            }
        }
        let adapter = LearnedDetectorAdapter {
            inner: Anomaly,
            detector_generation: 5,
        };
        let ev = adapter
            .inspect(&r, None)
            .expect("a high anomaly score must trigger Tier-B");
        assert_eq!(ev.tier, DetectorTier::LearnedB);
        assert_eq!(
            ev.source,
            TriggerSource::Detector {
                detector_generation: 5
            }
        );
    }

    /// A normal rollout with no bad signal produces no event from any detector.
    #[test]
    fn normal_rollout_triggers_nothing() {
        let r = rollout_with_confidence(Some(0.99));
        assert!(ReflexDetector.inspect(&r, None).is_none());
        assert!(LowConfidenceDetector.inspect(&r, None).is_none());
    }

    /// A budgeted (non-safety) event used to fill the broker.
    fn budgeted_event(priority: f32) -> TriggerEvent {
        TriggerEvent {
            rollout_id: RolloutId::new(),
            outcome_id: None,
            detector_id: "prefilter.low_confidence",
            tier: DetectorTier::PreFilterAPrime,
            source: TriggerSource::Detector {
                detector_generation: 0,
            },
            class_hint: FailureClass::Perception,
            raw_priority: priority,
        }
    }

    fn safety_event() -> TriggerEvent {
        TriggerEvent {
            rollout_id: RolloutId::new(),
            outcome_id: None,
            detector_id: "reflex.outcome",
            tier: DetectorTier::ReflexA,
            source: TriggerSource::Detector {
                detector_generation: 0,
            },
            class_hint: FailureClass::Manipulation,
            raw_priority: 1.0,
        }
    }

    /// Given more trigger events than the budget allows, the broker requests only the
    /// top-priority N, records the rest as dropped-and-counted, and ALWAYS includes a
    /// Tier-A safety event even past the budget.
    #[test]
    fn broker_keeps_top_n_and_always_keeps_safety() {
        // Budget of 2 droppable requests.
        let broker = SelectiveUploadBroker::new(UploadBudget::by_count(2), 1);

        let high = budgeted_event(0.9);
        let mid = budgeted_event(0.6);
        let low = budgeted_event(0.2);
        let lowest = budgeted_event(0.1);
        let safety = safety_event();

        // Deliberately out of priority order, and the safety event last.
        let events = vec![
            low.clone(),
            high.clone(),
            lowest.clone(),
            mid.clone(),
            safety.clone(),
        ];
        let decision = broker.select(&events);

        // The safety event is always requested; budget admits the top 2 droppable.
        let requested: Vec<RolloutId> = decision.requests.iter().map(|r| r.rollout_id).collect();
        assert!(
            requested.contains(&safety.rollout_id),
            "the Tier-A safety event must be requested even past budget"
        );
        assert!(requested.contains(&high.rollout_id), "highest is kept");
        assert!(
            requested.contains(&mid.rollout_id),
            "second-highest is kept"
        );

        // The two lowest are dropped-and-counted, never silent.
        assert_eq!(decision.dropped_count(), 2);
        let dropped: Vec<RolloutId> = decision.dropped.iter().map(|d| d.rollout_id).collect();
        assert!(dropped.contains(&low.rollout_id));
        assert!(dropped.contains(&lowest.rollout_id));

        // Safety is never in the dropped set.
        assert!(
            decision
                .dropped
                .iter()
                .all(|d| d.tier != DetectorTier::ReflexA)
        );

        // Total requested = 2 budgeted + 1 safety.
        assert_eq!(decision.requested_count(), 3);
    }

    /// The byte ceiling also binds: a small byte budget admits fewer than the count
    /// limit would, and the safety event still bypasses it.
    #[test]
    fn broker_byte_budget_binds_and_safety_bypasses() {
        // Count would allow 10, but bytes allow only 1 request of 100 bytes each.
        let broker = SelectiveUploadBroker::new(
            UploadBudget {
                max_requests: 10,
                max_bytes: 100,
            },
            100,
        );
        let events = vec![budgeted_event(0.9), budgeted_event(0.8), safety_event()];
        let decision = broker.select(&events);
        // 1 budgeted (byte-limited) + 1 safety (bypasses bytes).
        assert_eq!(decision.requested_count(), 2);
        assert_eq!(decision.dropped_count(), 1);
    }

    /// The two-flag cycle-break: a detector-sourced event at generation `g` is NOT
    /// training-eligible for generation `g` (it would teach the model to reproduce its
    /// own triggers), while an independently-sourced (human) event IS eligible — and a
    /// detector event from a *different* generation is eligible too.
    #[test]
    fn cycle_break_bars_same_generation_only() {
        let generation = 7;

        // Detector-sourced at generation 7.
        let det_event = TriggerEvent {
            source: TriggerSource::Detector {
                detector_generation: generation,
            },
            ..budgeted_event(0.5)
        };
        let det_record = TriggerRecord::new(det_event);
        assert!(det_record.seed_detection, "a detector fired → seeded");
        assert!(
            !det_record.is_training_eligible_for(generation),
            "generation g must NOT train on its own generation-g triggers"
        );
        assert!(
            det_record.is_training_eligible_for(generation + 1),
            "a generation-g event is independent evidence for a different generation"
        );

        // Human-sourced: independent of any detector, always eligible.
        let human_event = TriggerEvent {
            source: TriggerSource::Human,
            ..budgeted_event(0.5)
        };
        let human_record = TriggerRecord::new(human_event);
        assert!(
            !human_record.seed_detection,
            "a human-found event is not detector-seeded"
        );
        assert!(
            human_record.is_training_eligible_for(generation),
            "an independently-sourced event is always training-eligible"
        );
    }

    /// Active-learning: a bounded queue keeps the top-K events by the review priority
    /// function, ordered highest-first, regardless of insertion order.
    #[test]
    fn review_queue_keeps_top_k_by_priority() {
        // raw_priority 0.5 is maximally uncertain → highest review priority;
        // 0.0 and 1.0 are certain → lowest. Use non-safety so value weight is uniform.
        let coinflip = budgeted_event(0.5); // uncertainty 1.0
        let leaning = budgeted_event(0.7); // uncertainty 0.6
        let certain = budgeted_event(0.98); // uncertainty ~0.04

        let mut q = ReviewQueue::new(2);
        // Insert worst-first to prove ordering is by priority, not insertion.
        q.offer(certain.clone());
        q.offer(coinflip.clone());
        q.offer(leaning.clone());

        assert_eq!(q.len(), 2, "the queue is bounded to K=2");
        let ranked = q.ranked();
        assert_eq!(
            ranked[0].rollout_id, coinflip.rollout_id,
            "the most uncertain event ranks first"
        );
        assert_eq!(
            ranked[1].rollout_id, leaning.rollout_id,
            "the next-most-uncertain ranks second"
        );
        // The near-certain event was evicted (least worth a human's time).
        assert!(ranked.iter().all(|e| e.rollout_id != certain.rollout_id));
    }

    /// A safety-tier event is up-weighted in the review priority: at equal
    /// uncertainty, the safety event outranks the routine one.
    #[test]
    fn review_priority_up_weights_safety() {
        let mut safety = safety_event();
        safety.raw_priority = 0.5; // same uncertainty as the routine coin-flip
        let routine = budgeted_event(0.5);
        assert!(
            review_priority(&safety) > review_priority(&routine),
            "a safety event is worth more to label at equal uncertainty"
        );
    }

    /// Labeling: accepting a human label yields a LabelRecord carrying the human's
    /// FailureClass, distinct from the detector's hint — proving the human overrides
    /// the hint, and giving the ground truth calibration is fit against.
    #[test]
    fn human_label_overrides_detector_hint() {
        // The pre-filter hinted Perception; the human determines it was actually
        // Planning.
        let event = budgeted_event(0.5);
        assert_eq!(event.class_hint, FailureClass::Perception);

        let label =
            ReviewQueue::accept_label(&event, HumanVerdict::Failure(FailureClass::Planning));
        assert_eq!(label.detector_class_hint, FailureClass::Perception);
        assert_eq!(label.verdict, HumanVerdict::Failure(FailureClass::Planning));
        assert!(
            label.human_overrode_hint(),
            "the human's class differs from the detector's hint"
        );

        // A success verdict (detector fired, but nothing went wrong) also overrides.
        let fp = ReviewQueue::accept_label(&event, HumanVerdict::Success);
        assert!(fp.human_overrode_hint());
    }
}
