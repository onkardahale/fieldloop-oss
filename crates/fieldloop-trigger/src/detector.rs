//! Detectors: the rule-based logic that decides a rollout/outcome is "interesting"
//! enough to be worth pulling its heavy sensor payload and possibly a human's time.
//!
//! Fieldloop cannot ship every robot's full sensor stream to the cloud — that is
//! terabytes a day. So something on or near the robot must cheaply decide *which*
//! events deserve the expensive payload pull. A [`Detector`] inspects a
//! [`fieldloop_types::Rollout`] (and any [`fieldloop_types::OutcomeEvent`] already
//! attached to it) and may emit a [`TriggerEvent`] saying "this one looks worth
//! keeping".
//!
//! ## Detectors emit a HINT, never a verdict
//! A detector firing does NOT mean "this is a perception failure". It means "a cheap
//! rule matched a signal that *often* co-occurs with that failure class". The true
//! class is established later by the observed outcome or a human label. So every
//! [`TriggerEvent`] carries a [`fieldloop_types::FailureClass`] *hint* with no claim
//! of certainty — encoded as `class_hint`, not `class` — to keep a guess from being
//! mistaken downstream for ground truth.
//!
//! ## Tiers reflect where a detector runs and what it costs
//! Detectors are grouped by cost so the cheap, universal ones can run on every robot
//! while the expensive learned ones run plane-side on a sampled subset:
//! * [`DetectorTier::ReflexA`] — universal, near-free reflexes on unambiguous "it
//!   already went wrong" signals (an e-stop, a collision, a teleop takeover, a
//!   downstream failure). These are robot-agnostic safety signals.
//! * [`DetectorTier::PreFilterAPrime`] — a cheap on-robot pre-filter on "this looks
//!   risky" signals (low policy confidence, a timeout, a retry) that fire *before* a
//!   failure is confirmed, so a risky-but-not-yet-failed moment can still be captured.
//! * [`DetectorTier::LearnedB`] — heavier or learned detection (an anomaly model),
//!   behind the [`LearnedDetector`] trait with a stub default, so the framework can
//!   host a model later without building ML now.

use fieldloop_types::{FailureClass, OutcomeEvent, OutcomeId, OutcomeKind, Rollout, RolloutId};

/// Which tier a detector belongs to — its cost class and where it is expected to
/// run. Carried on every [`TriggerEvent`] so the upload broker can treat the cheap,
/// always-on safety reflexes (`ReflexA`) differently from the sampled heavier tiers.
///
/// A closed enum so adding a tier forces every consumer (the broker's
/// safety-always-keep rule, in particular) to decide how to treat it, rather than a
/// new tier silently slipping past the budget logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectorTier {
    /// Tier-A reflex: universal, near-free, fires on an unambiguous "something went
    /// wrong" signal. Treated as a safety event the upload broker must never drop.
    ReflexA,
    /// Tier-A′ on-robot cheap pre-filter: fires on a "this looks risky" signal before
    /// a failure is confirmed (low confidence, timeout, retry). Subject to the budget.
    PreFilterAPrime,
    /// Tier-B heavier/learned detection (e.g. an anomaly model). Subject to the
    /// budget; runs plane-side on a sampled subset, not on every robot.
    LearnedB,
}

impl DetectorTier {
    /// True iff a trigger from this tier is a hard-safety signal that the upload
    /// broker must always include, even past its budget. Only the universal Tier-A
    /// reflexes qualify: dropping the payload for a collision or an e-stop because a
    /// budget filled up would lose exactly the evidence a safety review needs, which
    /// is never an acceptable trade. The risky-looking (`PreFilterAPrime`) and learned
    /// (`LearnedB`) tiers are guesses about *unconfirmed* events and so are droppable.
    #[must_use]
    pub fn is_safety_critical(self) -> bool {
        matches!(self, DetectorTier::ReflexA)
    }
}

/// Where the record that became a [`TriggerEvent`] entered the pipeline. This is the
/// provenance the cycle-break rule reads: a detector-sourced event must be barred
/// from training the same detector generation, while an independently-sourced event
/// (a human, a held-out random sample) is safe to train on.
///
/// A closed enum so the eligibility rule is total — a new source forces an explicit
/// decision about whether it may become training data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerSource {
    /// A detector fired on this record (it entered the pipeline because a rule
    /// flagged it). Carries the generation of the detector that fired, because the
    /// cycle-break rule is per-generation.
    Detector { detector_generation: u32 },
    /// A human (curator/teleoperator) flagged or labeled this record directly,
    /// independent of any detector. Safe ground truth.
    Human,
    /// A held-out random/uniform sample, deliberately chosen *without* a detector so
    /// the dataset keeps a detector-independent slice. Safe ground truth.
    RandomSample,
}

/// One trigger: a detector's decision that a specific rollout (and optionally its
/// outcome) is worth keeping, plus everything downstream needs to prioritize and tag
/// it. Pure data — produced by detectors, consumed by the upload broker and the
/// review queue.
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerEvent {
    /// The rollout this fired on — the left side of the eventual binding and the
    /// thing whose payload may be pulled.
    pub rollout_id: RolloutId,
    /// The outcome that drove the trigger, when one exists. `None` for a pre-filter
    /// trigger that fired on the rollout alone (no confirmed outcome yet), modeled as
    /// an [`Option`] rather than a nil id so "fired before any outcome" is distinct
    /// from "fired on a real outcome".
    pub outcome_id: Option<OutcomeId>,
    /// A stable identifier for the detector that fired (e.g. `"reflex.estop"`), so a
    /// drop or a label can be attributed to the exact rule.
    pub detector_id: &'static str,
    /// The detector's tier (cost class / where it runs). Drives the broker's
    /// safety-always-keep rule.
    pub tier: DetectorTier,
    /// Where this record entered the pipeline. Feeds the cycle-break eligibility rule.
    pub source: TriggerSource,
    /// The failure class this signal *hints* at — NOT a verdict. The true class comes
    /// from the observed outcome or a human label; this is only the class the signal
    /// tends to co-occur with, kept separate so it is never read as ground truth.
    pub class_hint: FailureClass,
    /// A raw, mechanical priority score in `[0, 1]`: how strongly this detector
    /// believes the event is worth the payload pull. Higher = pull sooner. Used by
    /// the broker to rank events under the budget; it is a detector-asserted number,
    /// not a calibrated probability.
    pub raw_priority: f32,
}

/// A detector: inspects a rollout (and any outcome attached to it) and may emit a
/// [`TriggerEvent`].
///
/// Kept as a trait so the tiers share one uniform shape and the broker/queue can run
/// a heterogeneous set of detectors without knowing which concrete rule each is. A
/// detector is pure: same input, same output, no I/O — which is what makes the whole
/// subsystem deterministically testable.
pub trait Detector {
    /// The stable id of this detector, carried onto every event it emits so a
    /// downstream drop or label points back at the exact rule.
    fn id(&self) -> &'static str;

    /// The tier this detector runs in.
    fn tier(&self) -> DetectorTier;

    /// Inspect a rollout and an optional outcome already associated with it; return
    /// `Some(event)` iff the detector's rule matched. `None` is the common case (most
    /// rollouts are unremarkable and must not be captured), so the broker only ever
    /// considers the small flagged minority.
    fn inspect(&self, rollout: &Rollout, outcome: Option<&OutcomeEvent>) -> Option<TriggerEvent>;
}

/// Map an unambiguous outcome kind to the [`DetectorTier::ReflexA`] failure-class
/// *hint* it most often co-occurs with. This is a hint only — the human or the
/// observed terminal outcome assigns the real class — so e.g. an e-stop is hinted as
/// `Operator` (a person hit the button) without asserting the person was the root
/// cause. Returns `None` for kinds that are not a reflex signal (a heartbeat is
/// coverage, not a failure), so the reflex detector stays silent on them.
fn reflex_class_hint(kind: OutcomeKind) -> Option<FailureClass> {
    match kind {
        // A person intervened: hint operator-domain, pending the real root cause.
        OutcomeKind::TeleopTakeover | OutcomeKind::EStop => Some(FailureClass::Operator),
        // A physical contact event most often implicates manipulation/contact control.
        OutcomeKind::Collision => Some(FailureClass::Manipulation),
        // A consequence observed downstream (e.g. a jam) most often traces to the
        // environment the policy was acting in.
        OutcomeKind::DownstreamFailure => Some(FailureClass::Environment),
        // Heartbeats are a coverage signal, not a failure — never a reflex trigger.
        OutcomeKind::Heartbeat => None,
    }
}

/// A raw priority for a reflex trigger. Hard-safety events get the maximum so they
/// also sort first whenever the broker happens to rank them; the broker additionally
/// keeps them unconditionally, but a sane score keeps the ordering honest.
const REFLEX_PRIORITY: f32 = 1.0;

/// Tier-A reflex detector: fires on the unambiguous "it already went wrong" outcome
/// kinds (e-stop, collision, teleop takeover, downstream failure). Universal and
/// near-free, so it runs on every robot; its events are the ones the upload broker
/// must never drop. Stays silent on a rollout with no such outcome attached.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflexDetector;

impl Detector for ReflexDetector {
    fn id(&self) -> &'static str {
        "reflex.outcome"
    }

    fn tier(&self) -> DetectorTier {
        DetectorTier::ReflexA
    }

    fn inspect(&self, rollout: &Rollout, outcome: Option<&OutcomeEvent>) -> Option<TriggerEvent> {
        // A reflex needs a confirmed bad outcome to fire on; without one there is
        // nothing unambiguous to react to.
        let outcome = outcome?;
        let class_hint = reflex_class_hint(outcome.outcome_kind)?;
        Some(TriggerEvent {
            rollout_id: rollout.id,
            outcome_id: Some(outcome.id),
            detector_id: self.id(),
            tier: DetectorTier::ReflexA,
            // A detector firing makes this detector-sourced. The generation is carried
            // so the cycle-break rule can bar it from training that same generation.
            source: TriggerSource::Detector {
                detector_generation: 0,
            },
            class_hint,
            raw_priority: REFLEX_PRIORITY,
        })
    }
}

/// The policy-confidence threshold below which the pre-filter treats a rollout as
/// "risky-looking". A constant kept here (rather than in the shared config) to keep
/// this subsystem self-contained and its tests deterministic; it is the kind of dial
/// an operator would later move into config. The rollout reports its own confidence
/// in the `tags` map under [`POLICY_CONFIDENCE_TAG`].
pub const LOW_CONFIDENCE_THRESHOLD: f32 = 0.30;

/// The `tags` key the capture SDK writes the policy's self-reported confidence under.
/// The pre-filter reads it from the open `tags` map because confidence is an
/// embodiment-specific signal the schema does not carry as a typed column.
pub const POLICY_CONFIDENCE_TAG: &str = "policy_confidence";

/// Tier-A′ on-robot cheap pre-filter: fires when the rollout's self-reported policy
/// confidence (in its `tags`) is below [`LOW_CONFIDENCE_THRESHOLD`]. This catches a
/// "the policy was unsure, this could go wrong" moment *before* any failure is
/// confirmed, so a risky-but-not-yet-failed rollout can still be captured. Reads a
/// tag (an open-SDK field) defensively: a missing or unparseable confidence is
/// treated as "no signal" and the detector stays silent rather than firing on noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct LowConfidenceDetector;

impl Detector for LowConfidenceDetector {
    fn id(&self) -> &'static str {
        "prefilter.low_confidence"
    }

    fn tier(&self) -> DetectorTier {
        DetectorTier::PreFilterAPrime
    }

    fn inspect(&self, rollout: &Rollout, _outcome: Option<&OutcomeEvent>) -> Option<TriggerEvent> {
        // The confidence rides the open `tags` map; a missing or malformed value is
        // "no signal", so the detector stays silent rather than firing on garbage.
        let confidence: f32 = rollout.tags.get(POLICY_CONFIDENCE_TAG)?.parse().ok()?;
        if confidence >= LOW_CONFIDENCE_THRESHOLD {
            return None;
        }
        Some(TriggerEvent {
            rollout_id: rollout.id,
            // No outcome yet — this fires on the rollout alone, before any failure is
            // confirmed, so there is no originating outcome to point back to.
            outcome_id: None,
            detector_id: self.id(),
            tier: DetectorTier::PreFilterAPrime,
            source: TriggerSource::Detector {
                detector_generation: 0,
            },
            // Low policy confidence most often co-occurs with a perception problem
            // (the policy could not read the scene), but this is only a hint.
            class_hint: FailureClass::Perception,
            // The less confident the policy, the higher the priority to capture it.
            // Linearly maps confidence in `[0, threshold)` onto priority `(.., 1]`.
            raw_priority: (1.0 - confidence / LOW_CONFIDENCE_THRESHOLD).clamp(0.0, 1.0),
        })
    }
}

/// A heavier / learned detector (e.g. an anomaly model). Behind its own trait so the
/// framework supports plugging in a learned model later without building any ML now:
/// the rest of the subsystem (broker, queue, cycle-break) depends only on the
/// [`TriggerEvent`]s a learned detector emits, not on how it decides.
///
/// The default method returns no trigger, so the shipped build has an honest no-op
/// stub instead of a fabricated score. A real implementation overrides `score`.
pub trait LearnedDetector {
    /// The stable id of this learned detector.
    fn id(&self) -> &'static str;

    /// Score a rollout/outcome for anomalousness in `[0, 1]`. The default is `0.0`
    /// (no anomaly), so a build with no model attached fires nothing rather than
    /// emitting an invented score.
    fn score(&self, _rollout: &Rollout, _outcome: Option<&OutcomeEvent>) -> f32 {
        0.0
    }

    /// The score above which the learned detector emits a trigger.
    fn threshold(&self) -> f32 {
        0.5
    }
}

/// Adapts any [`LearnedDetector`] into the uniform [`Detector`] trait at
/// [`DetectorTier::LearnedB`], so a learned model drops into the same pipeline as the
/// rule-based tiers. Carries the detector generation the model belongs to, so a
/// trigger it produces is correctly barred from training that same generation.
#[derive(Debug, Clone, Copy)]
pub struct LearnedDetectorAdapter<D> {
    /// The wrapped learned detector.
    pub inner: D,
    /// The generation of the model that produced this trigger — threaded onto the
    /// event so the cycle-break rule can keep generation `g` from training on its own
    /// output.
    pub detector_generation: u32,
}

impl<D: LearnedDetector> Detector for LearnedDetectorAdapter<D> {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn tier(&self) -> DetectorTier {
        DetectorTier::LearnedB
    }

    fn inspect(&self, rollout: &Rollout, outcome: Option<&OutcomeEvent>) -> Option<TriggerEvent> {
        let score = self.inner.score(rollout, outcome);
        if score < self.inner.threshold() {
            return None;
        }
        Some(TriggerEvent {
            rollout_id: rollout.id,
            outcome_id: outcome.map(|o| o.id),
            detector_id: self.inner.id(),
            tier: DetectorTier::LearnedB,
            source: TriggerSource::Detector {
                detector_generation: self.detector_generation,
            },
            // A learned anomaly detector has no domain knowledge of the failure
            // taxonomy; it hints the catch-all `Environment` and defers the real
            // class entirely to the human label.
            class_hint: FailureClass::Environment,
            raw_priority: score.clamp(0.0, 1.0),
        })
    }
}
