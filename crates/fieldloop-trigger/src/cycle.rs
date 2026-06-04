//! The two-flag cycle-break: how the subsystem decides whether a triggered event may
//! become training data for the detector model — without the detector learning to
//! reproduce its own triggers.
//!
//! ## The degenerate loop this prevents
//! A detector flags events; humans label some of them; those labels train the *next*
//! detector. If the data a detector trains on is exactly the set its *own* generation
//! flagged, the model learns nothing except "reproduce my previous triggers". It gets
//! more confident on the cases it already fires on and blind to everything it has
//! never flagged — a self-confirming loop where the detector trains on its own output
//! and the dataset stops representing the real world.
//!
//! ## Two orthogonal flags + a generation counter
//! Each record carries two flags that mean genuinely different things, plus the
//! generation of the detector that touched it:
//! * `seed_detection` — did a detector fire on this? (i.e. it entered the pipeline
//!   *because* a detector flagged it). This is about *how it got here*.
//! * `training_eligible` — may this become training data for the detector model? This
//!   is about *what it may be used for*.
//!
//! Keeping them separate matters: an event can be detector-seeded yet still
//! train-eligible *for a different generation*, and an event can be train-eligible
//! while never having been detector-seeded (a human-found or randomly-sampled one).
//! Collapsing the two into a single boolean would lose exactly that distinction and
//! make the cycle-break impossible to state.
//!
//! ## The rule
//! [`training_eligible_for`] decides eligibility from `(source, target_generation)`:
//! a detector-seeded event produced by generation `g` is **not** eligible to train
//! generation `g` (that is the loop); it *is* eligible to train any *other*
//! generation, and an independently-sourced event (human or random sample) is always
//! eligible. See [`TriggerRecord::is_training_eligible_for`].

use crate::detector::{TriggerEvent, TriggerSource};

/// A trigger event enriched with its two cycle-break flags — the form that flows into
/// curation/training. `seed_detection` and `training_eligible` are deliberately two
/// fields, not one, because they answer two different questions (how it arrived vs.
/// what it may be used for).
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerRecord {
    /// The underlying trigger (ids, tier, source, hint, priority).
    pub event: TriggerEvent,
    /// True iff a detector fired on this record — i.e. it entered the pipeline
    /// because a detector flagged it. Derived from the event's source: detector
    /// sources set it, human/random sources do not.
    pub seed_detection: bool,
}

impl TriggerRecord {
    /// Wrap a [`TriggerEvent`], deriving `seed_detection` from its source. A
    /// detector-sourced event is by definition detector-seeded; a human- or
    /// random-sourced one is not.
    #[must_use]
    pub fn new(event: TriggerEvent) -> Self {
        let seed_detection = matches!(event.source, TriggerSource::Detector { .. });
        Self {
            event,
            seed_detection,
        }
    }

    /// The detector generation that seeded this record, or `None` if it was not
    /// detector-seeded. Used by the eligibility rule to compare against the
    /// generation being trained.
    #[must_use]
    pub fn seed_generation(&self) -> Option<u32> {
        match self.event.source {
            TriggerSource::Detector {
                detector_generation,
            } => Some(detector_generation),
            TriggerSource::Human | TriggerSource::RandomSample => None,
        }
    }

    /// Whether this record may be used to train detector generation
    /// `target_generation`. See [`training_eligible_for`] for the rule.
    #[must_use]
    pub fn is_training_eligible_for(&self, target_generation: u32) -> bool {
        training_eligible_for(self.event.source, target_generation)
    }
}

/// The cycle-break eligibility rule, stated once so every consumer applies it
/// identically.
///
/// Returns true iff a record from `source` may become training data for detector
/// generation `target_generation`:
/// * **Detector-seeded at the same generation** → `false`. Training generation `g` on
///   the very events generation `g` flagged is the self-confirming loop; the model
///   would just learn to reproduce its own triggers, so this is barred.
/// * **Detector-seeded at a *different* generation** → `true`. An event flagged by an
///   older (or newer) detector is independent evidence for the generation being
///   trained — it did not produce these triggers — so it is safe and valuable to use.
/// * **Human- or random-sourced** → `true`. These never came from the detector at
///   all, so they carry no self-confirmation risk for any generation; they are the
///   detector-independent ground truth the loop is broken with.
#[must_use]
pub fn training_eligible_for(source: TriggerSource, target_generation: u32) -> bool {
    match source {
        TriggerSource::Detector {
            detector_generation,
        } => detector_generation != target_generation,
        TriggerSource::Human | TriggerSource::RandomSample => true,
    }
}
