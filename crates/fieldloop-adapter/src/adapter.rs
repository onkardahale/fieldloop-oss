//! The [`EmbodimentAdapter`] trait — the full contract a new robot type implements —
//! and the small supporting value types it speaks in.
//!
//! Keeping the contract as one trait means "onboard a new embodiment" reduces to
//! "implement this trait and pass the conformance suite": the eng-team-consuming,
//! open-ended task becomes a known, finite deliverable with a mechanical
//! pass/fail check.

use std::collections::BTreeMap;

use fieldloop_config::{ActionSpace, AttributionWindow};
use fieldloop_types::{FailureClass, OutcomeEvent, OutcomeKind};
use serde::{Deserialize, Serialize};

/// A robot's raw action as it comes off that robot's own stack, before Fieldloop
/// canonicalizes it.
///
/// Modeled as a named vector of f64 channels so the adapter — which alone knows this
/// robot's joint order, units, and sign conventions — can map it into a canonical
/// form. Kept deliberately generic (channels, not fixed fields) because every
/// embodiment's raw action has a different shape, and pinning a struct here would
/// make this crate need to change for every new robot type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawAction {
    /// The raw per-channel values, keyed by this robot's own channel names.
    pub channels: BTreeMap<String, f64>,
}

impl RawAction {
    /// Construct a raw action from `(name, value)` channel pairs. A convenience for
    /// callers and fixtures so they need not build the map by hand.
    #[must_use]
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, f64)>,
        S: Into<String>,
    {
        Self {
            channels: pairs.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }
}

/// A raw action after the adapter has mapped it into Fieldloop's canonical form.
///
/// The canonical form pairs the declared [`ActionSpace`] with the ordered, named
/// channels in that space's convention. It carries the action space alongside the
/// values so a downstream consumer can never misread, say, joint velocities as
/// joint positions — the space travels with the numbers. A `BTreeMap` keeps the
/// channel order canonical (sorted), which is what makes equality — and therefore
/// the determinism check — well defined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedAction {
    /// The action space these canonical channels are expressed in.
    pub action_space: ActionSpace,
    /// The canonical per-channel values, in a fixed (sorted) channel order.
    pub channels: BTreeMap<String, f64>,
}

/// A total success decision for a rollout/episode signal.
///
/// A closed three-way enum, not a `bool`, because "we genuinely cannot tell" is a
/// distinct, first-class answer: forcing an `Unknown` signal into `false` would
/// silently fabricate failures and poison failure-rate metrics. The variant set is
/// closed so every consumer of a verdict must handle all three states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuccessVerdict {
    /// The signal indicates the task succeeded.
    Success,
    /// The signal indicates the task failed.
    Failure,
    /// The signal is insufficient to decide either way.
    Unknown,
}

/// One field in a training export's feature schema: its dtype and shape.
///
/// Modeled as a typed (dtype, shape) pair rather than a free string so a malformed
/// schema — an empty dtype or a zero-length axis — is detectable by the conformance
/// harness instead of slipping silently into a dataset that fails only at train
/// time on someone else's machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// The data type of the feature (for example `float32`, `int64`, `video`).
    pub dtype: String,
    /// The shape of the feature; an empty vec denotes a scalar. Each axis must be
    /// non-zero, since a zero-length axis is a degenerate, untrainable feature.
    pub shape: Vec<u32>,
}

impl FeatureSpec {
    /// A scalar feature of the given dtype (empty shape). A convenience so adapters
    /// can declare common scalar fields without spelling out an empty vec.
    #[must_use]
    pub fn scalar(dtype: impl Into<String>) -> Self {
        Self {
            dtype: dtype.into(),
            shape: Vec::new(),
        }
    }

    /// A feature of the given dtype and shape.
    #[must_use]
    pub fn new(dtype: impl Into<String>, shape: Vec<u32>) -> Self {
        Self {
            dtype: dtype.into(),
            shape,
        }
    }

    /// Whether this spec is well-formed: a non-empty dtype and no zero-length axis.
    /// A zero axis or empty dtype would produce a feature that cannot be allocated or
    /// trained on, so the conformance harness treats either as a schema defect.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        !self.dtype.trim().is_empty() && self.shape.iter().all(|&axis| axis > 0)
    }
}

/// The export feature schema for an embodiment's training data: a named set of
/// [`FeatureSpec`]s.
///
/// This is the schema (names, dtypes, shapes) the downstream LeRobot export must
/// conform to for this robot type — distinct from any concrete exported dataset. A
/// `BTreeMap` keys the features by name in a canonical order so two equal schemas
/// compare equal regardless of insertion order. It must be non-empty: an export with
/// no features describes no observation or action and cannot train anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeRobotFeatures {
    /// Feature name → its dtype/shape spec.
    pub features: BTreeMap<String, FeatureSpec>,
}

impl LeRobotFeatures {
    /// Build a feature schema from `(name, spec)` pairs.
    #[must_use]
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, FeatureSpec)>,
        S: Into<String>,
    {
        Self {
            features: pairs.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    /// Whether the schema has at least one feature. An empty schema is not a valid
    /// training export, so this is one of the conformance checks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// The contract a robot type ("embodiment") implements to be onboarded into
/// Fieldloop.
///
/// Implementing this one trait — and passing [`crate::check_conformance`] — is the
/// entire deliverable for a new robot type. Each method is a behavior that capture,
/// JOIN, or curation needs threaded per embodiment; gathering them here turns an
/// open-ended onboarding into a bounded, reviewable, testable surface.
///
/// Implementations must be pure and deterministic (no I/O, no clock reads, no
/// randomness), because the conformance harness validates them by calling them
/// repeatedly and comparing results, and because the live pipeline relies on the
/// same input always yielding the same answer for reproducible attribution.
pub trait EmbodimentAdapter {
    /// The embodiment name this adapter handles — the key that selects it. Must be
    /// non-empty, since an empty name cannot be matched against a rollout's declared
    /// `embodiment` field.
    fn embodiment(&self) -> &str;

    /// The control interface this robot type exposes. Drives how a raw action is
    /// interpreted and what canonical form [`Self::normalize_action`] targets.
    fn action_space(&self) -> ActionSpace;

    /// The attribution window for one outcome kind, or `None` if this embodiment does
    /// not attribute that kind at all.
    ///
    /// Returns the declared [`AttributionWindow`] (window length + clock-safety flag)
    /// so the JOIN knows how far back to look and whether a wall-clock match is
    /// permitted. `None` is a deliberate, distinct answer ("this kind is not
    /// attributed for this robot"), never a zero-length window.
    fn attribution_window(&self, kind: OutcomeKind) -> Option<AttributionWindow>;

    /// Whether attributing this outcome kind requires the rollout and outcome to
    /// share a boot and be compared on the skew-free monotonic clock.
    ///
    /// Must be `true` for tight-timing kinds such as a collision: binding a collision
    /// on a drifting wall-clock estimate would produce a false attribution, so those
    /// kinds may only bind when the two records are clock-colocated. This is a
    /// behavior the conformance harness checks against the declared window.
    fn requires_monotonic_colocation(&self, kind: OutcomeKind) -> bool;

    /// A failure-class *hint* derived from an observed outcome.
    ///
    /// This is a hint, not a verdict: it seeds triage, but a human curator or a
    /// downstream label can and does override it. It is total — it returns a valid
    /// [`FailureClass`] for every possible [`OutcomeEvent`] — so failure analysis can
    /// never hit an outcome it has no class for.
    fn classify(&self, outcome: &OutcomeEvent) -> FailureClass;

    /// A total success decision for one robot-specific scalar success signal.
    ///
    /// The adapter author chooses which robot-local telemetry to collapse into
    /// `signal`, and the caller computes that scalar before invoking this method. A
    /// common pattern is a normalized goal-distance in the closed range `[0.0, 1.0]`,
    /// where `0.0` means "at goal" and `1.0` means "maximally far"; the reference
    /// adapter uses exactly that convention. Other adapters may use a different scalar
    /// so long as the mapping is documented and deterministic.
    ///
    /// This method is the seam where robot-specific success logic plugs into the
    /// rest of the pipeline: once a caller has reduced raw telemetry to one scalar,
    /// `detect_success` turns it into the closed three-way [`SuccessVerdict`] the
    /// JOIN, curation, and evaluation flow can record without knowing the robot's
    /// private sensor shape. It must never panic and must return one of the three
    /// verdict states for every input, so an undecidable case becomes
    /// [`SuccessVerdict::Unknown`] rather than a crash or a fabricated failure.
    fn detect_success(&self, signal: f64) -> SuccessVerdict;

    /// Map this robot's raw action into Fieldloop's canonical [`NormalizedAction`].
    ///
    /// Must be deterministic: the same `raw` always yields the same normalized action,
    /// and normalizing an already-canonical action is stable (idempotent). The live
    /// pipeline and the training export both depend on this, since a non-deterministic
    /// normalization would make the same action appear as two different training
    /// samples.
    fn normalize_action(&self, raw: &RawAction) -> NormalizedAction;

    /// The (non-empty, well-formed) training-export feature schema for this
    /// embodiment. This is the schema the LeRobot export materializes against; an
    /// empty or malformed schema cannot train a policy, so the conformance harness
    /// rejects it.
    fn lerobot_features(&self) -> LeRobotFeatures;
}

/// Every [`OutcomeKind`], as a fixed slice.
///
/// Exposed so the conformance harness can exercise totality of [`EmbodimentAdapter::classify`]
/// and [`EmbodimentAdapter::requires_monotonic_colocation`] over *all* kinds. Written as an
/// explicit array with a compile-time exhaustiveness guard below, so adding a new
/// `OutcomeKind` to the schema forces this list — and therefore the harness's
/// coverage — to be updated rather than silently leaving a kind unchecked.
pub const ALL_OUTCOME_KINDS: [OutcomeKind; 5] = [
    OutcomeKind::TeleopTakeover,
    OutcomeKind::EStop,
    OutcomeKind::Collision,
    OutcomeKind::DownstreamFailure,
    OutcomeKind::Heartbeat,
];

/// Compile-time guard that [`ALL_OUTCOME_KINDS`] lists every variant. If a new
/// `OutcomeKind` is added to the schema, this match stops compiling until the new
/// variant is added to the list above, so the harness can never silently miss a kind.
const _: fn(OutcomeKind) = |kind| match kind {
    OutcomeKind::TeleopTakeover
    | OutcomeKind::EStop
    | OutcomeKind::Collision
    | OutcomeKind::DownstreamFailure
    | OutcomeKind::Heartbeat => {}
};
