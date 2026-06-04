//! LeRobot-style export: turn a [`ResolvedSlice`] into a
//! [`LeRobotDatasetManifest`] — the export *spec* a GR00T/openpi-style trainer
//! consumes (episodes -> frames, each frame pointing at a rollout's observation/action
//! payload and carrying the label/reward from its authoritative feedback).
//!
//! ## The `PayloadFetcher` seam
//! The manifest references payloads by pointer ([`fieldloop_types::PayloadRef`]) and a
//! resolved object key; it never carries bytes. Resolving a pointer to a concrete,
//! verifiable object location (and, later, decoding the MCAP bytes) is real,
//! sandboxed I/O — so it sits behind the [`PayloadFetcher`] trait. Tests use an
//! in-memory/stub fetcher, so the whole export is exercised without touching object
//! storage or an MCAP decoder. The real decode-of-bytes impl is a later, feature-gated
//! concern that slots in behind the same trait.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use fieldloop_types::{
    EpisodeId, Feedback, FeedbackId, FeedbackValue, PayloadRef, Provenance, Rollout, RolloutId,
};

use crate::slice::ResolvedSlice;

/// Resolves a payload pointer to a concrete, fetchable export reference (and is where
/// real, sandboxed MCAP-decode-of-bytes would later happen).
///
/// A trait so the pure export logic never touches object storage: a test supplies an
/// in-memory fetcher, production supplies one that reaches the customer's bucket. The
/// fetcher only resolves/locates — it does not return bytes here, keeping the manifest
/// a pure spec of *what to load*, decoded by the trainer in its own sandbox.
pub trait PayloadFetcher {
    /// Resolve a rollout's observation/action pointer to the export reference the
    /// trainer will load. `None` means the pointer references nothing (e.g. an empty
    /// action ref) and the frame should carry no payload on that channel.
    fn resolve(&self, payload: &PayloadRef) -> Option<ResolvedPayload>;
}

/// A payload pointer resolved to what the trainer needs to load it: the object key, an
/// optional `[start, end)` range, and the content digest (so the trainer can verify
/// integrity before decoding in its sandbox).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPayload {
    /// The resolved object key in the customer's store.
    pub object_key: String,
    /// Optional `[start, end)` range within the object (e.g. the MCAP message range).
    pub range: Option<[u64; 2]>,
    /// SHA-256 of the referenced bytes, when known — the trainer verifies it before
    /// decoding, so a corrupted or swapped object is caught before it trains.
    pub content_sha256: Option<String>,
}

/// The label/reward attached to a frame, drawn from the AUTHORITATIVE feedback the
/// slice pinned — never invented here.
///
/// A sum type mirroring [`FeedbackValue`] so the training signal's shape always
/// matches its meaning: a boolean success, a float reward/return, or a categorical
/// failure class. A demonstration-ref correction is carried as its object key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "label_type", rename_all = "snake_case")]
pub enum FrameLabel {
    /// Boolean success/failure signal.
    Boolean { value: bool },
    /// Scalar reward / return.
    Reward { value: f32 },
    /// Categorical failure class (rendered to its stable snake_case tag).
    FailureClass { class: String },
    /// A pointer to a corrected demonstration trajectory.
    DemonstrationRef { object_key: String },
}

impl FrameLabel {
    /// Derive the frame label from a feedback's typed value, so the training signal is
    /// exactly the authoritative binding's value and nothing the exporter made up.
    fn from_value(value: &FeedbackValue) -> Self {
        match value {
            FeedbackValue::Boolean { value } => FrameLabel::Boolean { value: *value },
            FeedbackValue::Float { value } => FrameLabel::Reward { value: *value },
            FeedbackValue::FailureClass { class } => FrameLabel::FailureClass {
                // Serialize the closed enum to its documented snake_case tag so the
                // trainer reads a stable string, not a Rust Debug rendering.
                class: serde_json::to_value(class)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_else(|| format!("{class:?}")),
            },
            FeedbackValue::DemonstrationRef { object_key, .. } => FrameLabel::DemonstrationRef {
                object_key: object_key.clone(),
            },
        }
    }
}

/// One frame: a single timestep's observation + action payload references plus the
/// label/reward, the unit a per-step trainer iterates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeRobotFrame {
    /// The rollout step this frame came from (provenance back to the source row).
    pub rollout_id: RolloutId,
    /// Step ordinal within the episode, so frames order deterministically in a
    /// trajectory regardless of how the manifest was assembled.
    pub step_index: u32,
    /// Resolved observation payload, or `None` if the rollout referenced no
    /// observation bytes.
    pub observation: Option<ResolvedPayload>,
    /// Resolved action payload, or `None` if the rollout referenced no action bytes.
    pub action: Option<ResolvedPayload>,
    /// The authoritative feedback id this label came from (provenance).
    pub feedback_id: FeedbackId,
    /// The training signal, taken verbatim from the authoritative feedback's value.
    pub label: FrameLabel,
    /// Whether the source rollout was real field evidence or synthetic (sim / Cosmos
    /// augmentation). Carried onto every frame so a trainer can down-weight or filter
    /// synthetic frames and they are never silently consumed as real — defaults to
    /// `Real` for a frame built from a manifest that predates this field.
    #[serde(default)]
    pub provenance: Provenance,
}

/// One episode: an ordered list of frames forming a trajectory — the unit
/// GR00T/openpi-style training consumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeRobotEpisode {
    /// The trajectory id these frames share.
    pub episode_id: EpisodeId,
    /// The frames, ordered by `step_index` so playback/training sees the trajectory in
    /// time order.
    pub frames: Vec<LeRobotFrame>,
}

/// The full export manifest: the episodes (each a sequence of frames) plus the content
/// hash of the slice it was built from, so an exported dataset is traceable to its
/// pinned, reproducible source slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeRobotDatasetManifest {
    /// The content hash of the source [`ResolvedSlice`] — the link from this export
    /// back to the exact pinned dataset, so the export inherits reproducibility.
    pub source_content_hash: String,
    /// The episodes, sorted by episode id for a deterministic manifest.
    pub episodes: Vec<LeRobotEpisode>,
}

impl LeRobotDatasetManifest {
    /// Total frame count across all episodes — the per-step size of the export.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.episodes.iter().map(|e| e.frames.len()).sum()
    }
}

/// Build the LeRobot export manifest from a resolved slice.
///
/// Needs the source rollouts and feedback to materialize each frame: the slice pins
/// *ids*, and the manifest needs the rollout's payload pointers and step index plus
/// the feedback's value. Looking them up by the pinned id (not re-querying) is exactly
/// what makes the export reproducible — a row whose id is pinned but whose content was
/// later changed still resolves through the pinned id.
///
/// Each pinned item becomes a frame: its observation/action pointers are resolved via
/// the [`PayloadFetcher`] (the sandboxed-I/O seam), and its label/reward is taken from
/// the pinned authoritative feedback. Frames are grouped into episodes by `episode_id`
/// and ordered by `step_index`. A pinned item whose rollout or feedback is not present
/// in the supplied rows is skipped (it cannot be materialized), keeping the export a
/// faithful function of what was actually provided.
#[must_use]
pub fn to_lerobot_manifest(
    slice: &ResolvedSlice,
    rollouts: &[Rollout],
    feedback: &[Feedback],
    fetcher: &dyn PayloadFetcher,
) -> LeRobotDatasetManifest {
    // Index the supplied rows by id for O(1) lookup of each pinned item.
    let rollout_by_id: BTreeMap<RolloutId, &Rollout> = rollouts.iter().map(|r| (r.id, r)).collect();
    let feedback_by_id: BTreeMap<FeedbackId, &Feedback> =
        feedback.iter().map(|f| (f.id, f)).collect();

    // Group frames by episode as we go, so the output is episode-structured.
    let mut by_episode: BTreeMap<EpisodeId, Vec<LeRobotFrame>> = BTreeMap::new();

    for item in &slice.items {
        let (Some(rollout), Some(fb)) = (
            rollout_by_id.get(&item.rollout_id).copied(),
            feedback_by_id.get(&item.feedback_id).copied(),
        ) else {
            // The pinned id was not supplied: cannot materialize this frame. (A real
            // rebuild would fetch it from the pinned-id store; here we only build from
            // what the caller gave us.)
            continue;
        };

        let observation = fetcher.resolve(&rollout.observation_ref);
        let action = fetcher.resolve(&rollout.action_ref);

        by_episode
            .entry(item.episode_id)
            .or_default()
            .push(LeRobotFrame {
                rollout_id: rollout.id,
                step_index: rollout.step_index,
                observation,
                action,
                feedback_id: fb.id,
                label: FrameLabel::from_value(&fb.value),
                // Carry the pinned item's provenance so the export stays separable: a
                // synthetic frame is visibly synthetic to the trainer, never silently real.
                provenance: item.provenance.clone(),
            });
    }

    // BTreeMap iterates episodes in sorted id order; within each, order frames by step
    // so a trajectory is in time order.
    let episodes = by_episode
        .into_iter()
        .map(|(episode_id, mut frames)| {
            frames.sort_by_key(|f| f.step_index);
            LeRobotEpisode { episode_id, frames }
        })
        .collect();

    LeRobotDatasetManifest {
        source_content_hash: slice.content_hash.clone(),
        episodes,
    }
}
