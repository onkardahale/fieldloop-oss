//! `compile_slice` — the core: turn a [`SliceSpec`] + resolved rollouts + feedback
//! into a [`ResolvedSlice`], the frozen, content-hashed manifest of exactly which
//! `(rollout_id, feedback_id)` pairs matched.
//!
//! ## Why a pinned manifest, not a stored query
//! If a dataset were defined by re-running the query, a later edit, retraction, or
//! garbage-collection of the source rows would silently change what "the dataset"
//! means — a rebuild would no longer reproduce the bytes a model was trained on. So a
//! [`ResolvedSlice`] pins the concrete ids that matched at compile time. A rebuild
//! reads the pinned ids and reproduces the exact same dataset even if the source rows
//! were later changed or removed. The [`ResolvedSlice::content_hash`] over that pinned
//! set is what makes identical inputs dedup to one commit and any change produce a new
//! one.
//!
//! ## Where the confidence gate runs
//! The gate runs once, here, per candidate target, on the *authoritative* binding
//! chosen by the JOIN crate's latest-wins resolver. A retracted authoritative binding
//! or one whose `join_confidence` is below the spec's floor is never admitted to the
//! pinned set — it is recorded as a [`NeedsReview`] item instead. A confidently-wrong
//! label is worse than a missing example, so a doubtful binding must surface for human
//! review rather than leak into the training set.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use fieldloop_join::winning_binding;
use fieldloop_types::{
    EpisodeId, Feedback, FeedbackId, FeedbackTarget, FeedbackValue, Provenance, Rollout, RolloutId,
    TenantId,
};

use crate::spec::{Grain, SITE_TAG_KEY, SliceSpec};

/// One pinned training item: a rollout step paired with the authoritative feedback
/// that scored it. This is the atom of the frozen manifest — the ids, not the rows,
/// so a rebuild re-fetches by id rather than trusting a (possibly mutated) live row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinnedItem {
    /// The episode this step belongs to. Carried even in a `Rollout`-grain slice so
    /// the export can always group steps into episodes without a second lookup.
    pub episode_id: EpisodeId,
    /// The pinned rollout step.
    pub rollout_id: RolloutId,
    /// The authoritative feedback chosen for this rollout's target (manual outranks
    /// automated; otherwise newest). Its label/reward is what training consumes.
    pub feedback_id: FeedbackId,
    /// Whether the pinned rollout was real field evidence or synthetic (sim / Cosmos
    /// augmentation). Carried per item so synthetic data stays distinguishable in the
    /// frozen manifest — a consumer can separate, down-weight, or exclude it, and it can
    /// never be silently treated as real. Defaults to `Real` so an old pinned item
    /// (serialized before this field existed) reads back as real.
    #[serde(default)]
    pub provenance: Provenance,
}

/// One candidate that did NOT enter the training set, with the reason — surfaced
/// rather than silently dropped, because the operator needs to see what their
/// confidence floor and retractions excluded, not have it vanish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeedsReview {
    /// The rollout whose authoritative binding failed the gate.
    pub rollout_id: RolloutId,
    /// The authoritative feedback that failed the gate, if there was one to inspect.
    /// `None` means no surviving (non-retracted) binding existed at all.
    pub feedback_id: Option<FeedbackId>,
    /// Why it was held out of training data.
    pub reason: ReviewReason,
}

/// Why a candidate was held out of the slice and flagged for review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewReason {
    /// The authoritative binding's `join_confidence` was below the spec floor. A
    /// label we are not confident in must not train a model that acts on it.
    BelowMinConfidence,
    /// The authoritative binding was retracted (tombstoned). A withdrawn label is not
    /// training data.
    Retracted,
    /// No non-retracted binding existed for this rollout at all, so there is nothing
    /// authoritative to train on.
    NoSurvivingBinding,
}

/// The frozen, content-hashed result of compiling a [`SliceSpec`].
///
/// Holds the spec it came from (for provenance) and the pinned `(rollout, feedback)`
/// set — NOT the query. A rebuild reads `items`/`episodes` and re-fetches by id, so it
/// reproduces the identical dataset even after the source rows changed. `content_hash`
/// is computed over the pinned set so equal sets hash equal and any change differs.
// No `Eq`: `SliceSpec` carries an `f32` (`min_confidence`), so the slice is only
// `PartialEq`. Equality of two slices is compared structurally in tests via that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedSlice {
    /// The tenant this slice belongs to — the isolation key, kept on the manifest so a
    /// commit can never be mistaken for another tenant's.
    pub tenant_id: TenantId,
    /// The spec that produced this slice, retained as provenance (it is the *intent*;
    /// `items` is the frozen *result*).
    pub spec: SliceSpec,
    /// The pinned training items, in a deterministic order (sorted by rollout id) so
    /// the content hash is stable across runs regardless of input ordering.
    pub items: Vec<PinnedItem>,
    /// The distinct episodes covered by `items`, sorted — the unit an `Episode`-grain
    /// dataset trains on, and the lineage key for "which policies came from this
    /// episode".
    pub episodes: Vec<EpisodeId>,
    /// Candidates excluded by the confidence gate (low-confidence / retracted /
    /// no-binding), surfaced for human review rather than dropped.
    pub needs_review: Vec<NeedsReview>,
    /// A content hash over `(spec, items, episodes)`. Stable and order-independent of
    /// the inputs; changing any pinned id or the spec changes it.
    pub content_hash: String,
}

impl ResolvedSlice {
    /// The number of pinned training items (per-step count). Distinct from
    /// [`ResolvedSlice::episodes`]`.len()` at episode grain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True iff nothing matched the spec (after the confidence gate).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A fully recorded dataset commit: the immutable, content-addressed unit training
/// consumes and lineage points at.
///
/// `commit_id` is derived from `content_hash`, so two commits built from identical
/// `(spec + inputs)` collapse to the same id (dedup) while any changed input yields a
/// new id. That is the reproducibility/dedup guarantee made concrete.
// No `Eq` for the same reason `ResolvedSlice` has none: the embedded `SliceSpec`
// carries an `f32`. The `commit_id`/`content_hash` strings are the identity to
// compare on, and those are plain `String`s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetCommit {
    /// The commit identity — equal to `content_hash`, so identical inputs dedup to the
    /// same commit and a changed input is a genuinely new commit.
    pub commit_id: String,
    /// The hash over the pinned set the commit id is derived from.
    pub content_hash: String,
    /// The spec that selected the data (provenance).
    pub slice_spec: SliceSpec,
    /// The frozen manifest the commit pins.
    pub resolved_manifest: ResolvedSlice,
    /// Free-form lineage notes (e.g. who built it, an upstream lakeFS ref). Kept as a
    /// sorted map so it never perturbs the content hash — lineage metadata is *about*
    /// the commit, not part of its identity.
    pub lineage: BTreeMap<String, String>,
}

impl DatasetCommit {
    /// Seal a resolved slice into a commit, deriving `commit_id` from the slice's
    /// content hash so identical inputs dedup. `lineage` is descriptive metadata and
    /// does NOT affect the id.
    #[must_use]
    pub fn seal(slice: ResolvedSlice, lineage: BTreeMap<String, String>) -> Self {
        Self {
            commit_id: slice.content_hash.clone(),
            content_hash: slice.content_hash.clone(),
            slice_spec: slice.spec.clone(),
            resolved_manifest: slice,
            lineage,
        }
    }
}

/// Compile a [`SliceSpec`] against already-resolved rollouts and feedback into a
/// pinned [`ResolvedSlice`].
///
/// For each rollout that passes the metadata filters (policy / task / site / window):
///   1. gather every feedback whose target is that rollout's step,
///   2. pick the authoritative one via the JOIN crate's latest-wins resolver — a
///      curator/manual binding outranks an automated one, else newest wins — so this
///      never re-implements precedence,
///   3. apply the confidence gate: a retracted authoritative binding, or one below
///      `spec.min_confidence`, is held out and flagged needs-review (a doubtful label
///      poisons the model), everything else is pinned.
///
/// The result is grain-aware: a `Rollout` slice pins one item per step; an `Episode`
/// slice still pins per-step items (the trajectory's frames) but its `episodes` list
/// is the rolled-up trajectory set the export trains on.
#[must_use]
pub fn compile_slice(
    spec: &SliceSpec,
    rollouts: &[Rollout],
    feedback: &[Feedback],
) -> ResolvedSlice {
    // Resolve the tenant from the inputs. All candidates share one tenant (curation is
    // a per-tenant operation); we read it off the first rollout, defaulting to an empty
    // tenant for the degenerate empty-input case so the function stays total.
    let tenant_id = rollouts
        .first()
        .map(|r| r.robot.tenant_id.clone())
        .unwrap_or_else(|| TenantId::new(""));

    // Index feedback by the rollout it targets, so each candidate's bindings are an
    // O(1) lookup instead of a full scan per rollout.
    let mut by_rollout: BTreeMap<RolloutId, Vec<Feedback>> = BTreeMap::new();
    for fb in feedback {
        if let FeedbackTarget::Rollout(rid) = fb.target {
            by_rollout.entry(rid).or_default().push(fb.clone());
        }
    }

    let mut items: Vec<PinnedItem> = Vec::new();
    let mut needs_review: Vec<NeedsReview> = Vec::new();

    for rollout in rollouts {
        if !matches_metadata(spec, rollout) {
            continue;
        }

        // Synthetic data is opt-in. Unless the spec explicitly sets
        // `include_synthetic`, a synthetic rollout is dropped here so it is never
        // silently mixed into a training set — a sim is an approximation and synthetic
        // data must not be mistaken for real field evidence. When admitted, it is still
        // tagged synthetic on the pinned item below so it stays separable/down-weightable.
        if rollout.provenance.is_synthetic() && !spec.include_synthetic {
            continue;
        }

        let candidates = by_rollout
            .get(&rollout.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        // Latest-wins, reused from the JOIN crate: manual outranks automated, else
        // newest. Returns the non-retracted winner, or `None` if all are retracted.
        let Some(winner) = winning_binding(&tenant_id, candidates) else {
            // Either no binding at all, or every binding is retracted: a retraction is
            // an explicit withdrawal, so distinguish the two for the reviewer.
            let any_retracted = candidates.iter().any(|f| f.retracted);
            needs_review.push(NeedsReview {
                rollout_id: rollout.id,
                feedback_id: None,
                reason: if any_retracted {
                    ReviewReason::Retracted
                } else {
                    ReviewReason::NoSurvivingBinding
                },
            });
            continue;
        };

        // The confidence gate. `winning_binding` already drops retracted rows, so a
        // surviving winner is non-retracted; the remaining bar is the confidence floor.
        // A below-floor binding is held out and flagged — never trained on.
        if winner.join_confidence < spec.min_confidence {
            needs_review.push(NeedsReview {
                rollout_id: rollout.id,
                feedback_id: Some(winner.id),
                reason: ReviewReason::BelowMinConfidence,
            });
            continue;
        }

        // The failure-class filter is checked against the AUTHORITATIVE binding's
        // value (not any candidate), so a class filter can never be satisfied by a
        // losing/retracted row that the latest-wins rule discarded.
        if !matches_failure_class(spec, &winner.value) {
            continue;
        }

        // The outcome-time window is checked against the authoritative binding too,
        // since the time we care about is when the scoring outcome happened.
        if !spec.outcome_ts.contains(winner.outcome_ts_ns) {
            continue;
        }

        items.push(PinnedItem {
            episode_id: rollout.episode_id,
            rollout_id: rollout.id,
            feedback_id: winner.id,
            provenance: rollout.provenance.clone(),
        });
    }

    // Deterministic order: sort the pinned items by rollout id so the content hash is
    // independent of input ordering. Two runs over the same data hash identically.
    items.sort_by_key(|i| i.rollout_id.as_uuid());
    needs_review.sort_by_key(|n| n.rollout_id.as_uuid());

    // Roll up the distinct episodes the pinned items cover, sorted. This is the unit an
    // episode-grain dataset trains on and the lineage key for episode<->policy.
    let mut episodes: Vec<EpisodeId> = items.iter().map(|i| i.episode_id).collect();
    episodes.sort_by_key(EpisodeId::as_uuid);
    episodes.dedup();

    let content_hash = hash_pinned_set(spec, &items, &episodes);

    ResolvedSlice {
        tenant_id,
        spec: spec.clone(),
        items,
        episodes,
        needs_review,
        content_hash,
    }
}

/// True iff the rollout passes the spec's metadata filters (policy / task / site).
/// Each `None` filter matches everything, so filters only ever narrow.
fn matches_metadata(spec: &SliceSpec, rollout: &Rollout) -> bool {
    if let Some(pv) = &spec.policy_version
        && &rollout.policy_version != pv
    {
        return false;
    }
    if let Some(task) = &spec.task_id
        && &rollout.task_id != task
    {
        return false;
    }
    if let Some(site) = &spec.site {
        // `site` is an open tag; absence of the tag means "not this site".
        if rollout.tags.get(SITE_TAG_KEY) != Some(site) {
            return false;
        }
    }
    true
}

/// True iff the spec's failure-class filter (if any) matches the authoritative value.
/// A `None` filter matches any value; a `Some` filter matches only a
/// `FeedbackValue::FailureClass` of that exact class.
fn matches_failure_class(spec: &SliceSpec, value: &FeedbackValue) -> bool {
    match &spec.failure_class {
        None => true,
        Some(want) => matches!(value, FeedbackValue::FailureClass { class } if class == want),
    }
}

/// Hash the pinned set into a stable hex digest.
///
/// Computed over the spec plus the (already sorted, so order-stable) pinned items and
/// episodes, via a deterministic `serde_json` serialization fed to a fixed hasher.
/// Equal pinned sets produce equal hashes (dedup), and changing any pinned id, the
/// spec, or an episode changes the hash (a genuinely new dataset). Episodes are
/// derivable from items, but folding them in explicitly keeps the hash sensitive to
/// the grain rollup as a first-class part of the dataset identity.
fn hash_pinned_set(spec: &SliceSpec, items: &[PinnedItem], episodes: &[EpisodeId]) -> String {
    // A small, fixed-field-order view so the bytes hashed are deterministic. Using
    // serde_json (not Debug) keeps the serialization a stable, documented form.
    #[derive(Serialize)]
    struct HashView<'a> {
        spec: &'a SliceSpec,
        items: &'a [PinnedItem],
        episodes: &'a [EpisodeId],
    }
    let view = HashView {
        spec,
        items,
        episodes,
    };
    let bytes = serde_json::to_vec(&view).expect("pinned-set view serializes (no non-string keys)");

    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    let digest = hasher.finish();
    // The grain is part of the dataset's identity; including its tag in the printed
    // hash makes a per-step vs per-trajectory dataset visibly distinct even at a glance.
    let grain_tag = match spec.grain {
        Grain::Rollout => "rollout",
        Grain::Episode => "episode",
    };
    format!("sha-fnv:{grain_tag}:{digest:016x}")
}
