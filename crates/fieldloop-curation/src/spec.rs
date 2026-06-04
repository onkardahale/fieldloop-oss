//! `SliceSpec` — the query that selects training data.
//!
//! A slice spec is the human-written *intent*: "give me every step of policy X that
//! failed with a planning error at site Y, in this time window, that I'm confident
//! enough about to train on". It is deliberately just filters, never the resolved
//! rows — the rows are pinned separately (see [`crate::slice::ResolvedSlice`]) so the
//! query and the frozen result stay distinct.
//!
//! All fields except the grain are *optional* filters: a `None`/empty filter matches
//! everything on that axis. This makes a spec additive — adding a filter can only
//! narrow the slice, never silently widen it.

use serde::{Deserialize, Serialize};

use fieldloop_types::{FailureClass, PolicyVersion};

/// The default confidence floor a slice applies when the spec does not name one.
///
/// A binding below this confidence is excluded from training data and surfaced as
/// "needs review" instead. The value is intentionally conservative: a confidently
/// *wrong* label poisons the model far worse than a missing example helps it, so the
/// default leans toward excluding a doubtful binding rather than admitting it. An
/// operator who knows their attribution is well-calibrated can lower it in the spec.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.8;

/// The grain a slice is compiled at — whether each training item is a single step or
/// a whole trajectory.
///
/// Closed enum, not a bool, so the two meanings can never be confused and a future
/// grain forces every consumer to account for it. The grain decides how candidates
/// are grouped: a `Rollout` slice is one item per inference step, an `Episode` slice
/// rolls the steps up by their shared `episode_id` into one trajectory item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grain {
    /// One training item per inference step (per-step). The finest grain.
    Rollout,
    /// One training item per trajectory: the steps sharing an `episode_id` are rolled
    /// up into a single episode. The grain GR00T/openpi-style training consumes.
    Episode,
}

/// A half-open `[start, end)` window over outcome event time (nanoseconds since the
/// epoch), used to filter on when the outcome that produced a binding actually
/// happened.
///
/// Half-open so adjacent windows tile without double-counting the boundary instant.
/// Either edge is optional: `None` means unbounded on that side, so a spec can ask
/// for "everything since T" without inventing a far-future end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OutcomeTsWindow {
    /// Inclusive lower bound on `outcome_ts_ns`. `None` == unbounded below.
    pub start_ns: Option<i64>,
    /// Exclusive upper bound on `outcome_ts_ns`. `None` == unbounded above.
    pub end_ns: Option<i64>,
}

impl OutcomeTsWindow {
    /// True iff `ts_ns` falls in this (possibly unbounded) window. An absent edge
    /// never excludes, so an all-`None` window matches every timestamp.
    #[must_use]
    pub fn contains(&self, ts_ns: i64) -> bool {
        self.start_ns.is_none_or(|s| ts_ns >= s) && self.end_ns.is_none_or(|e| ts_ns < e)
    }
}

/// The query that selects training data.
///
/// Every filter is optional and additive (a `None`/empty filter matches everything),
/// so building a spec up can only narrow the slice. The one required choice is the
/// [`Grain`], because per-step and per-trajectory datasets are genuinely different
/// shapes and there is no safe default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SliceSpec {
    /// Keep only rollouts produced by this policy version. `None` matches any policy.
    #[serde(default)]
    pub policy_version: Option<PolicyVersion>,
    /// Keep only items whose authoritative feedback carries this categorical failure
    /// class. `None` matches any class (and items with non-failure-class values).
    #[serde(default)]
    pub failure_class: Option<FailureClass>,
    /// Keep only rollouts for this task. `None` matches any task.
    #[serde(default)]
    pub task_id: Option<String>,
    /// Keep only rollouts carrying this `site` tag value. `None` matches any site.
    /// Read from the rollout's open `tags` map under the key `site`.
    #[serde(default)]
    pub site: Option<String>,
    /// Keep only items whose authoritative feedback's `outcome_ts_ns` is in this
    /// window. Defaults to fully unbounded (matches every time).
    #[serde(default)]
    pub outcome_ts: OutcomeTsWindow,
    /// The confidence floor. A binding whose `join_confidence` is below this is
    /// EXCLUDED from the slice and flagged needs-review, because a low-confidence
    /// label that is silently wrong poisons the model. Defaults to
    /// [`DEFAULT_MIN_CONFIDENCE`] when omitted from the spec.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    /// Whether synthetic rollouts (sim / Cosmos-style augmentation) are admitted into
    /// the slice. Defaults to `false`: synthetic training data must be *opt-in*, never
    /// silently mixed in, because a sim is an approximation and synthetic data must
    /// never be mistaken for real field evidence — especially in a safety-relevant set.
    /// When `false`, a synthetic rollout is dropped from the slice; when `true`, it is
    /// admitted but stays tagged synthetic on every pinned item so a consumer can still
    /// down-weight or separate it.
    #[serde(default)]
    pub include_synthetic: bool,
    /// Per-step or per-trajectory — the one non-optional choice.
    pub grain: Grain,
}

/// The serde default for [`SliceSpec::min_confidence`]: the conservative floor that
/// keeps a doubtful binding out of training data unless the operator opts lower.
fn default_min_confidence() -> f32 {
    DEFAULT_MIN_CONFIDENCE
}

/// The well-known rollout tag key a [`SliceSpec::site`] filter matches against. A
/// constant so the producer (capture SDK) and this consumer agree on one spelling.
pub const SITE_TAG_KEY: &str = "site";
