//! `Feedback` — the attributed binding: append-only, the output of the JOIN, and
//! the place delayed outcome-to-rollout attribution is surfaced as data.
//!
//! Feedback is append-only with an explicit `label_kind` slot so two ideas stay
//! distinct:
//!   * **supersession** — the newest authoritative label within one slot wins, and
//!   * **multiplicity** — a takeover *and* a downstream failure on the same target
//!     coexist because they sit in different slots.
//!
//! With these separated, retraction is also expressible (append a retracted row).
//! Collapsing everything into one row-per-metric-table would conflate the two and
//! make a correction silently overwrite an unrelated signal.
//!
//! Several illegal states are made unrepresentable:
//!   * the JOIN target is a single [`FeedbackTarget`] sum type, so the target id
//!     and its grain (rollout vs episode) can never disagree and double-count;
//!   * the value is a typed [`FeedbackValue`] sum type, so a categorical failure
//!     class can't be stuffed into a float column;
//!   * `source_outcome_id` is `Option`: `None` is exactly the synthesized-absence
//!     case (there is no originating outcome to point back to), not a nil sentinel.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ids::{EpisodeId, FeedbackId, OutcomeId, RolloutId};

/// The JOIN target: which grain this feedback scores. Bundling the id and its grain
/// into one sum type means a row can never say "this scores a rollout" while
/// carrying an episode id, so the classic double-count bug — where the id column and
/// a separate type column disagree — is unrepresentable.
///
/// Modeled as a plain Rust enum carrying the strongly-typed id ([`RolloutId`] /
/// [`EpisodeId`]). The storage layer (not this crate) projects it onto the flat
/// `(target_type, target_id)` columns via [`FeedbackTarget::type_code`] and
/// [`FeedbackTarget::target_uuid`]. Kept *un-flattened* on the Rust wire form
/// deliberately: flattening a tagged enum into a struct is a serde footgun, so we
/// serialize it as a small tagged object under `target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackTarget {
    /// Scores a single rollout step (the inference-grain JOIN: `rollout.id =
    /// target_id`).
    Rollout(RolloutId),
    /// Scores a whole episode/trajectory (the episode-grain JOIN: `episode_id =
    /// target_id`) — e.g. an episode return or task-completed signal.
    Episode(EpisodeId),
}

impl FeedbackTarget {
    /// The discriminant as the schema's `Enum8` value (1 = rollout, 2 = episode),
    /// for storage encoding.
    #[must_use]
    pub fn type_code(&self) -> u8 {
        match self {
            FeedbackTarget::Rollout(_) => 1,
            FeedbackTarget::Episode(_) => 2,
        }
    }

    /// The raw target UUID, for the flat `target_id UUID` storage column.
    #[must_use]
    pub fn target_uuid(&self) -> uuid::Uuid {
        match self {
            FeedbackTarget::Rollout(id) => id.as_uuid(),
            FeedbackTarget::Episode(id) => id.as_uuid(),
        }
    }
}

/// The supersession slot. Supersession is resolved *per `(target, label_kind)`* —
/// so a newer `Intervention` supersedes an older `Intervention`, while a coexisting
/// `TerminalOutcome` on the same target is left untouched (multiplicity). A closed
/// enum, so the per-slot grouping the read does is total and a new slot kind forces
/// every consumer to account for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelKind {
    /// The terminal success/failure of the grain.
    TerminalOutcome,
    /// A human intervention (takeover / e-stop) during the grain.
    Intervention,
    /// A curator/manual annotation.
    Annotation,
    /// An episode-level scalar return (episode grain).
    EpisodeReturn,
}

/// The failure taxonomy: a closed set used as the value of a categorical-failure
/// feedback. No free strings — a new failure mode is a new variant the compiler
/// forces every consumer to handle, so failure analysis can't silently miss a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Perception,
    Planning,
    Manipulation,
    Hardware,
    Network,
    Environment,
    Operator,
}

/// The typed feedback value: boolean / float / correction-pointer, plus the
/// categorical failure class. A sum type so the value's shape always matches its
/// meaning — a demonstration ref can't be read as a float, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "value_type", rename_all = "snake_case")]
pub enum FeedbackValue {
    /// Boolean outcome (e.g. success/failure, grasped, collision).
    Boolean { value: bool },
    /// Float score (e.g. reward, return, distance-to-goal). `f32` to match the
    /// storage column width.
    Float { value: f32 },
    /// A categorical failure class (closed taxonomy).
    FailureClass { class: FailureClass },
    /// A pointer to a corrected trajectory/demonstration in the customer's store.
    /// The bytes live in the customer's object storage (Fieldloop indexes pointers,
    /// not bytes); this is the object key + digest.
    DemonstrationRef {
        object_key: String,
        content_sha256: Option<String>,
    },
}

/// How the outcome was attributed to its target — the attribution cascade. A closed
/// enum, ordered by how the cascade is tried: explicit → temporal (the workhorse) →
/// spatial → causal → manual; plus the synthesized-absence path for "nothing
/// happened in a covered window".
///
/// **Precedence:** `Manual` outranks all automated methods — the read orders manual
/// first, then by recency, so an automated re-join can never shadow a curator's
/// correction. That ordering is a read-side rule; this enum just names the method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinMethod {
    /// The source threaded an explicit rollout id (e.g. eval rollouts). The target
    /// is known, not inferred, so `confidence == 1.0`.
    Explicit,
    /// Windowed on the reconciled monotonic clock (the workhorse).
    Temporal,
    /// Spatial co-location.
    Spatial,
    /// Causal inference over downstream effects.
    Causal,
    /// A curator's manual correction — highest read precedence; also serves as a
    /// ground-truth label used to calibrate the confidence of automated methods.
    Manual,
    /// Coverage-computed absence-of-event: no outcome occurred in a window proven
    /// covered by heartbeats. Has no `source_outcome_id` (there is no real outcome).
    /// Confidence is a function of how fully the window was covered, not a flat
    /// constant.
    SyntheticAbsence,
}

/// Where the outcome signal came from. A closed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackSource {
    /// An on-robot/edge failure detector.
    Detector,
    /// A teleoperator action.
    Teleop,
    /// A human curator.
    Curator,
    /// A sim / eval harness, whose outcomes are explicitly attributed.
    Sim,
    /// The absence-synthesis sweeper.
    AbsenceSweeper,
}

/// `Feedback` — one attributed binding row. Append-only; supersession and
/// retraction are expressed by appending newer rows, never by mutating in place.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feedback {
    /// FRESH `Uuid::now_v7()` per row, in real wall time — **never** hash-derived. A
    /// hash-seeded v7 would fix the timestamp and break "latest write wins"; instead
    /// a re-attribution mints a genuinely-newer id so it can win on recency.
    pub id: FeedbackId,
    /// `tenant_id` — leading column of the storage sort and the isolation key. Kept
    /// out of [`FeedbackTarget`] because the tenant is an independent isolation axis,
    /// not part of the grain being scored.
    pub tenant_id: crate::tenant::TenantId,

    /// The grain this scores: id + grain, indivisible, so the two can't disagree.
    pub target: FeedbackTarget,
    /// The supersession slot. Supersession is per `(target, label_kind)`; distinct
    /// slots coexist on the same target (multiplicity).
    pub label_kind: LabelKind,
    /// The metric this row scores. Part of the read predicate and the by-target
    /// lookup key.
    pub metric_name: String,
    /// The typed value (shape always matches meaning).
    pub value: FeedbackValue,

    // ---- attribution provenance -------------------------------------------
    /// How the binding was made (the cascade).
    pub join_method: JoinMethod,
    /// CALIBRATED confidence in `[0,1]` — measured against ground-truth labels, not
    /// an asserted constant. Safety eval consumes only rows with confidence `1.0`
    /// that are also backed by a ledger-trusted rollout, so a guessed binding can
    /// never feed a safety decision.
    pub join_confidence: f32,
    /// Version of the attribution logic that produced this row (feeds `dedup_key`).
    pub join_version: String,
    /// Version of the calibration curves used for `join_confidence`, so a row's
    /// confidence can be traced to how it was calibrated.
    pub calibration_version: String,
    /// Back-pointer to the originating [`OutcomeEvent`]. `None` **iff**
    /// `join_method == SyntheticAbsence` (there is no real outcome) — modeled as
    /// `Option`, not a nil sentinel.
    pub source_outcome_id: Option<OutcomeId>,
    /// Outcome-arrival delay relative to the rollout, in milliseconds. `Option`
    /// because synthetic-absence has no real outcome event to measure against.
    pub delay_ms: Option<i64>,

    // ---- lifecycle ----------------------------------------------------------
    /// Tombstone-by-append: a curator delete or a post-close re-open appends a
    /// `retracted = true` row rather than mutating history, keeping the log
    /// append-only.
    #[serde(default)]
    pub retracted: bool,
    /// Idempotency key derived from the attribution inputs (source outcome, target,
    /// join version, and a digest of the attribution inputs). A **string**, NOT a
    /// hash-derived UUIDv7: a re-attribution intentionally changes the digest → a new
    /// key → a genuinely-newer winning row, while a true replay reuses the key and is
    /// skipped. (Contrast the `id` above, which is a fresh v7 every time.)
    pub dedup_key: String,
    /// The outcome's own event time (ns since epoch) — drives partition pruning on
    /// read, so a dataset build prunes by time window instead of scanning full
    /// history.
    pub outcome_ts_ns: i64,

    // ---- distributed credit (multi-step windowed attribution) ---------------
    /// This row's share of its outcome's calibrated in-window confidence, in
    /// `(0, 1]`. When several rollouts fall inside one outcome's attribution
    /// window, the outcome's single calibrated confidence is split across them by a
    /// recency-decay kernel and each contributor carries its own `credit_weight`;
    /// the weights of one contributing set sum to that calibrated confidence. The
    /// unambiguous single-candidate case is `1.0` (full credit), which is also the
    /// `#[serde(default)]` so any row written before this field existed reads back
    /// as full credit rather than zero — keeping the column a safe additive append.
    #[serde(default = "default_credit_weight")]
    pub credit_weight: f32,
    /// Groups the N rows produced for one outcome's distributed credit under a shared
    /// id, so a reader can recognize co-contributors and re-sum their weights.
    /// `None` is the single-contributor (or pre-field) case: a lone full-credit row
    /// is not part of any multi-contributor set, so it carries no group id. Defaults
    /// to `None` for rows written before the field existed.
    #[serde(default)]
    pub contributing_set_id: Option<Uuid>,
}

/// The default `credit_weight` for a row missing the field: full credit. A row that
/// predates distributed credit was the single unambiguous binding, so it reads back
/// as `1.0`, never `0.0` — which would silently zero out historical credit.
fn default_credit_weight() -> f32 {
    1.0
}

impl Feedback {
    /// True iff this row clears the *confidence* bar for the safety-regression eval
    /// path: not retracted, `join_confidence == 1.0`, AND it is the single
    /// unambiguous in-window binding (`credit_weight == 1.0` and no contributing-set
    /// group). Safety must rest only on bindings that are certain, never guessed, and
    /// never on one share of a credit that was split across several candidate
    /// rollouts — when the cause is genuinely ambiguous between co-contributors, no
    /// single share is a certain enough anchor for a physical-safety veto, so a
    /// distributed-credit row is excluded here even if its calibrated confidence
    /// reads 1.0.
    ///
    /// This is only the confidence half of the gate. The full gate also requires the
    /// underlying rollout to be ledger-`trusted` — a property of the
    /// [`crate::Rollout`], not visible on this row — so callers MUST also check
    /// `Rollout.trust == Trusted`. Kept here as a guarded helper, not enforced.
    #[must_use]
    pub fn is_safety_eligible_confidence(&self) -> bool {
        !self.retracted
            && self.join_confidence == 1.0
            && self.credit_weight == 1.0
            && self.contributing_set_id.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenant::TenantId;

    /// Build a full-confidence, single-contributor rollout-target feedback row.
    fn full_credit_row() -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(RolloutId::new()),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "m".into(),
            value: FeedbackValue::Boolean { value: true },
            join_method: JoinMethod::Explicit,
            join_confidence: 1.0,
            join_version: "j".into(),
            calibration_version: "c".into(),
            source_outcome_id: None,
            delay_ms: None,
            retracted: false,
            dedup_key: "dk".into(),
            outcome_ts_ns: 0,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    /// A single full-confidence, full-credit, ungrouped row is safety-eligible — the
    /// unambiguous binding the safety gate is allowed to rest on.
    #[test]
    fn single_full_credit_row_is_safety_eligible() {
        assert!(full_credit_row().is_safety_eligible_confidence());
    }

    /// A distributed-credit share is NEVER safety-eligible even at full calibrated
    /// confidence: a partial `credit_weight` means the cause was split across several
    /// candidates, so no one share is a certain anchor for a physical-safety veto.
    #[test]
    fn distributed_credit_share_is_not_safety_eligible() {
        let mut f = full_credit_row();
        f.credit_weight = 0.6;
        f.contributing_set_id = Some(Uuid::now_v7());
        assert!(!f.is_safety_eligible_confidence());
    }

    /// Belonging to a contributing set disqualifies a row from safety even if its
    /// own weight happened to round to 1.0 — group membership alone marks ambiguity.
    #[test]
    fn grouped_row_is_not_safety_eligible() {
        let mut f = full_credit_row();
        f.contributing_set_id = Some(Uuid::now_v7());
        assert!(!f.is_safety_eligible_confidence());
    }

    /// A row whose JSON predates the credit columns deserializes with full credit and
    /// no group, so it stays safety-eligible — the additive append is back-compatible.
    #[test]
    fn old_row_without_credit_fields_reads_as_full_credit() {
        let mut json = serde_json::to_value(full_credit_row()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("credit_weight");
        obj.remove("contributing_set_id");
        let back: Feedback = serde_json::from_value(json).unwrap();
        assert_eq!(back.credit_weight, 1.0);
        assert!(back.contributing_set_id.is_none());
        assert!(back.is_safety_eligible_confidence());
    }
}
