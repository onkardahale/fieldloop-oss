//! The canonical rollout -> outcome (reward) join read.
//!
//! This is the read that turns the append-only logs into a labeled dataset: for each
//! rollout in a time window, attach its latest authoritative feedback in a given slot
//! (e.g. the `reward` `TerminalOutcome`). It is built entirely through the
//! fail-closed [`crate::tenant::TenantQueryBuilder`], so it is always tenant-scoped by
//! a bound parameter and never string-interpolates a value.
//!
//! Several correctness properties are baked into the generated SQL, each guarding a
//! specific failure mode:
//!
//!   * **Confidence predicate in the `ON` clause, not the `WHERE`.** A LEFT JOIN with
//!     a feedback predicate in the `WHERE` silently degrades to an INNER JOIN —
//!     rollouts still awaiting an outcome would vanish from the result. Putting the
//!     confidence filter in the `ON` keeps the LEFT JOIN left, so awaiting-outcome
//!     rollouts survive, and a `has_outcome` column surfaces coverage explicitly.
//!   * **Windowed by `outcome_ts`.** Both the feedback subquery and the result are
//!     bounded by an `outcome_ts` range so the read prunes day partitions instead of
//!     scanning all history.
//!   * **Latest-wins per slot via `ROW_NUMBER`.** Supersession is resolved at read
//!     time by `ROW_NUMBER() OVER (PARTITION BY (target_id, target_type, label_kind)
//!     ORDER BY ts DESC)` keeping `rn = 1`. Partitioning by the full slot — NOT by
//!     outcome/metric type — means a correction supersedes within its slot while a
//!     different slot on the same target coexists.
//!   * **`SETTINGS join_use_nulls = 1` so the coverage column is correct.** ClickHouse
//!     fills the unmatched side of a LEFT JOIN with each column's type default (a
//!     non-nullable `UUID` becomes the all-zero UUID, not NULL), which would make
//!     `f.target_id IS NOT NULL` — and therefore `has_outcome` — always true even for a
//!     rollout with no feedback. Enabling `join_use_nulls` gives the join standard SQL
//!     semantics (unmatched right-side columns are NULL), so `has_outcome` truly
//!     distinguishes an attributed rollout from one still awaiting an outcome.
//!   * **Reads through the `*ByTargetId` materialized view.** The base feedback table
//!     is sorted for ingestion, not for the id join; the `FeedbackByTargetId` view is
//!     re-keyed on `target_id`, so the join is a cheap merge.

use crate::tenant::{Dialect, ParamValue, TenantQuery, TenantQueryBuilder};
use fieldloop_types::TenantId;

/// Parameters for the canonical reward/outcome join read. All values become bound
/// parameters; none is ever interpolated into the SQL text.
#[derive(Debug, Clone)]
pub struct RewardJoinParams {
    /// The tenant whose data to read — REQUIRED. The builder fails closed without it.
    pub tenant: TenantId,
    /// The metric to attach (e.g. `"reward"`). A bound parameter, matched in the
    /// feedback subquery.
    pub metric_name: String,
    /// The supersession slot to resolve, as its storage `label_kind` string (e.g.
    /// `"terminal_outcome"`). A bound parameter.
    pub label_kind: String,
    /// Minimum calibrated join confidence to accept a binding, in `[0,1]`. Applied in
    /// the `ON` clause so a low-confidence binding drops the *match* but keeps the
    /// rollout row (with `has_outcome = 0`).
    pub min_confidence: f64,
    /// Inclusive lower bound of the `outcome_ts` window (fractional seconds since the
    /// epoch). Prunes feedback partitions.
    pub outcome_ts_from: f64,
    /// Exclusive upper bound of the `outcome_ts` window (fractional seconds since the
    /// epoch).
    pub outcome_ts_to: f64,
}

/// Build the canonical reward/outcome join read as a tenant-scoped, fully
/// parameterized [`TenantQuery`].
///
/// Joins `Rollout` (LEFT) to the latest-wins feedback resolved through the
/// `FeedbackByTargetId` view, scoped to one tenant, windowed by `outcome_ts`, with
/// the confidence filter in the `ON` clause and a `has_outcome` coverage column. The
/// returned `(sql, params)` is ready for the live ClickHouse client; `params[0]` is
/// always the tenant.
///
/// # Errors
/// Returns [`crate::tenant::BuildError::MissingTenant`] only if the builder is misused
/// internally; with a tenant always supplied here it builds successfully. The
/// `Result` is surfaced so the fail-closed guarantee is visible at the call site.
pub fn reward_join(p: &RewardJoinParams) -> Result<TenantQuery, crate::tenant::BuildError> {
    // The body uses two placeholder kinds the builder understands:
    //   * `{tenant}`  -> the injected leading tenant scope (bound param p0).
    //   * `{param}`   -> the next pushed value, in order.
    // Every value (metric, label_kind, confidence, window bounds) is a `{param}` —
    // none is written inline, so a value with SQL metacharacters can never reach the
    // SQL text. The tenant predicate appears in BOTH the rollout filter and the
    // feedback subquery, so neither side leaks across tenants; the builder injects the
    // same leading bound value for `{tenant}`.
    //
    // Note the LEFT JOIN with the confidence predicate in the ON clause, the
    // `f.target_id IS NOT NULL AS has_outcome` coverage column, the ROW_NUMBER over
    // the full slot, the outcome_ts window on both the subquery and the outer filter,
    // and the FeedbackByTargetId view as the feedback source.
    let body = "\
SELECT
    r.id AS rollout_id,
    r.episode_id AS episode_id,
    r.policy_version AS policy_version,
    f.value_float AS reward,
    f.value_bool AS reward_bool,
    f.join_confidence AS join_confidence,
    f.join_method AS join_method,
    (f.target_id IS NOT NULL) AS has_outcome
FROM Rollout AS r
LEFT JOIN (
    SELECT
        target_id,
        target_type,
        label_kind,
        value_float,
        value_bool,
        join_confidence,
        join_method,
        id_uint,
        ROW_NUMBER() OVER (
            PARTITION BY target_id, target_type, label_kind
            ORDER BY ts DESC, id_uint DESC
        ) AS rn
    FROM FeedbackByTargetId
    WHERE tenant_id = {tenant}
      AND metric_name = {param}
      AND label_kind = {param}
      AND retracted = 0
      AND outcome_ts >= {param}
      AND outcome_ts < {param}
) AS f
    ON r.id = f.target_id
   AND f.target_type = 'rollout'
   AND f.rn = 1
   AND f.join_confidence >= {param}
WHERE r.tenant_id = {tenant}
ORDER BY r.id
SETTINGS join_use_nulls = 1;";

    TenantQueryBuilder::new(Dialect::ClickHouse)
        .tenant(p.tenant.clone())
        .body(body)
        // Push the values in the exact order their `{param}` placeholders appear.
        .push(ParamValue::Str(p.metric_name.clone()))
        .push(ParamValue::Str(p.label_kind.clone()))
        .push(ParamValue::F64(p.outcome_ts_from))
        .push(ParamValue::F64(p.outcome_ts_to))
        .push(ParamValue::F64(p.min_confidence))
        .build()
}

/// The bounded window for a write-side attribution read: one tenant plus an inclusive
/// `ts_wall_ns` (robot wall-estimate nanosecond) range. All three reads
/// ([`recent_rollouts`], [`recent_outcomes`], [`recent_feedback`]) share this shape so a
/// single attribution pass reads a consistent slice across the three streams.
///
/// The window bounds are `ts_wall_ns` because that is the one coarse, cross-boot
/// nanosecond stamp every stream carries on its base table (`Rollout` / `OutcomeEvent`
/// store `ts_wall_ns`; `Feedback` stores `outcome_ts` derived from it). The monotonic
/// clock is the attribution AUTHORITY once rows are in memory, but it is per-boot and so
/// cannot bound a SQL scan that spans boots — the wall estimate is the only column that
/// can prune the read to a recent slice. A narrow window keeps the write-side cascade
/// reading "what just landed" rather than all history.
#[derive(Debug, Clone)]
pub struct RecentWindowParams {
    /// The tenant whose rows to read — REQUIRED. The builder fails closed without it.
    pub tenant: TenantId,
    /// Inclusive lower bound of the `ts_wall_ns` window (nanoseconds since the epoch).
    pub ts_wall_from_ns: i64,
    /// Exclusive upper bound of the `ts_wall_ns` window (nanoseconds since the epoch).
    pub ts_wall_to_ns: i64,
}

/// Read one tenant's recent [`fieldloop_types::Rollout`] rows in a `ts_wall_ns` window,
/// as a tenant-scoped, fully parameterized [`TenantQuery`].
///
/// This is the LEFT side the write-side attribution cascade consumes: the deployed-policy
/// decisions an outcome must be bound back to. It is the read counterpart of the
/// `Rollout` insert path — same base table, same flat columns — selecting exactly the
/// columns [`crate::clickhouse::rows::rollout_row`] writes, so a stored rollout can be
/// reconstructed and fed to the engine. Every value is a bound parameter; the tenant is
/// the leading bound scope, matching the `reward_join` discipline so the same isolation
/// guarantee covers this read.
///
/// # Errors
/// Returns [`crate::tenant::BuildError`] only on builder misuse; with a tenant always
/// supplied it builds. The `Result` keeps the fail-closed guarantee visible at the call
/// site.
pub fn recent_rollouts(p: &RecentWindowParams) -> Result<TenantQuery, crate::tenant::BuildError> {
    // Select exactly the columns the row serializer writes, so the read is the inverse of
    // the write and a parser can rebuild the typed Rollout. `ts_wall_ns` bounds the scan;
    // every value is a `{param}`, the tenant the leading `{tenant}` scope.
    let body = "\
SELECT
    id,
    tenant_id,
    robot_id,
    episode_id,
    step_index,
    boot_id,
    mono_ns,
    ts_wall_ns,
    server_ts_ingest_ns,
    server_anchor_offset_ns,
    policy_version,
    model_hash,
    trust,
    embodiment,
    task_id,
    eval_run_id,
    arm_label,
    schema_conformance,
    inference_us,
    provenance_kind,
    provenance_generator,
    has_pose,
    pose_x,
    pose_y,
    pose_z,
    pose_qw,
    pose_qx,
    pose_qy,
    pose_qz,
    frame_id,
    station_id
FROM Rollout
WHERE tenant_id = {tenant}
  AND ts_wall_ns >= {param}
  AND ts_wall_ns < {param}
ORDER BY id;";

    TenantQueryBuilder::new(Dialect::ClickHouse)
        .tenant(p.tenant.clone())
        .body(body)
        .push(ParamValue::I64(p.ts_wall_from_ns))
        .push(ParamValue::I64(p.ts_wall_to_ns))
        .build()
}

/// Read one tenant's recent [`fieldloop_types::OutcomeEvent`] rows in a `ts_wall_ns`
/// window, as a tenant-scoped, fully parameterized [`TenantQuery`].
///
/// These are the raw, often-implicit outcomes the cascade attributes back to rollouts;
/// a `Heartbeat`-kind outcome in this set rides the coverage path. The selected columns
/// mirror [`crate::clickhouse::rows::outcome_row`] so the typed `OutcomeEvent` (including
/// its monotonic clock and any explicit rollout id) can be reconstructed for the engine.
/// Every value is bound; the tenant is the leading scope.
///
/// # Errors
/// Returns [`crate::tenant::BuildError`] only on builder misuse; the `Result` surfaces
/// the fail-closed guarantee.
pub fn recent_outcomes(p: &RecentWindowParams) -> Result<TenantQuery, crate::tenant::BuildError> {
    // The spatial/causal columns are selected too, so the read rebuilds the inputs the
    // spatial (pose co-location) and causal (downstream-edge) tiers fire on — without
    // them those tiers could never run on the live read path. `has_pose` is the
    // load-bearing flag that distinguishes "no pose" from "a pose at the origin".
    let body = "\
SELECT
    id,
    tenant_id,
    robot_id,
    boot_id,
    mono_ns,
    ts_wall_ns,
    outcome_kind,
    explicit_rollout_id,
    trust,
    has_pose,
    pose_x,
    pose_y,
    pose_z,
    pose_qw,
    pose_qx,
    pose_qy,
    pose_qz,
    frame_id,
    causal_parents
FROM OutcomeEvent
WHERE tenant_id = {tenant}
  AND ts_wall_ns >= {param}
  AND ts_wall_ns < {param}
ORDER BY id;";

    TenantQueryBuilder::new(Dialect::ClickHouse)
        .tenant(p.tenant.clone())
        .body(body)
        .push(ParamValue::I64(p.ts_wall_from_ns))
        .push(ParamValue::I64(p.ts_wall_to_ns))
        .build()
}

/// Read one tenant's existing [`fieldloop_types::Feedback`] rows in an `outcome_ts`
/// window, as a tenant-scoped, fully parameterized [`TenantQuery`].
///
/// The write-side cascade consumes this to FIT its calibrator: a curator's non-retracted
/// `Manual` row is the ground truth, and a prior automated binding is a labeled sample
/// against it. So this read deliberately includes BOTH manual and automated rows
/// (no `join_method` filter) and BOTH retracted and live rows (a retracted automated
/// binding is itself a ground-truth-incorrect sample), leaving the polarity/agreement
/// judgement to `FittedCalibrator::fit`. The window is on `outcome_ts` (a real
/// `DateTime64` column the table is partitioned by) so the read prunes day partitions
/// rather than scanning all feedback history; the bounds are supplied as fractional
/// seconds to match that column. Reads through the re-keyed `FeedbackByTargetId` view so
/// the scan is the cheap, target-sorted one, exactly as `reward_join` does.
///
/// # Errors
/// Returns [`crate::tenant::BuildError`] only on builder misuse; the `Result` surfaces
/// the fail-closed guarantee.
pub fn recent_feedback(p: &RecentFeedbackParams) -> Result<TenantQuery, crate::tenant::BuildError> {
    let body = "\
SELECT
    id,
    tenant_id,
    target_id,
    target_type,
    label_kind,
    metric_name,
    value_type,
    value_bool,
    value_float,
    value_failure_class,
    value_object_key,
    value_content_sha256,
    join_method,
    join_confidence,
    join_version,
    calibration_version,
    source_outcome_id,
    delay_ms,
    retracted,
    dedup_key,
    toUnixTimestamp64Nano(outcome_ts) AS outcome_ts_ns,
    credit_weight,
    contributing_set_id
FROM FeedbackByTargetId
WHERE tenant_id = {tenant}
  AND outcome_ts >= {param}
  AND outcome_ts < {param}
ORDER BY target_id;";

    TenantQueryBuilder::new(Dialect::ClickHouse)
        .tenant(p.tenant.clone())
        .body(body)
        .push(ParamValue::F64(p.outcome_ts_from))
        .push(ParamValue::F64(p.outcome_ts_to))
        .build()
}

/// Parameters for [`recent_feedback`]: one tenant plus an inclusive `outcome_ts`
/// (fractional-seconds) window. Separate from [`RecentWindowParams`] because feedback
/// is partitioned and windowed on the `outcome_ts` `DateTime64` column, not the raw
/// `ts_wall_ns` the append-only Rollout/OutcomeEvent streams carry.
#[derive(Debug, Clone)]
pub struct RecentFeedbackParams {
    /// The tenant whose feedback to read — REQUIRED. The builder fails closed without it.
    pub tenant: TenantId,
    /// Inclusive lower bound of the `outcome_ts` window (fractional seconds since the
    /// epoch). Prunes feedback partitions.
    pub outcome_ts_from: f64,
    /// Exclusive upper bound of the `outcome_ts` window (fractional seconds since the
    /// epoch).
    pub outcome_ts_to: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RewardJoinParams {
        RewardJoinParams {
            tenant: TenantId::new("acme"),
            metric_name: "reward".into(),
            label_kind: "terminal_outcome".into(),
            min_confidence: 0.95,
            outcome_ts_from: 1_700_000_000.0,
            outcome_ts_to: 1_700_086_400.0,
        }
    }

    /// The confidence filter must live in the `ON` clause, not the `WHERE`, so the
    /// LEFT JOIN stays a LEFT JOIN and awaiting-outcome rollouts are not dropped.
    #[test]
    fn confidence_predicate_is_in_the_on_clause() {
        let q = reward_join(&sample()).unwrap();
        // Find the ON section (between "ON" and the final WHERE) and assert the
        // confidence predicate appears there.
        let on_start = q.sql.find("    ON r.id = f.target_id").expect("ON clause");
        let where_start = q.sql[on_start..]
            .find("WHERE r.tenant_id")
            .expect("outer WHERE")
            + on_start;
        let on_clause = &q.sql[on_start..where_start];
        assert!(
            on_clause.contains("f.join_confidence >="),
            "confidence must be in the ON clause: {on_clause}"
        );
    }

    /// The read must surface a `has_outcome` coverage column so a consumer can tell a
    /// rollout that has feedback from one still awaiting an outcome.
    #[test]
    fn has_outcome_coverage_column_present() {
        let q = reward_join(&sample()).unwrap();
        assert!(q.sql.contains("AS has_outcome"), "{}", q.sql);
        // It is a LEFT JOIN (awaiting-outcome rollouts survive).
        assert!(q.sql.contains("LEFT JOIN"), "{}", q.sql);
    }

    /// Latest-wins must partition by the full slot `(target_id, target_type,
    /// label_kind)` ordered by `ts DESC`, so a correction supersedes within its slot
    /// while a different slot coexists.
    #[test]
    fn latest_wins_partitions_by_full_slot() {
        let q = reward_join(&sample()).unwrap();
        assert!(
            q.sql.contains("ROW_NUMBER() OVER"),
            "must resolve latest-wins via ROW_NUMBER: {}",
            q.sql
        );
        assert!(
            q.sql
                .contains("PARTITION BY target_id, target_type, label_kind"),
            "must partition by the full slot: {}",
            q.sql
        );
        // The recency order must break ties on `id_uint` (the full-resolution UUIDv7
        // ordering): `ts` is derived from the id at millisecond resolution, so two
        // feedbacks in the same millisecond tie on `ts` and the newest would be picked
        // nondeterministically without the `id_uint` tiebreak.
        assert!(
            q.sql.contains("ORDER BY ts DESC, id_uint DESC"),
            "latest-wins must tiebreak on id_uint: {}",
            q.sql
        );
        assert!(q.sql.contains("f.rn = 1"), "{}", q.sql);
    }

    /// The LEFT JOIN must enable `join_use_nulls` so an unmatched feedback side is NULL
    /// (not the column-type default), keeping the `has_outcome` coverage column honest.
    #[test]
    fn join_use_nulls_is_enabled() {
        let q = reward_join(&sample()).unwrap();
        assert!(
            q.sql.contains("SETTINGS join_use_nulls = 1"),
            "must enable join_use_nulls for correct LEFT JOIN NULL semantics: {}",
            q.sql
        );
    }

    /// The read must window on `outcome_ts` so it prunes day partitions instead of
    /// scanning all history.
    #[test]
    fn outcome_ts_window_is_present() {
        let q = reward_join(&sample()).unwrap();
        assert!(q.sql.contains("outcome_ts >="), "{}", q.sql);
        assert!(q.sql.contains("outcome_ts <"), "{}", q.sql);
    }

    /// The feedback source must be the re-keyed `FeedbackByTargetId` materialized
    /// view, not the ingestion-sorted base table.
    #[test]
    fn reads_through_by_target_id_view() {
        let q = reward_join(&sample()).unwrap();
        assert!(q.sql.contains("FROM FeedbackByTargetId"), "{}", q.sql);
        // Must NOT read feedback from the base `Feedback` table.
        assert!(!q.sql.contains("FROM Feedback\n"), "{}", q.sql);
    }

    /// The whole read is tenant-scoped through the fail-closed builder: the tenant is
    /// the leading bound parameter and appears as a bound marker on both the
    /// subquery and the outer filter — never interpolated.
    #[test]
    fn tenant_scoped_through_the_builder() {
        let q = reward_join(&sample()).unwrap();
        assert_eq!(q.params[0], ParamValue::Str("acme".into()));
        // The literal tenant string never appears in the SQL text.
        assert!(!q.sql.contains("acme"), "tenant must be bound: {}", q.sql);
        // Both tenant predicates resolved to the leading bound param p0.
        let p0_count = q.sql.matches("{p0:String}").count();
        assert_eq!(p0_count, 2, "both tenant predicates use p0: {}", q.sql);
        // The window bounds and confidence landed as bound float params, not inline.
        assert!(q.params.contains(&ParamValue::F64(0.95)));
        assert!(
            !q.sql.contains("0.95"),
            "confidence must be bound: {}",
            q.sql
        );
    }

    fn recent_window() -> RecentWindowParams {
        RecentWindowParams {
            tenant: TenantId::new("acme"),
            ts_wall_from_ns: 1_700_000_000_000_000_000,
            ts_wall_to_ns: 1_700_000_086_400_000_000,
        }
    }

    /// The recent-rollouts read is tenant-scoped through the fail-closed builder: the
    /// tenant is the leading bound param, the literal never appears inline, and the
    /// window bounds land as bound params — never interpolated.
    #[test]
    fn recent_rollouts_is_tenant_scoped_and_parameterized() {
        let q = recent_rollouts(&recent_window()).unwrap();
        // Reads the Rollout base table (the LEFT side of attribution).
        assert!(q.sql.contains("FROM Rollout"), "{}", q.sql);
        // Tenant is the leading bound param p0; the literal never appears in the SQL.
        assert_eq!(q.params[0], ParamValue::Str("acme".into()));
        assert!(!q.sql.contains("acme"), "tenant must be bound: {}", q.sql);
        assert!(q.sql.contains("tenant_id = {p0:String}"), "{}", q.sql);
        // The window bounds are bound Int64 params, not inline literals.
        assert!(q.sql.contains("ts_wall_ns >= {p1:Int64}"), "{}", q.sql);
        assert!(q.sql.contains("ts_wall_ns < {p2:Int64}"), "{}", q.sql);
        assert!(
            q.params
                .contains(&ParamValue::I64(1_700_000_000_000_000_000))
        );
        assert!(
            !q.sql.contains("1_700_000_000_000_000_000") && !q.sql.contains("1700000000000000000"),
            "window bound must be bound, not inline: {}",
            q.sql
        );
        // It selects the columns the row serializer writes, so the read inverts the
        // write: a parser can rebuild the typed Rollout from these.
        for col in ["boot_id", "mono_ns", "embodiment", "policy_version"] {
            assert!(q.sql.contains(col), "missing column {col}: {}", q.sql);
        }
    }

    /// The recent-outcomes read is tenant-scoped and parameterized the same way, reading
    /// the raw OutcomeEvent log the cascade attributes.
    #[test]
    fn recent_outcomes_is_tenant_scoped_and_parameterized() {
        let q = recent_outcomes(&recent_window()).unwrap();
        assert!(q.sql.contains("FROM OutcomeEvent"), "{}", q.sql);
        assert_eq!(q.params[0], ParamValue::Str("acme".into()));
        assert!(!q.sql.contains("acme"), "tenant must be bound: {}", q.sql);
        assert!(q.sql.contains("tenant_id = {p0:String}"), "{}", q.sql);
        assert!(q.sql.contains("ts_wall_ns >= {p1:Int64}"), "{}", q.sql);
        assert!(q.sql.contains("ts_wall_ns < {p2:Int64}"), "{}", q.sql);
        // Carries the fields needed to attribute: the clock spine, the kind, and any
        // explicit rollout id.
        for col in ["boot_id", "mono_ns", "outcome_kind", "explicit_rollout_id"] {
            assert!(q.sql.contains(col), "missing column {col}: {}", q.sql);
        }
    }

    /// The recent-feedback read is tenant-scoped and parameterized, windows on
    /// `outcome_ts` (so it prunes partitions), and reads through the re-keyed
    /// `FeedbackByTargetId` view exactly as the canonical join does.
    #[test]
    fn recent_feedback_is_tenant_scoped_windowed_and_uses_the_view() {
        let q = recent_feedback(&RecentFeedbackParams {
            tenant: TenantId::new("acme"),
            outcome_ts_from: 1_700_000_000.0,
            outcome_ts_to: 1_700_086_400.0,
        })
        .unwrap();
        // Reads the re-keyed view, not the ingestion-sorted base table.
        assert!(q.sql.contains("FROM FeedbackByTargetId"), "{}", q.sql);
        assert!(!q.sql.contains("FROM Feedback\n"), "{}", q.sql);
        // Tenant leading bound param; literal never inline.
        assert_eq!(q.params[0], ParamValue::Str("acme".into()));
        assert!(!q.sql.contains("acme"), "tenant must be bound: {}", q.sql);
        assert!(q.sql.contains("tenant_id = {p0:String}"), "{}", q.sql);
        // Windowed on outcome_ts via bound Float64 params.
        assert!(q.sql.contains("outcome_ts >= {p1:Float64}"), "{}", q.sql);
        assert!(q.sql.contains("outcome_ts < {p2:Float64}"), "{}", q.sql);
        assert!(q.params.contains(&ParamValue::F64(1_700_000_000.0)));
        // No join_method / retracted filter: the fit needs BOTH manual (truth) and
        // automated (sample) rows, and a retracted row is itself a labeled sample.
        assert!(
            !q.sql.contains("join_method ="),
            "feedback read must not pre-filter by method: {}",
            q.sql
        );
        assert!(
            !q.sql.contains("retracted ="),
            "feedback read must not pre-filter by retracted: {}",
            q.sql
        );
        // Surfaces the integer-nanosecond outcome stamp the typed Feedback carries, so a
        // parser reconstructs `outcome_ts_ns` exactly rather than re-deriving from a float.
        assert!(
            q.sql
                .contains("toUnixTimestamp64Nano(outcome_ts) AS outcome_ts_ns"),
            "{}",
            q.sql
        );
    }
}
