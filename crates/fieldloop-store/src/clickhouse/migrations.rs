//! Idempotent ClickHouse schema migrations as plain Rust structs.
//!
//! There is no external migration tool: each migration is a Rust value implementing
//! [`Migration`], and its [`Migration::up_sql`] returns a `CREATE ... IF NOT EXISTS`
//! statement so applying it twice is a no-op (idempotent). [`all`] returns the
//! ordered list; a startup path would run them in order, but the closed-loop check
//! only asserts on the generated SQL — nothing here connects to a database.
//!
//! The schema follows the append-heavy event+feedback shape:
//!   * `Rollout` and `OutcomeEvent` are immutable base tables.
//!   * Per-grain `*Feedback` tables are append-only and carry the attribution slot
//!     `(target_id, target_type, label_kind)` plus the `dedup_key`, `join_confidence`,
//!     and `outcome_ts` columns the canonical read needs.
//!   * `*ByTargetId` / `*ById` materialized views re-key the same rows for cheap
//!     point/JOIN lookups. Because ClickHouse stores `UUID` big-endian — which does
//!     NOT preserve UUIDv7 chronological order — any id used in an `ORDER BY` is
//!     stored as `UInt128` (`toUInt128` on write) and read back with the
//!     `uint_to_uuid` SQL function created by the first migration.
//!
//! Rollout/Feedback are `PARTITION BY` a day bucket on their driving timestamp so a
//! time-windowed read prunes whole partitions instead of scanning all history, and
//! an operator can `DROP PARTITION` to age data out.

/// One ordered, idempotent ClickHouse schema migration.
///
/// A migration is identified by [`Migration::name`] (recorded in an applied-set
/// ledger by a real runner) and produces its DDL via [`Migration::up_sql`]. The DDL
/// MUST be idempotent — every statement uses `CREATE ... IF NOT EXISTS` — so a
/// re-run over an already-migrated database changes nothing and never errors.
pub trait Migration {
    /// Stable, unique name for this migration. Used as the ledger key that records
    /// "this migration has been applied", so it must never change once shipped.
    fn name(&self) -> &'static str;

    /// The idempotent DDL for this migration. Every statement is
    /// `CREATE ... IF NOT EXISTS`, so applying it more than once is a safe no-op.
    fn up_sql(&self) -> String;
}

/// The ordered list of all ClickHouse migrations. Order matters: the `uint_to_uuid`
/// function and base tables must exist before the materialized views that reference
/// them. A real runner applies these front-to-back and skips any already in the
/// applied-set ledger.
#[must_use]
pub fn all() -> Vec<Box<dyn Migration>> {
    vec![
        Box::new(M0000Functions),
        Box::new(M0001Rollout),
        Box::new(M0002OutcomeEvent),
        Box::new(M0003Feedback),
        Box::new(M0004RolloutLookups),
        Box::new(M0005FeedbackByTargetId),
        Box::new(M0006RolloutProvenance),
        Box::new(M0007RolloutClock),
        Box::new(M0008RolloutServerAnchor),
        Box::new(M0009FeedbackDistributedCredit),
        Box::new(M0010OutcomeSpatialCausal),
        Box::new(M0011RolloutSpatialCausal),
        Box::new(M0012RolloutSignals),
    ]
}

/// Migration 0000 — the `uint_to_uuid` read helper.
///
/// Re-keyed lookup tables store ids as `UInt128` (so a UUIDv7 sort key orders
/// chronologically). This function converts that `UInt128` back to a `UUID` on read,
/// so a query against a `*ById` table can still return a normal UUID string.
#[derive(Debug)]
pub struct M0000Functions;

impl Migration for M0000Functions {
    fn name(&self) -> &'static str {
        "0000_functions"
    }

    fn up_sql(&self) -> String {
        // `IF NOT EXISTS` makes re-creating the function a no-op (idempotent). The
        // body reinterprets the 128-bit integer's bytes as a UUID.
        "CREATE FUNCTION IF NOT EXISTS uint_to_uuid AS (x) -> reinterpretAsUUID(\
         reverse(reinterpretAsString(x)));"
            .to_string()
    }
}

/// Migration 0001 — the `Rollout` base table (immutable deployed-policy inference
/// log).
#[derive(Debug)]
pub struct M0001Rollout;

impl Migration for M0001Rollout {
    fn up_sql(&self) -> String {
        // `id` stays `UUID` here: this base table is keyed on (tenant, policy,
        // episode), NOT on the id, so it does not need the UInt128 chronological-sort
        // fix. The `timestamp` is derived from the v7 id and the table is partitioned
        // by its day so a windowed read prunes partitions.
        "CREATE TABLE IF NOT EXISTS Rollout (\n\
         \x20   id UUID,\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   robot_id LowCardinality(String),\n\
         \x20   episode_id UUID,\n\
         \x20   step_index UInt32,\n\
         \x20   policy_version LowCardinality(String),\n\
         \x20   model_hash String,\n\
         \x20   trust LowCardinality(String),\n\
         \x20   embodiment LowCardinality(String),\n\
         \x20   task_id LowCardinality(String),\n\
         \x20   eval_run_id UUID,\n\
         \x20   arm_label LowCardinality(String),\n\
         \x20   observation_ref String,\n\
         \x20   action_ref String,\n\
         \x20   context String,\n\
         \x20   schema_conformance LowCardinality(String),\n\
         \x20   inference_us UInt32,\n\
         \x20   tags Map(String, String),\n\
         \x20   timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)\n\
         ) ENGINE = MergeTree\n\
         PARTITION BY toYYYYMMDD(timestamp)\n\
         ORDER BY (tenant_id, policy_version, episode_id, id);"
            .to_string()
    }

    fn name(&self) -> &'static str {
        "0001_rollout"
    }
}

/// Migration 0002 — the `OutcomeEvent` base table (immutable raw outcome log).
#[derive(Debug)]
pub struct M0002OutcomeEvent;

impl Migration for M0002OutcomeEvent {
    fn name(&self) -> &'static str {
        "0002_outcome_event"
    }

    fn up_sql(&self) -> String {
        // Raw outcomes before attribution. `explicit_rollout_id` is nullable because
        // most outcomes arrive with no known target (the implicit case the cascade
        // must infer). Partitioned by day on the derived timestamp for pruning.
        "CREATE TABLE IF NOT EXISTS OutcomeEvent (\n\
         \x20   id UUID,\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   robot_id LowCardinality(String),\n\
         \x20   boot_id UUID,\n\
         \x20   mono_ns UInt64,\n\
         \x20   ts_wall_ns UInt64,\n\
         \x20   outcome_kind LowCardinality(String),\n\
         \x20   explicit_rollout_id Nullable(UUID),\n\
         \x20   trust LowCardinality(String),\n\
         \x20   payload String,\n\
         \x20   timestamp DateTime MATERIALIZED UUIDv7ToDateTime(id)\n\
         ) ENGINE = MergeTree\n\
         PARTITION BY toYYYYMMDD(timestamp)\n\
         ORDER BY (tenant_id, robot_id, boot_id, mono_ns);"
            .to_string()
    }
}

/// Migration 0003 — the per-grain `Feedback` base table (append-only attributed
/// bindings).
#[derive(Debug)]
pub struct M0003Feedback;

impl Migration for M0003Feedback {
    fn name(&self) -> &'static str {
        "0003_feedback"
    }

    fn up_sql(&self) -> String {
        // The attribution slot is `(target_id, target_type, label_kind)`: latest-wins
        // supersession is resolved per-slot, so a correction supersedes within its
        // slot while a different slot on the same target coexists. `dedup_key` is a
        // SEPARATE string column (the row `id` is a fresh UUIDv7, never hash-derived).
        // `outcome_ts` is a real DateTime64 column (not derived from the id) so the
        // table can be partitioned by it and a read can window on it. `value_*`
        // columns are a typed union projected from the Rust `FeedbackValue` sum type.
        "CREATE TABLE IF NOT EXISTS Feedback (\n\
         \x20   id UUID,\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   target_id UUID,\n\
         \x20   target_type Enum8('rollout' = 1, 'episode' = 2),\n\
         \x20   label_kind LowCardinality(String),\n\
         \x20   metric_name LowCardinality(String),\n\
         \x20   value_type Enum8('boolean' = 1, 'float' = 2, 'failure_class' = 3, 'demonstration_ref' = 4),\n\
         \x20   value_bool Nullable(Bool),\n\
         \x20   value_float Nullable(Float32),\n\
         \x20   value_failure_class LowCardinality(String),\n\
         \x20   value_object_key String,\n\
         \x20   value_content_sha256 Nullable(String),\n\
         \x20   join_method LowCardinality(String),\n\
         \x20   join_confidence Float32,\n\
         \x20   join_version LowCardinality(String),\n\
         \x20   calibration_version LowCardinality(String),\n\
         \x20   source_outcome_id Nullable(UUID),\n\
         \x20   delay_ms Nullable(Int64),\n\
         \x20   retracted Bool,\n\
         \x20   dedup_key String,\n\
         \x20   outcome_ts DateTime64(9),\n\
         \x20   ts DateTime64(9) MATERIALIZED toDateTime64(UUIDv7ToDateTime(id), 9)\n\
         ) ENGINE = MergeTree\n\
         PARTITION BY toYYYYMMDD(outcome_ts)\n\
         ORDER BY (tenant_id, target_id, target_type, label_kind);"
            .to_string()
    }
}

/// Migration 0004 — the `RolloutById` / `RolloutByEpisodeId` lookup tables and the
/// materialized views that populate them.
#[derive(Debug)]
pub struct M0004RolloutLookups;

impl Migration for M0004RolloutLookups {
    fn name(&self) -> &'static str {
        "0004_rollout_lookups"
    }

    fn up_sql(&self) -> String {
        // Re-keyed lookups for "find one rollout by id" and "whole trajectory in
        // order". `id` is stored as `id_uint UInt128` because it is the sort key and a
        // UUIDv7 stored as `UUID` does not order chronologically; read it back with
        // `uint_to_uuid`. The MV is a write-time trigger: every Rollout insert fans a
        // re-keyed copy into these tables, so no batch job is needed.
        "CREATE TABLE IF NOT EXISTS RolloutById (\n\
         \x20   id_uint UInt128,\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   episode_id UUID,\n\
         \x20   policy_version LowCardinality(String)\n\
         ) ENGINE = MergeTree ORDER BY id_uint;\n\
         \n\
         CREATE MATERIALIZED VIEW IF NOT EXISTS RolloutByIdMV TO RolloutById AS\n\
         SELECT toUInt128(id) AS id_uint, tenant_id, episode_id, policy_version\n\
         FROM Rollout;\n\
         \n\
         CREATE TABLE IF NOT EXISTS RolloutByEpisodeId (\n\
         \x20   episode_id_uint UInt128,\n\
         \x20   id_uint UInt128,\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   step_index UInt32\n\
         ) ENGINE = MergeTree ORDER BY (episode_id_uint, id_uint);\n\
         \n\
         CREATE MATERIALIZED VIEW IF NOT EXISTS RolloutByEpisodeIdMV TO RolloutByEpisodeId AS\n\
         SELECT toUInt128(episode_id) AS episode_id_uint, toUInt128(id) AS id_uint, tenant_id, step_index\n\
         FROM Rollout;"
            .to_string()
    }
}

/// Migration 0005 — the `FeedbackByTargetId` lookup table and its materialized view.
#[derive(Debug)]
pub struct M0005FeedbackByTargetId;

impl Migration for M0005FeedbackByTargetId {
    fn name(&self) -> &'static str {
        "0005_feedback_by_target_id"
    }

    fn up_sql(&self) -> String {
        // The canonical join reads feedback through THIS view, not the base table:
        // it is sorted by `target_id` so the rollout->feedback join is a cheap merge.
        // Carries the slot `(target_id, target_type, label_kind)` plus the columns the
        // read needs: `join_confidence` (for the ON-clause predicate), `outcome_ts`
        // (for the time window + partition pruning), and `ts` (for the latest-wins
        // ROW_NUMBER ordering). `id` rides along as `id_uint UInt128` so the
        // recency tiebreak still orders chronologically.
        "CREATE TABLE IF NOT EXISTS FeedbackByTargetId (\n\
         \x20   tenant_id LowCardinality(String),\n\
         \x20   target_id UUID,\n\
         \x20   target_type Enum8('rollout' = 1, 'episode' = 2),\n\
         \x20   label_kind LowCardinality(String),\n\
         \x20   metric_name LowCardinality(String),\n\
         \x20   id_uint UInt128,\n\
         \x20   value_type Enum8('boolean' = 1, 'float' = 2, 'failure_class' = 3, 'demonstration_ref' = 4),\n\
         \x20   value_bool Nullable(Bool),\n\
         \x20   value_float Nullable(Float32),\n\
         \x20   value_failure_class LowCardinality(String),\n\
         \x20   join_method LowCardinality(String),\n\
         \x20   join_confidence Float32,\n\
         \x20   retracted Bool,\n\
         \x20   outcome_ts DateTime64(9),\n\
         \x20   ts DateTime64(9)\n\
         ) ENGINE = MergeTree\n\
         PARTITION BY toYYYYMMDD(outcome_ts)\n\
         ORDER BY (tenant_id, target_id, target_type, label_kind);\n\
         \n\
         CREATE MATERIALIZED VIEW IF NOT EXISTS FeedbackByTargetIdMV TO FeedbackByTargetId AS\n\
         SELECT tenant_id, target_id, target_type, label_kind, metric_name, toUInt128(id) AS id_uint,\n\
         \x20      value_type, value_bool, value_float, value_failure_class,\n\
         \x20      join_method, join_confidence, retracted, outcome_ts, ts\n\
         FROM Feedback;"
            .to_string()
    }
}

/// Migration 0006 — append the real-vs-synthetic provenance columns to `Rollout`.
///
/// An append, not a rewrite: it `ALTER ... ADD COLUMN IF NOT EXISTS`s two columns onto
/// the existing table, so rows already written keep their data and gain the new columns
/// with the `DEFAULT 'real'` / empty-generator values — meaning every pre-existing row
/// reads back as real field evidence. That mirrors the `#[serde(default)]` on the type
/// and is what keeps adding provenance a safe append, not a fleet-wide migration event.
#[derive(Debug)]
pub struct M0006RolloutProvenance;

impl Migration for M0006RolloutProvenance {
    fn name(&self) -> &'static str {
        "0006_rollout_provenance"
    }

    fn up_sql(&self) -> String {
        // `provenance_kind DEFAULT 'real'` so any row predating this column is real;
        // `provenance_generator` empty for real data, carrying the generator name for
        // synthetic so synthetic rows stay auditable and filterable by source.
        "ALTER TABLE Rollout\n\
         \x20   ADD COLUMN IF NOT EXISTS provenance_kind LowCardinality(String) DEFAULT 'real',\n\
         \x20   ADD COLUMN IF NOT EXISTS provenance_generator LowCardinality(String) DEFAULT '';"
            .to_string()
    }
}

/// Migration 0007 — persist the two-clock attribution spine on `Rollout`.
///
/// The `Rollout` base table stored no clock, so a rollout lost its monotonic timeline the
/// moment it was written and skew-free attribution was impossible on the rollout side (the
/// authority survived only on `OutcomeEvent`). These columns mirror `OutcomeEvent`'s clock so a
/// rollout carries the same `(boot_id, mono_ns)` authority plus the advisory `ts_wall_ns`. Old
/// rows predating the columns default to the nil boot and zero clock — an honest "unknown
/// timeline", never a fabricated one.
#[derive(Debug)]
pub struct M0007RolloutClock;

impl Migration for M0007RolloutClock {
    fn name(&self) -> &'static str {
        "0007_rollout_clock"
    }

    fn up_sql(&self) -> String {
        "ALTER TABLE Rollout\n\
         \x20   ADD COLUMN IF NOT EXISTS boot_id UUID DEFAULT '00000000-0000-0000-0000-000000000000',\n\
         \x20   ADD COLUMN IF NOT EXISTS mono_ns UInt64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS ts_wall_ns UInt64 DEFAULT 0;"
            .to_string()
    }
}

/// Migration 0008 — persist the server-trusted anchor on `Rollout`.
///
/// The ingest gateway stamps a `ServerAnchor` (trusted ingest time + the offset that reconciles
/// a robot's monotonic origin to server time), but it was dropped at serialization and never
/// reached storage. These columns carry it so a stored rollout keeps the trusted-time
/// reconciliation handle. Signed `Int64` because the offset can be negative; old rows and
/// unstamped rollouts default to 0 (an honest "no anchor", since a real ingest time is
/// epoch-nanoseconds and never 0).
#[derive(Debug)]
pub struct M0008RolloutServerAnchor;

impl Migration for M0008RolloutServerAnchor {
    fn name(&self) -> &'static str {
        "0008_rollout_server_anchor"
    }

    fn up_sql(&self) -> String {
        "ALTER TABLE Rollout\n\
         \x20   ADD COLUMN IF NOT EXISTS server_ts_ingest_ns Int64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS server_anchor_offset_ns Int64 DEFAULT 0;"
            .to_string()
    }
}

/// Migration 0009 — append the distributed-credit columns to `Feedback`.
///
/// One outcome may now emit N feedback rows whose credit is split across the rollouts
/// in its attribution window. An append, not a rewrite: it `ALTER ... ADD COLUMN IF NOT
/// EXISTS`es two columns so rows already written keep their data and gain the new
/// columns with their defaults — `credit_weight DEFAULT 1.0` so every pre-existing row
/// reads back as the single full-credit binding, and `contributing_set_id` defaulting
/// to the nil UUID meaning "not part of any multi-contributor set". That mirrors the
/// `#[serde(default)]` on the type and keeps adding distributed credit a safe append
/// rather than a fleet-wide migration event.
#[derive(Debug)]
pub struct M0009FeedbackDistributedCredit;

impl Migration for M0009FeedbackDistributedCredit {
    fn name(&self) -> &'static str {
        "0009_feedback_distributed_credit"
    }

    fn up_sql(&self) -> String {
        // `credit_weight DEFAULT 1.0` so any row predating this column is full credit;
        // `contributing_set_id DEFAULT nil-UUID` is the "no group" encoding (a single
        // unambiguous binding, or a pre-field row, belongs to no contributing set).
        "ALTER TABLE Feedback\n\
         \x20   ADD COLUMN IF NOT EXISTS credit_weight Float32 DEFAULT 1.0,\n\
         \x20   ADD COLUMN IF NOT EXISTS contributing_set_id UUID DEFAULT '00000000-0000-0000-0000-000000000000';"
            .to_string()
    }
}

/// Migration 0010 — append the spatial-pose and causal-edge columns to `OutcomeEvent`.
///
/// The two cross-boot inferred tiers need their inputs on the immutable outcome log: the
/// spatial tier binds a co-located rollout by an outcome's pose, and the causal tier
/// follows an outcome's downstream edges to an upstream station. An append, not a
/// rewrite: it `ALTER ... ADD COLUMN IF NOT EXISTS`es the columns so rows already written
/// keep their data and gain the new columns with nil/empty defaults — `has_pose DEFAULT 0`
/// (so a pre-existing row reads back as having NO pose, never a fabricated one at the
/// origin), the pose components defaulting to the nil/identity transform, `frame_id`
/// empty (no comparable frame), and `causal_parents` defaulting to the empty-list JSON
/// string (no causal hint). That mirrors the `#[serde(default)]` on the type and keeps
/// adding the spatial/causal channels a safe append rather than a fleet-wide migration.
#[derive(Debug)]
pub struct M0010OutcomeSpatialCausal;

impl Migration for M0010OutcomeSpatialCausal {
    fn name(&self) -> &'static str {
        "0010_outcome_spatial_causal"
    }

    fn up_sql(&self) -> String {
        // `has_pose DEFAULT 0` is the load-bearing flag: it distinguishes "no pose" from
        // "a pose that happens to be at the origin", so an old row (or a non-localized
        // outcome) is never mistaken for a localized one. The quaternion defaults to the
        // identity (`qw = 1`). `causal_parents` is a JSON-string column defaulting to the
        // empty list, matching the serializer's encoding.
        "ALTER TABLE OutcomeEvent\n\
         \x20   ADD COLUMN IF NOT EXISTS has_pose UInt8 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_x Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_y Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_z Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qw Float64 DEFAULT 1,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qx Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qy Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qz Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS frame_id LowCardinality(String) DEFAULT '',\n\
         \x20   ADD COLUMN IF NOT EXISTS causal_parents String DEFAULT '[]';"
            .to_string()
    }
}

/// Migration 0011 — append the spatial-pose and causal-station columns to `Rollout`.
///
/// The rollout is the BIND TARGET of the two cross-boot inferred tiers: the spatial tier
/// binds a co-located rollout by its pose, and the causal tier binds the rollouts at a
/// named upstream station. So the rollout table must carry the same pose channel as the
/// outcome plus a `station_id`. An append, not a rewrite, with nil/empty defaults
/// matching the type's `#[serde(default)]` — `has_pose DEFAULT 0` so an old row reads
/// back as having NO pose (never a fabricated origin pose), the quaternion defaulting to
/// the identity, and `frame_id`/`station_id` empty (no comparable frame, no station). A
/// pre-existing rollout is simply invisible to the spatial/causal tiers, never bound on
/// invented data.
#[derive(Debug)]
pub struct M0011RolloutSpatialCausal;

impl Migration for M0011RolloutSpatialCausal {
    fn name(&self) -> &'static str {
        "0011_rollout_spatial_causal"
    }

    fn up_sql(&self) -> String {
        "ALTER TABLE Rollout\n\
         \x20   ADD COLUMN IF NOT EXISTS has_pose UInt8 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_x Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_y Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_z Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qw Float64 DEFAULT 1,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qx Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qy Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS pose_qz Float64 DEFAULT 0,\n\
         \x20   ADD COLUMN IF NOT EXISTS frame_id LowCardinality(String) DEFAULT '',\n\
         \x20   ADD COLUMN IF NOT EXISTS station_id LowCardinality(String) DEFAULT '';"
            .to_string()
    }
}

/// Migration 0012 — persist the per-rollout named signal channels.
///
/// `signals` (e.g. `gripper_force`, `perception_confidence`) is stored as a JSON-string column so
/// the diagnostic signal-deviation summary can run over stored rollouts, instead of the field
/// being dropped at the serializer. Old rows default to the empty object.
#[derive(Debug)]
pub struct M0012RolloutSignals;

impl Migration for M0012RolloutSignals {
    fn name(&self) -> &'static str {
        "0012_rollout_signals"
    }

    fn up_sql(&self) -> String {
        "ALTER TABLE Rollout ADD COLUMN IF NOT EXISTS signals String DEFAULT '{}';".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every migration's DDL must be idempotent: it uses `IF NOT EXISTS`, so applying
    /// it twice over an already-migrated database is a safe no-op.
    #[test]
    fn every_migration_is_idempotent() {
        for m in all() {
            assert!(
                m.up_sql().contains("IF NOT EXISTS"),
                "migration {} must be idempotent (CREATE ... IF NOT EXISTS)",
                m.name()
            );
        }
    }

    /// Migration names are unique and stable: they double as the applied-set ledger
    /// key, so a duplicate would mask a later migration.
    #[test]
    fn migration_names_are_unique() {
        let names: Vec<&str> = all().iter().map(|m| m.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate migration name");
    }

    /// Rollout and Feedback must be partitioned by a day bucket on their driving
    /// timestamp so a windowed read prunes whole partitions instead of scanning all
    /// history.
    #[test]
    fn rollout_and_feedback_partition_by_day() {
        assert!(
            M0001Rollout
                .up_sql()
                .contains("PARTITION BY toYYYYMMDD(timestamp)")
        );
        assert!(
            M0003Feedback
                .up_sql()
                .contains("PARTITION BY toYYYYMMDD(outcome_ts)")
        );
    }

    /// The Feedback table must carry the attribution slot columns the latest-wins
    /// read partitions on, plus the separate `dedup_key` string.
    #[test]
    fn feedback_carries_the_attribution_slot_columns() {
        let sql = M0003Feedback.up_sql();
        for col in [
            "target_id",
            "target_type",
            "label_kind",
            "dedup_key",
            "join_confidence",
            "outcome_ts",
        ] {
            assert!(sql.contains(col), "Feedback table is missing column {col}");
        }
    }

    /// A `*ByTargetId` materialized view and the `uint_to_uuid` read helper must
    /// exist, since the canonical join reads through the view and decodes UInt128 ids.
    #[test]
    fn by_target_id_view_and_uint_to_uuid_exist() {
        let joined: String = all().iter().map(|m| m.up_sql()).collect();
        assert!(joined.contains("FeedbackByTargetId"));
        assert!(joined.contains("MATERIALIZED VIEW IF NOT EXISTS FeedbackByTargetIdMV"));
        assert!(joined.contains("uint_to_uuid"));
    }

    /// Re-keyed lookup tables must store the id as `UInt128` with a `toUInt128` on
    /// write, because a UUIDv7 stored as `UUID` does not sort chronologically.
    #[test]
    fn lookup_tables_rekey_ids_to_uint128() {
        let sql = M0004RolloutLookups.up_sql();
        assert!(sql.contains("id_uint UInt128"));
        assert!(sql.contains("toUInt128(id)"));
    }

    /// The distributed-credit migration is an additive ALTER that adds the two new
    /// columns with back-compatible defaults: `credit_weight DEFAULT 1.0` (an old row
    /// reads as full credit) and `contributing_set_id` defaulting to the nil UUID (no
    /// group). Idempotent via `ADD COLUMN IF NOT EXISTS`.
    #[test]
    fn distributed_credit_migration_adds_columns_with_defaults() {
        let sql = M0009FeedbackDistributedCredit.up_sql();
        assert!(sql.contains("ALTER TABLE Feedback"));
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS credit_weight Float32 DEFAULT 1.0"));
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS contributing_set_id UUID"));
        // The "no group" default is the nil UUID, matching the serializer's encoding.
        assert!(sql.contains("'00000000-0000-0000-0000-000000000000'"));
    }

    /// The spatial/causal migration is an additive ALTER on `OutcomeEvent` that adds the
    /// pose columns (with `has_pose DEFAULT 0` so an old row reads back as having no pose,
    /// never a fabricated origin pose), the empty `frame_id`, and the `causal_parents`
    /// JSON-string column defaulting to the empty list. Idempotent via
    /// `ADD COLUMN IF NOT EXISTS`.
    #[test]
    fn outcome_spatial_causal_migration_adds_columns_with_defaults() {
        let sql = M0010OutcomeSpatialCausal.up_sql();
        assert!(sql.contains("ALTER TABLE OutcomeEvent"));
        // `has_pose DEFAULT 0` is the load-bearing "no pose vs origin pose" flag.
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS has_pose UInt8 DEFAULT 0"));
        // The quaternion defaults to the identity rotation (qw = 1).
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS pose_qw Float64 DEFAULT 1"));
        for col in ["pose_x", "pose_y", "pose_z", "frame_id", "causal_parents"] {
            assert!(sql.contains(col), "migration is missing column {col}");
        }
        // Causal edges default to the empty-list JSON string, matching the serializer.
        assert!(sql.contains("causal_parents String DEFAULT '[]'"));
    }

    /// The rollout-side spatial/causal migration adds the bind-target pose columns and
    /// the causal `station_id` to `Rollout`, with the same `has_pose DEFAULT 0` /
    /// identity-quaternion / empty-label back-compat defaults, idempotently.
    #[test]
    fn rollout_spatial_causal_migration_adds_columns_with_defaults() {
        let sql = M0011RolloutSpatialCausal.up_sql();
        assert!(sql.contains("ALTER TABLE Rollout"));
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS has_pose UInt8 DEFAULT 0"));
        assert!(sql.contains("ADD COLUMN IF NOT EXISTS pose_qw Float64 DEFAULT 1"));
        for col in ["pose_x", "frame_id", "station_id"] {
            assert!(sql.contains(col), "migration is missing column {col}");
        }
    }
}
