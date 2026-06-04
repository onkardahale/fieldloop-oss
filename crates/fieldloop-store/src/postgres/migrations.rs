//! Idempotent Postgres control-plane migrations, plus per-tenant row-level-security.
//!
//! Same shape as the ClickHouse side: each migration is a Rust value implementing
//! [`Migration`], its [`Migration::up_sql`] returns idempotent DDL (`CREATE ...
//! IF NOT EXISTS` / `... IF NOT EXISTS`), and [`all`] returns the ordered list. The
//! closed-loop check only asserts on the generated SQL; nothing here connects to a
//! database.
//!
//! Every tenant-scoped table carries a `tenant_id` column and a **row-level-security
//! policy** keyed on a `current_setting('fieldloop.tenant_id')` session GUC, so
//! isolation is enforced by Postgres itself — not by a hand-written `WHERE` predicate
//! a developer might forget. `FORCE ROW LEVEL SECURITY` makes the policy apply even
//! to the table owner, so no connection can read across tenants by accident.
//!
//! OPERATIONAL SAFETY — RLS is bypassed by privileged roles: a Postgres `SUPERUSER` or
//! `BYPASSRLS` role IGNORES row-level security entirely, even on a table with `FORCE
//! ROW LEVEL SECURITY`. For such a role the tenant policy is silently void and one
//! tenant can read another's rows with no error. Therefore the application /
//! tenant-scoped connection MUST be a NON-superuser, NON-`BYPASSRLS` role; otherwise
//! per-tenant isolation evaporates invisibly. (Observed concretely: connecting as the
//! bootstrap superuser made tenant A see 2 rows where 1 was expected.) Run migrations
//! as the owner/superuser, but do all per-tenant data access as a plain login role.

/// One ordered, idempotent Postgres migration.
///
/// Identified by [`Migration::name`] (the applied-set ledger key) and producing
/// idempotent DDL via [`Migration::up_sql`], so a re-run over an already-migrated
/// database is a safe no-op.
pub trait Migration {
    /// Stable, unique migration name; doubles as the applied-set ledger key, so it
    /// must never change once shipped.
    fn name(&self) -> &'static str;

    /// The idempotent DDL for this migration.
    fn up_sql(&self) -> String;
}

/// The ordered list of all Postgres migrations. Tables come before the RLS policies
/// that reference them, so a real runner applies these front-to-back.
#[must_use]
pub fn all() -> Vec<Box<dyn Migration>> {
    vec![
        Box::new(M0001PolicyRegistry),
        Box::new(M0002DeploymentLedger),
        Box::new(M0003PendingOutcomeQueue),
        Box::new(M0004IngestIdempotency),
        Box::new(M0005EpisodeRollup),
        Box::new(M0006RowLevelSecurity),
        Box::new(M0007DeployAttempts),
        Box::new(M0008RolloutState),
        Box::new(M0009DeployAttemptsEvidence),
    ]
}

/// Migration 0001 — the policy-version **registry**: the immutable record of each
/// trained policy and its provenance, the anchor for bidirectional provenance.
#[derive(Debug)]
pub struct M0001PolicyRegistry;

impl Migration for M0001PolicyRegistry {
    fn name(&self) -> &'static str {
        "0001_policy_registry"
    }

    fn up_sql(&self) -> String {
        // `policy_version` is a content-hash string (its identity IS the hash), so it
        // is the primary key rather than a surrogate UUID. `IF NOT EXISTS` is what
        // makes this idempotent.
        "CREATE TABLE IF NOT EXISTS policy_registry (\n\
         \x20   tenant_id        text        NOT NULL,\n\
         \x20   policy_version   text        NOT NULL,\n\
         \x20   artifact_format  text        NOT NULL,\n\
         \x20   weights_sha256   text        NOT NULL,\n\
         \x20   provenance       jsonb       NOT NULL DEFAULT '{}'::jsonb,\n\
         \x20   created_at       timestamptz NOT NULL DEFAULT now(),\n\
         \x20   PRIMARY KEY (tenant_id, policy_version)\n\
         );"
        .to_string()
    }
}

/// Migration 0002 — the **deployment ledger**: which policy was actually deployed to
/// which robot and when. The server-side truth a robot's self-reported policy version
/// is reconciled against at ingest.
#[derive(Debug)]
pub struct M0002DeploymentLedger;

impl Migration for M0002DeploymentLedger {
    fn name(&self) -> &'static str {
        "0002_deployment_ledger"
    }

    fn up_sql(&self) -> String {
        "CREATE TABLE IF NOT EXISTS deployment_ledger (\n\
         \x20   id             uuid        PRIMARY KEY,\n\
         \x20   tenant_id      text        NOT NULL,\n\
         \x20   robot_id       text        NOT NULL,\n\
         \x20   policy_version text        NOT NULL,\n\
         \x20   model_hash     text        NOT NULL,\n\
         \x20   deployed_at    timestamptz NOT NULL DEFAULT now(),\n\
         \x20   retired_at     timestamptz\n\
         );\n\
         CREATE INDEX IF NOT EXISTS deployment_ledger_lookup\n\
         \x20   ON deployment_ledger (tenant_id, robot_id, deployed_at DESC);"
            .to_string()
    }
}

/// Migration 0003 — the **pending-outcome work queue**: mutable attribution
/// bookkeeping (claim/lease/retry) for outcomes awaiting a binding. Kept here, NOT in
/// the immutable ClickHouse outcome log, so the log stays append-only.
#[derive(Debug)]
pub struct M0003PendingOutcomeQueue;

impl Migration for M0003PendingOutcomeQueue {
    fn name(&self) -> &'static str {
        "0003_pending_outcome_queue"
    }

    fn up_sql(&self) -> String {
        // `status` + `locked_until` let a worker lease an item; `attempts` bounds
        // retries. A partial index over not-done rows keeps the claim scan cheap.
        "CREATE TABLE IF NOT EXISTS pending_outcome (\n\
         \x20   outcome_id    uuid        PRIMARY KEY,\n\
         \x20   tenant_id     text        NOT NULL,\n\
         \x20   robot_id      text        NOT NULL,\n\
         \x20   status        text        NOT NULL DEFAULT 'pending',\n\
         \x20   attempts      int         NOT NULL DEFAULT 0,\n\
         \x20   locked_until  timestamptz,\n\
         \x20   enqueued_at   timestamptz NOT NULL DEFAULT now()\n\
         );\n\
         CREATE INDEX IF NOT EXISTS pending_outcome_claimable\n\
         \x20   ON pending_outcome (tenant_id, status, locked_until)\n\
         \x20   WHERE status <> 'done';"
            .to_string()
    }
}

/// Migration 0004 — the **ingest idempotency** table: records which upload batch ids
/// have already been ingested, so a retried upload is a no-op rather than a
/// double-write.
#[derive(Debug)]
pub struct M0004IngestIdempotency;

impl Migration for M0004IngestIdempotency {
    fn name(&self) -> &'static str {
        "0004_ingest_idempotency"
    }

    fn up_sql(&self) -> String {
        // The `(tenant_id, batch_id)` primary key is the dedup key: a second insert of
        // the same batch conflicts and is skipped (`ON CONFLICT DO NOTHING` at the use
        // site).
        "CREATE TABLE IF NOT EXISTS ingest_idempotency (\n\
         \x20   tenant_id   text        NOT NULL,\n\
         \x20   batch_id    text        NOT NULL,\n\
         \x20   row_count   int         NOT NULL,\n\
         \x20   seen_at     timestamptz NOT NULL DEFAULT now(),\n\
         \x20   PRIMARY KEY (tenant_id, batch_id)\n\
         );"
        .to_string()
    }
}

/// Migration 0005 — the **episode/outcome rollup**: the per-episode aggregate
/// (return, success, step count). In Postgres — NOT a ClickHouse materialized view —
/// because it is updated as late feedback arrives and must support read-your-write.
#[derive(Debug)]
pub struct M0005EpisodeRollup;

impl Migration for M0005EpisodeRollup {
    fn name(&self) -> &'static str {
        "0005_episode_rollup"
    }

    fn up_sql(&self) -> String {
        "CREATE TABLE IF NOT EXISTS episode_rollup (\n\
         \x20   tenant_id      text        NOT NULL,\n\
         \x20   episode_id     uuid        NOT NULL,\n\
         \x20   policy_version text,\n\
         \x20   step_count     int         NOT NULL DEFAULT 0,\n\
         \x20   episode_return double precision,\n\
         \x20   success        boolean,\n\
         \x20   updated_at     timestamptz NOT NULL DEFAULT now(),\n\
         \x20   PRIMARY KEY (tenant_id, episode_id)\n\
         );"
        .to_string()
    }
}

/// Migration 0006 — per-tenant **row-level-security** policies on every tenant-scoped
/// table. Isolation is enforced by Postgres against the `fieldloop.tenant_id` session
/// GUC, so a connection can only ever see rows for its own tenant — even a buggy or
/// missing `WHERE` predicate cannot leak across tenants.
#[derive(Debug)]
pub struct M0006RowLevelSecurity;

impl M0006RowLevelSecurity {
    /// The tenant-scoped tables RLS is applied to. Each gets `ENABLE` + `FORCE` RLS
    /// and a policy comparing `tenant_id` to the session GUC.
    const TABLES: &'static [&'static str] = &[
        "policy_registry",
        "deployment_ledger",
        "pending_outcome",
        "ingest_idempotency",
        "episode_rollup",
    ];
}

impl Migration for M0006RowLevelSecurity {
    fn name(&self) -> &'static str {
        "0006_row_level_security"
    }

    fn up_sql(&self) -> String {
        let mut sql = String::new();
        for t in Self::TABLES {
            // ENABLE turns RLS on; FORCE makes it apply even to the table owner (so no
            // privileged connection bypasses it). `DROP POLICY IF EXISTS` before
            // `CREATE POLICY` makes the whole statement idempotent (Postgres has no
            // `CREATE POLICY IF NOT EXISTS`). `current_setting(..., true)` returns NULL
            // rather than erroring when the GUC is unset, and a NULL comparison fails
            // closed — no tenant set => no rows visible.
            sql.push_str(&format!(
                "ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;\n\
                 ALTER TABLE {t} FORCE ROW LEVEL SECURITY;\n\
                 DROP POLICY IF EXISTS {t}_tenant_isolation ON {t};\n\
                 CREATE POLICY {t}_tenant_isolation ON {t}\n\
                 \x20   USING (tenant_id = current_setting('fieldloop.tenant_id', true))\n\
                 \x20   WITH CHECK (tenant_id = current_setting('fieldloop.tenant_id', true));\n"
            ));
        }
        sql
    }
}

/// Migration 0007 — the **deploy-attempt audit**: every attempt to send a policy
/// toward a robot, including refused attempts. The deployment ledger intentionally
/// remains only the set of policies that actually reached a robot.
#[derive(Debug)]
pub struct M0007DeployAttempts;

impl Migration for M0007DeployAttempts {
    fn name(&self) -> &'static str {
        "0007_deploy_attempts"
    }

    fn up_sql(&self) -> String {
        "CREATE TABLE IF NOT EXISTS deploy_attempts (\n\
         \x20   id                 uuid        PRIMARY KEY,\n\
         \x20   tenant_id          text        NOT NULL,\n\
         \x20   robot_id           text,\n\
         \x20   policy_version     text        NOT NULL,\n\
         \x20   dataset_commit_id  text        NOT NULL,\n\
         \x20   weights_sha256     text        NOT NULL,\n\
         \x20   include_synthetic  boolean     NOT NULL,\n\
         \x20   only_failures      boolean     NOT NULL,\n\
         \x20   gate_verdict       text        NOT NULL,\n\
         \x20   deployed           boolean     NOT NULL,\n\
         \x20   refused            boolean     NOT NULL,\n\
         \x20   reason             text,\n\
         \x20   attempted_at       timestamptz NOT NULL DEFAULT now()\n\
         );\n\
         CREATE INDEX IF NOT EXISTS deploy_attempts_lookup\n\
         \x20   ON deploy_attempts (tenant_id, attempted_at DESC);\n\
         ALTER TABLE deploy_attempts ENABLE ROW LEVEL SECURITY;\n\
         ALTER TABLE deploy_attempts FORCE ROW LEVEL SECURITY;\n\
         DROP POLICY IF EXISTS deploy_attempts_tenant_isolation ON deploy_attempts;\n\
         CREATE POLICY deploy_attempts_tenant_isolation ON deploy_attempts\n\
         \x20   USING (tenant_id = current_setting('fieldloop.tenant_id', true))\n\
         \x20   WITH CHECK (tenant_id = current_setting('fieldloop.tenant_id', true));"
            .to_string()
    }
}

/// Migration 0008 — the **per-robot rollout state** plus an **append-only transition
/// audit** that together turn the staged shadow→canary→fleet rollout into durable,
/// scopable truth.
///
/// `rollout_state` records, for each `(tenant_id, robot_id)`, which `policy_version` that
/// robot is currently assigned and at which `stage` — the row the weights-pull endpoint
/// reads to tell a robot what to run, and the row the platform updates as the FSM
/// promotes or rolls back. One row per robot (the `(tenant_id, robot_id)` primary key)
/// because a robot is on exactly one assignment at a time; an UPDATE moves it.
///
/// `rollout_transitions` is append-only: every promotion AND every rollback writes one
/// row recording `from_stage`, `to_stage`, the `trigger` that drove it (e.g. the live
/// safety signal or a blocked gate), and the `policy_version` in play. It exists so an
/// operator can reconstruct exactly why a robot moved cohorts — a rollback with no record
/// of what caused it would be unauditable. There is no UPDATE/DELETE here by design; the
/// history is the evidence.
#[derive(Debug)]
pub struct M0008RolloutState;

impl Migration for M0008RolloutState {
    fn name(&self) -> &'static str {
        "0008_rollout_state"
    }

    fn up_sql(&self) -> String {
        // Both tables are `CREATE TABLE IF NOT EXISTS` (idempotent re-run is a no-op) and
        // carry the same fail-closed RLS as every other tenant-scoped table: ENABLE +
        // FORCE so even the owner is scoped, and a policy comparing `tenant_id` to the
        // `fieldloop.tenant_id` session GUC (read with `, true` so an unset GUC yields NULL
        // and the comparison fails closed — no tenant set means no rows visible).
        // `DROP POLICY IF EXISTS` before `CREATE POLICY` keeps the policy creation
        // idempotent since Postgres has no `CREATE POLICY IF NOT EXISTS`.
        "CREATE TABLE IF NOT EXISTS rollout_state (\n\
         \x20   tenant_id      text        NOT NULL,\n\
         \x20   robot_id       text        NOT NULL,\n\
         \x20   policy_version text        NOT NULL,\n\
         \x20   stage          text        NOT NULL,\n\
         \x20   updated_at     timestamptz NOT NULL DEFAULT now(),\n\
         \x20   PRIMARY KEY (tenant_id, robot_id)\n\
         );\n\
         CREATE TABLE IF NOT EXISTS rollout_transitions (\n\
         \x20   id             uuid        PRIMARY KEY,\n\
         \x20   tenant_id      text        NOT NULL,\n\
         \x20   robot_id       text        NOT NULL,\n\
         \x20   from_stage     text        NOT NULL,\n\
         \x20   to_stage       text        NOT NULL,\n\
         \x20   trigger        text        NOT NULL,\n\
         \x20   policy_version text        NOT NULL,\n\
         \x20   at             timestamptz NOT NULL DEFAULT now()\n\
         );\n\
         CREATE INDEX IF NOT EXISTS rollout_transitions_lookup\n\
         \x20   ON rollout_transitions (tenant_id, robot_id, at DESC);\n\
         ALTER TABLE rollout_state ENABLE ROW LEVEL SECURITY;\n\
         ALTER TABLE rollout_state FORCE ROW LEVEL SECURITY;\n\
         DROP POLICY IF EXISTS rollout_state_tenant_isolation ON rollout_state;\n\
         CREATE POLICY rollout_state_tenant_isolation ON rollout_state\n\
         \x20   USING (tenant_id = current_setting('fieldloop.tenant_id', true))\n\
         \x20   WITH CHECK (tenant_id = current_setting('fieldloop.tenant_id', true));\n\
         ALTER TABLE rollout_transitions ENABLE ROW LEVEL SECURITY;\n\
         ALTER TABLE rollout_transitions FORCE ROW LEVEL SECURITY;\n\
         DROP POLICY IF EXISTS rollout_transitions_tenant_isolation ON rollout_transitions;\n\
         CREATE POLICY rollout_transitions_tenant_isolation ON rollout_transitions\n\
         \x20   USING (tenant_id = current_setting('fieldloop.tenant_id', true))\n\
         \x20   WITH CHECK (tenant_id = current_setting('fieldloop.tenant_id', true));"
            .to_string()
    }
}

/// Migration 0009 — record the evidence bundle behind each deploy decision.
///
/// A deploy attempt stored the verdict + reason but not the EVIDENCE that drove it (which bindings
/// were confirmed/gate-eligible, which were investigation hypotheses, which were excluded and why).
/// Without it an auditor cannot reconstruct *why* a past rollout was blocked or approved. This
/// column carries the classified evidence bundle as JSON, so every historical decision is auditable.
/// Old rows default to empty. `ADD COLUMN IF NOT EXISTS` keeps the migration idempotent.
#[derive(Debug)]
pub struct M0009DeployAttemptsEvidence;

impl Migration for M0009DeployAttemptsEvidence {
    fn name(&self) -> &'static str {
        "0009_deploy_attempts_evidence"
    }

    fn up_sql(&self) -> String {
        "ALTER TABLE deploy_attempts ADD COLUMN IF NOT EXISTS evidence_json text DEFAULT '';"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every migration's DDL is idempotent (re-runnable). Table DDL uses
    /// `IF NOT EXISTS`; the RLS migration uses `DROP POLICY IF EXISTS` before each
    /// `CREATE POLICY`.
    #[test]
    fn every_migration_is_idempotent() {
        for m in all() {
            let sql = m.up_sql();
            let idempotent = sql.contains("IF NOT EXISTS") || sql.contains("DROP POLICY IF EXISTS");
            assert!(idempotent, "migration {} must be idempotent", m.name());
        }
    }

    /// All control-plane tables are present in the migration set.
    #[test]
    fn all_control_plane_tables_present() {
        let joined: String = all().iter().map(|m| m.up_sql()).collect();
        for table in [
            "policy_registry",
            "deployment_ledger",
            "pending_outcome",
            "ingest_idempotency",
            "episode_rollup",
            "deploy_attempts",
            "rollout_state",
            "rollout_transitions",
        ] {
            assert!(
                joined.contains(&format!("CREATE TABLE IF NOT EXISTS {table}")),
                "missing table {table}"
            );
        }
    }

    /// The rollout-state and transition tables are added after the original RLS
    /// migration, so their own migration must carry the same fail-closed tenant policy
    /// on BOTH tables (state is mutated per robot; transitions are append-only) — without
    /// it, one tenant's connection could read or halt another tenant's rollout.
    #[test]
    fn rollout_state_rls_is_self_contained() {
        let sql = M0008RolloutState.up_sql();
        for t in ["rollout_state", "rollout_transitions"] {
            assert!(sql.contains(&format!("ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")));
            assert!(sql.contains(&format!("ALTER TABLE {t} FORCE ROW LEVEL SECURITY")));
            assert!(sql.contains(&format!("CREATE POLICY {t}_tenant_isolation ON {t}")));
        }
        assert!(sql.contains("current_setting('fieldloop.tenant_id', true)"));
        // One row per robot: the primary key is the (tenant, robot) pair, so an UPDATE
        // moves a robot's assignment rather than accumulating duplicate rows.
        assert!(sql.contains("PRIMARY KEY (tenant_id, robot_id)"));
    }

    /// Deploy attempts are added after the original RLS migration, so their own
    /// migration must carry the same fail-closed tenant policy.
    #[test]
    fn deploy_attempts_rls_is_self_contained() {
        let sql = M0007DeployAttempts.up_sql();
        assert!(sql.contains("ALTER TABLE deploy_attempts ENABLE ROW LEVEL SECURITY"));
        assert!(sql.contains("ALTER TABLE deploy_attempts FORCE ROW LEVEL SECURITY"));
        assert!(sql.contains("CREATE POLICY deploy_attempts_tenant_isolation ON deploy_attempts"));
        assert!(sql.contains("current_setting('fieldloop.tenant_id', true)"));
    }

    /// Every tenant-scoped table gets a row-level-security policy that enforces
    /// per-tenant isolation against the session GUC, plus FORCE so the owner can't
    /// bypass it.
    #[test]
    fn rls_enforces_per_tenant_isolation() {
        let sql = M0006RowLevelSecurity.up_sql();
        for t in M0006RowLevelSecurity::TABLES {
            assert!(sql.contains(&format!("ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")));
            assert!(sql.contains(&format!("ALTER TABLE {t} FORCE ROW LEVEL SECURITY")));
            assert!(sql.contains(&format!("CREATE POLICY {t}_tenant_isolation ON {t}")));
        }
        assert!(sql.contains("current_setting('fieldloop.tenant_id', true)"));
    }

    /// Migration names are unique (the applied-set ledger key).
    #[test]
    fn migration_names_are_unique() {
        let names: Vec<&str> = all().iter().map(|m| m.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }
}
