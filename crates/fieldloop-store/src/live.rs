//! Thin async client wrappers for a live ClickHouse / Postgres.
//!
//! This module is behind the optional `live-db` feature, so the default build (and the
//! closed-loop check) never compiles it and never opens a socket. Everything the
//! clients send — the migration DDL, the row lines, the canonical join SQL — is
//! produced by the database-free modules ([`crate::clickhouse`], [`crate::postgres`],
//! [`crate::tenant`]); a client here only carries those strings over the wire and
//! parses the rows back.
//!
//! ## What each client actually does
//! * [`ClickHouseClient`] POSTs a statement to the ClickHouse HTTP interface (the
//!   statement IS the request body): migration DDL, `INSERT ... FORMAT JSONEachRow`
//!   with the generated row lines, and `SELECT ... FORMAT JSONEachRow` reads whose
//!   response body is parsed one JSON object per line.
//! * [`PostgresClient`] connects with `tokio-postgres`, runs the control-plane DDL,
//!   sets the per-connection tenant GUC (so Postgres row-level security scopes every
//!   query by itself), does an idempotent `ON CONFLICT DO NOTHING` insert, and reads
//!   rows back — proving a connection pinned to tenant A never sees tenant B's rows.
//!
//! The integration tests are `#[ignore]`d and additionally short-circuit unless
//! `CLICKHOUSE_URL` / `DATABASE_URL` are set, so even `cargo test --features live-db`
//! never requires a running database unless an operator opts in.

use serde_json::Value;

/// Errors from the live clients. Transport/protocol failures only — the SQL itself is
/// generated and validated by the database-free modules, so it never fails here for a
/// shape reason.
#[derive(Debug, thiserror::Error)]
pub enum LiveError {
    /// The transport (HTTP / socket) to the database failed, or the server returned a
    /// non-success status. Carries the server's response body when there is one so an
    /// operator sees the actual ClickHouse/Postgres error text, not just "failed".
    #[error("transport error talking to {target}: {detail}")]
    Transport {
        /// Which store ("clickhouse" / "postgres").
        target: &'static str,
        /// Human-readable detail (often the server's own error body).
        detail: String,
    },
}

/// A thin ClickHouse HTTP client: just enough to run migration DDL, post rows as
/// `INSERT ... FORMAT JSONEachRow`, and run `SELECT` reads. No ORM and no query builder
/// of its own — the SQL and rows come from the database-free modules; this only carries
/// them over HTTP, because the ClickHouse HTTP interface takes the statement as the
/// POST body.
#[derive(Debug, Clone)]
pub struct ClickHouseClient {
    /// The ClickHouse HTTP endpoint (e.g. `http://localhost:8123`). Statements are
    /// POSTed here as the request body.
    pub base_url: String,
    /// The reqwest client (connection pool). Cloned cheaply; reused across calls so a
    /// migration run does not open a fresh TCP connection per statement.
    http: reqwest::Client,
    /// Optional `(user, password)` sent as HTTP basic auth on every request. `None`
    /// means connect anonymously — a password-less local ClickHouse still works. This
    /// exists because the stock server image restricts the password-less `default` user
    /// to localhost, so a containerized server reached across Docker's network needs a
    /// real user/password (the ClickHouse HTTP interface accepts HTTP basic auth).
    credentials: Option<(String, String)>,
}

impl ClickHouseClient {
    /// Build a client pointed at a ClickHouse HTTP endpoint, with no credentials. Use
    /// [`Self::with_credentials`] (or [`Self::from_env`]) when the server requires a
    /// user/password.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::new(),
            credentials: None,
        }
    }

    /// Attach a ClickHouse user/password, sent as HTTP basic auth on every request.
    /// Needed when the server rejects the anonymous `default` user (the stock image
    /// pins that user to localhost), which is the case for the containerized server the
    /// live tests talk to.
    #[must_use]
    pub fn with_credentials(
        mut self,
        user: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.credentials = Some((user.into(), password.into()));
        self
    }

    /// Build a client from `base_url`, reading `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD`
    /// from the environment and attaching them as basic auth if `CLICKHOUSE_USER` is
    /// set. With neither var set the client stays anonymous, so a password-less local
    /// ClickHouse still works.
    #[must_use]
    pub fn from_env(base_url: impl Into<String>) -> Self {
        let client = Self::new(base_url);
        match std::env::var("CLICKHOUSE_USER") {
            Ok(user) => {
                let password = std::env::var("CLICKHOUSE_PASSWORD").unwrap_or_default();
                client.with_credentials(user, password)
            }
            Err(_) => client,
        }
    }

    /// POST one statement (DDL or a write) to the ClickHouse HTTP interface and discard
    /// the (empty) response body. The statement is the raw request body, which is
    /// exactly how the HTTP interface accepts a query.
    ///
    /// # Errors
    /// [`LiveError::Transport`] on a connection failure or a non-2xx status; the
    /// server's response body is included so the actual ClickHouse error is visible.
    pub async fn execute(&self, sql: &str) -> Result<(), LiveError> {
        self.post(sql).await.map(drop)
    }

    /// Apply every ClickHouse migration's DDL in order, splitting each migration on `;`
    /// because the HTTP interface runs one statement per request. Idempotent because
    /// the DDL is all `CREATE ... IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`, so a
    /// second apply is a no-op rather than an error.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if any statement fails to apply.
    pub async fn apply_migrations(&self) -> Result<(), LiveError> {
        for m in crate::clickhouse::migrations::all() {
            for stmt in m.up_sql().split(';') {
                let stmt = stmt.trim();
                if !stmt.is_empty() {
                    self.execute(stmt).await?;
                }
            }
        }
        Ok(())
    }

    /// Insert pre-rendered `JSONEachRow` lines into a table. `lines` are the row strings
    /// from [`crate::clickhouse::rows::to_line`]; they are sent verbatim as the body
    /// after the `INSERT ... FORMAT JSONEachRow` header.
    ///
    /// # Errors
    /// [`LiveError::Transport`] on a connection failure or a non-2xx status.
    pub async fn insert_rows(&self, table: &str, lines: &[String]) -> Result<(), LiveError> {
        let body = format!(
            "INSERT INTO {table} FORMAT JSONEachRow\n{}",
            lines.join("\n")
        );
        self.execute(&body).await
    }

    /// Insert rows into a table as `JSONEachRow`, taking the JSON objects produced by
    /// [`crate::clickhouse::rows`] and rendering each to its line here.
    ///
    /// # Errors
    /// [`LiveError::Transport`] on a connection failure or a non-2xx status.
    pub async fn insert_json_each_row(&self, table: &str, rows: &[Value]) -> Result<(), LiveError> {
        let lines: Vec<String> = rows.iter().map(crate::clickhouse::rows::to_line).collect();
        self.insert_rows(table, &lines).await
    }

    /// Run a `SELECT` and parse the response as `JSONEachRow` — one JSON object per
    /// line. The caller's `sql` should NOT include the `FORMAT` clause; it is appended
    /// here so the response is line-delimited JSON this can parse into [`Value`]s for a
    /// test to assert on.
    ///
    /// # Errors
    /// [`LiveError::Transport`] on a connection failure, a non-2xx status, or a
    /// response line that is not valid JSON.
    pub async fn query(&self, sql: &str) -> Result<Vec<Value>, LiveError> {
        // Append the line-delimited-JSON format clause to the SELECT. The statement may
        // carry a trailing `;` (canonical SQL ends a statement that way), but ClickHouse
        // over HTTP rejects `<select>; FORMAT JSONEachRow` as two statements, so trim any
        // trailing semicolon/whitespace before appending the clause.
        let trimmed = sql.trim().trim_end_matches(';').trim_end();
        let body = self.post(&format!("{trimmed} FORMAT JSONEachRow")).await?;
        let mut rows = Vec::new();
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(line).map_err(|e| LiveError::Transport {
                target: "clickhouse",
                detail: format!("response line was not JSON: {e}: {line}"),
            })?;
            rows.push(v);
        }
        Ok(rows)
    }

    /// POST a statement and return the response body on success. Shared by every method
    /// so the status-check and error-mapping live in one place.
    async fn post(&self, statement: &str) -> Result<String, LiveError> {
        let mut req = self.http.post(&self.base_url).body(statement.to_string());
        // The server may reject the anonymous `default` user, so authenticate when
        // credentials are configured; the HTTP interface accepts HTTP basic auth.
        if let Some((user, password)) = &self.credentials {
            req = req.basic_auth(user, Some(password));
        }
        let resp = req.send().await.map_err(|e| LiveError::Transport {
            target: "clickhouse",
            detail: e.to_string(),
        })?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| LiveError::Transport {
            target: "clickhouse",
            detail: e.to_string(),
        })?;
        // A non-2xx ClickHouse response carries the SQL error in its body; surface it so
        // an operator sees the real cause rather than a bare status code.
        if !status.is_success() {
            return Err(LiveError::Transport {
                target: "clickhouse",
                detail: format!("HTTP {status}: {text}"),
            });
        }
        Ok(text)
    }
}

/// A thin Postgres client wrapper: runs the control-plane migrations, sets the
/// per-connection tenant GUC so row-level security applies, and reads/writes through
/// `tokio-postgres`.
#[derive(Debug)]
pub struct PostgresClient {
    /// The connection (one open session). The tenant GUC set with
    /// [`Self::set_tenant`] is connection-scoped, so a client instance is pinned to one
    /// tenant for the life of the connection.
    conn: tokio_postgres::Client,
}

impl PostgresClient {
    /// The SQL that pins this connection's tenant so the row-level-security policies
    /// scope every query. The tenant value is bound as `$1`, never interpolated, so
    /// even this setup statement cannot be injected.
    #[must_use]
    pub fn set_tenant_guc_sql() -> &'static str {
        "SELECT set_config('fieldloop.tenant_id', $1, false);"
    }

    /// Connect to Postgres at `dsn` (e.g. `postgres://user:pw@host/db`) and spawn the
    /// connection's background task. `tokio-postgres` splits the protocol driver from
    /// the query handle, so the returned driver future must be polled on a task or the
    /// connection stalls — that is what the spawn here does.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the connection cannot be established.
    pub async fn connect(dsn: &str) -> Result<Self, LiveError> {
        let (conn, driver) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
            .await
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })?;
        // The driver future owns the socket I/O; if it is dropped the connection dies.
        // Spawning it lets the query handle (`conn`) be used freely from this task.
        tokio::spawn(async move {
            let _ = driver.await;
        });
        Ok(Self { conn })
    }

    /// Run a statement (DDL or a write) with no parameters.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the statement fails.
    pub async fn execute(&self, sql: &str) -> Result<(), LiveError> {
        self.conn
            .batch_execute(sql)
            .await
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })
    }

    /// Apply every Postgres control-plane migration in order. `batch_execute` runs the
    /// multi-statement DDL of each migration in one round trip. Idempotent: the table
    /// DDL is `CREATE ... IF NOT EXISTS` and the RLS migration `DROP POLICY IF EXISTS`s
    /// before each `CREATE POLICY`, so a second apply is a no-op.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if any migration fails.
    pub async fn apply_migrations(&self) -> Result<(), LiveError> {
        for m in crate::postgres::migrations::all() {
            self.execute(&m.up_sql()).await?;
        }
        Ok(())
    }

    /// Create the dedicated application login role and grant it the data-plane DML it
    /// needs on the tenant-scoped tables. Idempotent: the role is created only if it is
    /// absent, and `GRANT` is repeatable.
    ///
    /// OPERATIONAL SAFETY: this role MUST be `NOSUPERUSER NOBYPASSRLS`. Postgres
    /// row-level security is *bypassed* by superuser and `BYPASSRLS` roles even when a
    /// table has `FORCE ROW LEVEL SECURITY` — for such a role the tenant policy is
    /// silently void and one tenant can read another tenant's rows. The bootstrap role
    /// (from `POSTGRES_USER`) is a superuser, so the application/tenant-scoped
    /// connection must NOT use it; it must connect as this non-privileged role, or
    /// per-tenant isolation evaporates with no error. Migrations and this setup run as
    /// the superuser; per-tenant reads/writes run as `app_rw`.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the role cannot be created or granted.
    pub async fn ensure_app_role(&self, role: &str, password: &str) -> Result<(), LiveError> {
        // Guard the role creation so a re-run is a no-op (Postgres has no
        // `CREATE ROLE IF NOT EXISTS`). A plain `IF NOT EXISTS ... CREATE ROLE` still
        // races two concurrent callers (both pass the check, both CREATE, one errors), so
        // attempt the CREATE unconditionally and swallow the `duplicate_object` error —
        // this is idempotent AND safe when parallel tests create the same role at once.
        // NOSUPERUSER + NOBYPASSRLS are load-bearing: a privileged role would bypass RLS
        // and break tenant isolation.
        let mut sql = format!(
            "DO $$ BEGIN \
             CREATE ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS; \
             EXCEPTION WHEN duplicate_object OR unique_violation THEN NULL; END $$;\n"
        );
        for t in [
            "policy_registry",
            "deployment_ledger",
            "pending_outcome",
            "ingest_idempotency",
            "episode_rollup",
            "deploy_attempts",
            "rollout_state",
            "rollout_transitions",
        ] {
            sql.push_str(&format!(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON {t} TO {role};\n"
            ));
        }
        self.execute(&sql).await
    }

    /// Pin this connection to a tenant by setting the `fieldloop.tenant_id` session GUC
    /// the row-level-security policies read. After this, the policies scope every query
    /// on this connection to that tenant — a forgotten `WHERE tenant_id = ...` cannot
    /// leak across tenants because Postgres itself filters.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the GUC cannot be set.
    pub async fn set_tenant(&self, tenant_id: &str) -> Result<(), LiveError> {
        self.conn
            .execute(Self::set_tenant_guc_sql(), &[&tenant_id])
            .await
            .map(drop)
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })
    }

    /// Idempotently record an ingest batch in `ingest_idempotency`. The
    /// `(tenant_id, batch_id)` primary key plus `ON CONFLICT DO NOTHING` makes a second
    /// insert of the same batch a no-op, so a retried upload writes once. Returns the
    /// number of rows actually inserted (1 the first time, 0 on a duplicate) so a test
    /// can assert the dedup happened. Values are bound, never interpolated.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the insert fails.
    pub async fn record_batch(&self, batch_id: &str, row_count: i32) -> Result<u64, LiveError> {
        // tenant_id is filled from the connection's tenant GUC via the RLS WITH CHECK,
        // but the column is NOT NULL, so it must be supplied; read it back from the GUC
        // so the insert always matches the connection's tenant.
        let sql = "INSERT INTO ingest_idempotency (tenant_id, batch_id, row_count) \
                   VALUES (current_setting('fieldloop.tenant_id'), $1, $2) \
                   ON CONFLICT (tenant_id, batch_id) DO NOTHING";
        self.conn
            .execute(sql, &[&batch_id, &row_count])
            .await
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })
    }

    /// Run a row-returning `SELECT` and hand each row back as a JSON object keyed by
    /// column name. The control-plane tables this reads (`deployment_ledger`,
    /// `policy_registry`) are text/uuid/timestamptz/jsonb columns, so each value is
    /// rendered as a JSON string (or null) rather than a typed number — a string is
    /// lossless for those column types and lets a JSON-only caller (the demo dashboard)
    /// read the deploy ledger without a typed row struct per query. The connection's
    /// tenant GUC still scopes the read under row-level security, so this never widens
    /// what a tenant-pinned connection can see; it only changes how the rows come back.
    ///
    /// Timestamps and uuids come back as strings because the demo only displays them and
    /// a string round-trips exactly what the operator stored; a caller needing a typed
    /// value can parse the string itself.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the query fails.
    pub async fn query(&self, sql: &str) -> Result<Vec<Value>, LiveError> {
        let rows = self
            .conn
            .query(sql, &[])
            .await
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut obj = serde_json::Map::new();
            for (i, col) in row.columns().iter().enumerate() {
                // Read every column as `Option<String>`. tokio-postgres' `FromSql` for
                // `String` covers text/uuid/timestamptz when the value is sent as text;
                // to be type-agnostic across the control-plane columns, try a string
                // first and fall back to null on any column type that does not decode as
                // a string, so an unexpected column never fails the whole read.
                let value: Value = match row.try_get::<_, Option<String>>(i) {
                    Ok(Some(s)) => Value::String(s),
                    Ok(None) => Value::Null,
                    Err(_) => Value::Null,
                };
                obj.insert(col.name().to_string(), value);
            }
            out.push(Value::Object(obj));
        }
        Ok(out)
    }

    /// Count the rows visible for the current tenant in `ingest_idempotency`. Because
    /// RLS scopes the read to the connection's tenant GUC, this counts ONLY this
    /// tenant's rows — so a tenant-A connection counting after a tenant-B insert proves
    /// B's rows are invisible to A.
    ///
    /// # Errors
    /// [`LiveError::Transport`] if the query fails.
    pub async fn count_batches(&self) -> Result<i64, LiveError> {
        let row = self
            .conn
            .query_one("SELECT count(*) FROM ingest_idempotency", &[])
            .await
            .map_err(|e| LiveError::Transport {
                target: "postgres",
                detail: e.to_string(),
            })?;
        Ok(row.get::<_, i64>(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, Feedback, FeedbackId, FeedbackTarget, FeedbackValue,
        JoinMethod, LabelKind, MonoClock, OutcomeEvent, OutcomeKind, PayloadRef, PolicyVersion,
        RobotId, RobotIdentity, Rollout, RolloutId, TenantId,
    };

    /// The Postgres tenant GUC setter binds the tenant as `$1` (never interpolated),
    /// so row-level security is scoped without an injectable setup statement. Pure
    /// string check — no database needed.
    #[test]
    fn postgres_tenant_guc_is_parameterized() {
        let sql = PostgresClient::set_tenant_guc_sql();
        assert!(sql.contains("set_config('fieldloop.tenant_id', $1"));
    }

    /// Compare a JSON number against an expected value within a Float32-scale tolerance.
    ///
    /// The stored reward/confidence columns are ClickHouse `Float32`, so a value read
    /// back over JSON is the nearest f32 to the original, which does not bit-match a
    /// literal f64. A small absolute epsilon makes the assertion test the value, not the
    /// floating-point representation.
    fn approx_eq(v: &serde_json::Value, expected: f64) -> bool {
        v.as_f64().is_some_and(|x| (x - expected).abs() < 1e-6)
    }

    /// A test rollout with a fixed id so the feedback can target it.
    fn rollout_for(tenant: &str, id: RolloutId) -> Rollout {
        let robot = RobotIdentity::new(TenantId::new(tenant), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 1,
            ts_wall_ns: 1_700_000_000_000_000_000,
        };
        let mut r = Rollout::new(
            robot,
            EpisodeId::new(),
            0,
            clock,
            PolicyVersion::new("pol@v1+abc123def456"),
            "sha256:weights".into(),
            "arm6dof".into(),
            "pick_place".into(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            1,
        );
        r.id = id;
        r
    }

    /// A reward Feedback bound to a rollout, with a chosen confidence, freshness
    /// (via id), retraction flag, and value — enough to drive the canonical join.
    #[allow(clippy::too_many_arguments)]
    fn feedback_for(
        tenant: &str,
        target: RolloutId,
        value: f32,
        confidence: f32,
        retracted: bool,
        dedup_key: &str,
        outcome_ts_ns: i64,
    ) -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new(tenant),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "reward".into(),
            value: FeedbackValue::Float { value },
            join_method: JoinMethod::Temporal,
            join_confidence: confidence,
            join_version: "j1".into(),
            calibration_version: "c1".into(),
            source_outcome_id: None,
            delay_ms: Some(100),
            retracted,
            dedup_key: dedup_key.into(),
            outcome_ts_ns,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    /// Live ClickHouse end-to-end: apply migrations (twice, to prove idempotency),
    /// insert real Rollout + OutcomeEvent + Feedback rows, run the canonical reward
    /// join, and assert the attributed row carries the right reward, confidence, and
    /// latest-wins/retraction behavior. Ignored and a no-op unless `CLICKHOUSE_URL` is
    /// set, so neither `cargo test` nor `cargo test --features live-db` needs a DB.
    #[tokio::test]
    #[ignore = "requires a live ClickHouse; set CLICKHOUSE_URL"]
    async fn clickhouse_join_is_correct() {
        let Ok(url) = std::env::var("CLICKHOUSE_URL") else {
            return;
        };
        // Read CLICKHOUSE_USER/CLICKHOUSE_PASSWORD from env: the containerized server
        // rejects the anonymous `default` user, so the live run sets a real user/password.
        let ch = ClickHouseClient::from_env(url);

        // Apply migrations twice: the second apply must succeed as a no-op, which is
        // the live proof that the `IF NOT EXISTS` DDL is genuinely idempotent.
        ch.apply_migrations().await.expect("first migrate");
        ch.apply_migrations().await.expect("second migrate (no-op)");

        // Use a unique tenant per run so repeated runs against the same server do not
        // see each other's rows (the canonical join is tenant-scoped).
        let tenant = format!("t-{}", FeedbackId::new().as_uuid().simple());
        let target = RolloutId::new();
        let other_target = RolloutId::new();

        // One rollout we will attribute, plus one that stays awaiting-outcome so the
        // LEFT JOIN's `has_outcome = 0` survival is exercised too.
        let r1 = rollout_for(&tenant, target);
        let r2 = rollout_for(&tenant, other_target);
        ch.insert_json_each_row(
            "Rollout",
            &[
                crate::clickhouse::rows::rollout_row(&r1),
                crate::clickhouse::rows::rollout_row(&r2),
            ],
        )
        .await
        .expect("insert rollouts");

        // A raw outcome row, just to exercise the OutcomeEvent insert path.
        let robot = RobotIdentity::new(TenantId::new(&tenant), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 2,
            ts_wall_ns: 1,
        };
        let outcome = OutcomeEvent::new(
            robot,
            clock,
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        );
        ch.insert_json_each_row(
            "OutcomeEvent",
            &[crate::clickhouse::rows::outcome_row(&outcome)],
        )
        .await
        .expect("insert outcome");

        // Three feedbacks on the SAME slot for `target`:
        //   * an early low-value binding,
        //   * a LATER (fresher id) correction to 0.9 — latest-wins must pick this,
        //   * a high-confidence retracted row — must be excluded by `retracted = 0`.
        // The fresher id is minted last, so its UUIDv7 sorts newest under `ORDER BY ts`.
        let ts = 1_700_000_100_000_000_000i64;
        let f_old = feedback_for(&tenant, target, 0.10, 0.99, false, "dk-old", ts);
        let f_new = feedback_for(&tenant, target, 0.90, 0.97, false, "dk-new", ts + 1);
        let f_retracted = feedback_for(&tenant, target, 0.50, 0.99, true, "dk-ret", ts + 2);
        // Insert oldest-first so f_new genuinely has the newest minted id.
        ch.insert_json_each_row(
            "Feedback",
            &[
                crate::clickhouse::rows::feedback_row(&f_old),
                crate::clickhouse::rows::feedback_row(&f_new),
                crate::clickhouse::rows::feedback_row(&f_retracted),
            ],
        )
        .await
        .expect("insert feedback");

        // The materialized view fan-out can lag the base insert briefly; force the
        // merge so the read sees a settled view.
        ch.execute("OPTIMIZE TABLE FeedbackByTargetId FINAL")
            .await
            .ok();

        // Build the canonical join (tenant-scoped, parameterized) and substitute the
        // bound params into the ClickHouse `{pN:Type}` markers via the query string
        // params HTTP form is overkill for a test; instead we render the values inline
        // ONLY for the test read, since they are test-controlled constants (never user
        // input). The production path binds them; here we just need a runnable SELECT.
        let q = crate::clickhouse::queries::reward_join(
            &crate::clickhouse::queries::RewardJoinParams {
                tenant: TenantId::new(&tenant),
                metric_name: "reward".into(),
                label_kind: "terminal_outcome".into(),
                min_confidence: 0.95,
                outcome_ts_from: 1_700_000_000.0,
                outcome_ts_to: 1_700_001_000.0,
            },
        )
        .expect("build join");
        let sql = bind_clickhouse_params(&q);

        let rows = ch.query(&sql).await.expect("run join");

        // Find the attributed row for our target.
        let attributed = rows
            .iter()
            .find(|r| r["rollout_id"] == serde_json::json!(target.to_string()))
            .expect("attributed rollout present");

        // Latest-wins picked f_new (0.90), NOT the older 0.10, and the retracted 0.50
        // was excluded. This is the core thing string-validation cannot prove: the real
        // SQL actually joins and resolves supersession correctly.
        // `value_float`/`join_confidence` are stored as ClickHouse Float32, so the JSON
        // number comes back as the shortest decimal for that f32 (0.9), which is NOT
        // bit-equal to a literal 0.9_f32 widened to f64 — compare within a Float32-scale
        // tolerance instead of demanding exact equality.
        assert!(
            approx_eq(&attributed["reward"], 0.9),
            "reward should be the fresh 0.90: {attributed}"
        );
        // The surviving binding's confidence is f_new's (0.97), above the 0.95 floor.
        assert!(
            approx_eq(&attributed["join_confidence"], 0.97),
            "{attributed}"
        );
        // has_outcome is true (1) for the attributed rollout.
        assert_eq!(
            attributed["has_outcome"],
            serde_json::json!(1),
            "{attributed}"
        );

        // The awaiting-outcome rollout survives the LEFT JOIN with has_outcome = 0,
        // proving the confidence predicate is in the ON clause (not the WHERE).
        let awaiting = rows
            .iter()
            .find(|r| r["rollout_id"] == serde_json::json!(other_target.to_string()))
            .expect("awaiting-outcome rollout still present");
        assert_eq!(awaiting["has_outcome"], serde_json::json!(0), "{awaiting}");
    }

    /// Live ClickHouse tenant isolation: insert feedback for tenant A and tenant B on
    /// rollouts with the SAME target id, run the tenant-injected join for A, and assert
    /// only A's reward comes back — B's row is never returned. Proves the bound tenant
    /// predicate actually scopes the read against a real server.
    #[tokio::test]
    #[ignore = "requires a live ClickHouse; set CLICKHOUSE_URL"]
    async fn clickhouse_tenant_isolation() {
        let Ok(url) = std::env::var("CLICKHOUSE_URL") else {
            return;
        };
        let ch = ClickHouseClient::from_env(url);
        ch.apply_migrations().await.expect("migrate");

        let suffix = FeedbackId::new().as_uuid().simple().to_string();
        let tenant_a = format!("a-{suffix}");
        let tenant_b = format!("b-{suffix}");
        // SAME target id in both tenants: the only thing that must keep them apart is
        // the tenant predicate, so a shared id is the strongest isolation test.
        let target = RolloutId::new();

        let ra = rollout_for(&tenant_a, target);
        let rb = rollout_for(&tenant_b, target);
        ch.insert_json_each_row(
            "Rollout",
            &[
                crate::clickhouse::rows::rollout_row(&ra),
                crate::clickhouse::rows::rollout_row(&rb),
            ],
        )
        .await
        .expect("insert rollouts");

        let ts = 1_700_000_200_000_000_000i64;
        // A gets reward 0.11, B gets reward 0.88 — distinct so a leak is unmistakable.
        let fa = feedback_for(&tenant_a, target, 0.11, 0.99, false, "dk-a", ts);
        let fb = feedback_for(&tenant_b, target, 0.88, 0.99, false, "dk-b", ts);
        ch.insert_json_each_row(
            "Feedback",
            &[
                crate::clickhouse::rows::feedback_row(&fa),
                crate::clickhouse::rows::feedback_row(&fb),
            ],
        )
        .await
        .expect("insert feedback");
        ch.execute("OPTIMIZE TABLE FeedbackByTargetId FINAL")
            .await
            .ok();

        // Run the join scoped to tenant A.
        let q = crate::clickhouse::queries::reward_join(
            &crate::clickhouse::queries::RewardJoinParams {
                tenant: TenantId::new(&tenant_a),
                metric_name: "reward".into(),
                label_kind: "terminal_outcome".into(),
                min_confidence: 0.5,
                outcome_ts_from: 1_700_000_000.0,
                outcome_ts_to: 1_700_001_000.0,
            },
        )
        .expect("build join");
        let rows = ch
            .query(&bind_clickhouse_params(&q))
            .await
            .expect("run join");

        // A's reward (0.11) is present; B's reward (0.88) is NEVER returned.
        // Float32 storage: compare within a tolerance, not bit-exact (see the join test).
        assert!(
            rows.iter().any(|r| approx_eq(&r["reward"], 0.11)),
            "tenant A's own row must be present: {rows:?}"
        );
        assert!(
            rows.iter().all(|r| !approx_eq(&r["reward"], 0.88)),
            "tenant B's row must NEVER leak into tenant A's read: {rows:?}"
        );
    }

    /// Live Postgres: apply migrations (twice — idempotency), set the tenant GUC, and
    /// prove (a) the same batch id inserted twice has one logical effect (idempotency),
    /// and (b) a connection pinned to tenant A never sees tenant B's rows (RLS
    /// isolation). Ignored and a no-op unless `DATABASE_URL` is set.
    ///
    /// The migrations and role setup run on the bootstrap (superuser) DSN, but the
    /// per-tenant reads/writes connect as a dedicated NOSUPERUSER/NOBYPASSRLS role
    /// (`app_rw`), because superuser/`BYPASSRLS` roles bypass row-level security even
    /// under `FORCE` — so a superuser connection would silently see across tenants and
    /// the isolation assertion would (wrongly) pass or, as observed, fail by seeing two
    /// rows. This mirrors the production rule: the application connection must never be
    /// privileged.
    #[tokio::test]
    #[ignore = "requires a live Postgres; set DATABASE_URL"]
    async fn postgres_idempotency_and_isolation() {
        let Ok(dsn) = std::env::var("DATABASE_URL") else {
            return;
        };
        // Admin connection (superuser, from POSTGRES_USER): migrations + role setup only.
        let admin = PostgresClient::connect(&dsn).await.expect("connect admin");
        admin.apply_migrations().await.expect("first migrate");
        admin
            .apply_migrations()
            .await
            .expect("second migrate (no-op)");
        admin
            .ensure_app_role("app_rw", "app_rw")
            .await
            .expect("create app_rw role");

        // Build the unprivileged DSN by swapping the userinfo in the admin DSN, so the
        // tenant-scoped connections are subject to RLS (a superuser would bypass it).
        let app_dsn = app_rw_dsn(&dsn);

        let suffix = FeedbackId::new().as_uuid().simple().to_string();
        let tenant_a = format!("a-{suffix}");
        let tenant_b = format!("b-{suffix}");
        let batch = format!("batch-{suffix}");

        // --- Idempotency: same batch id twice => inserted once. ---
        let pg = PostgresClient::connect(&app_dsn)
            .await
            .expect("connect app_rw A");
        pg.set_tenant(&tenant_a).await.expect("set tenant A");
        let first = pg.record_batch(&batch, 10).await.expect("first insert");
        let second = pg.record_batch(&batch, 10).await.expect("retry insert");
        assert_eq!(first, 1, "first insert writes one row");
        assert_eq!(
            second, 0,
            "retried batch is a no-op (ON CONFLICT DO NOTHING)"
        );
        assert_eq!(
            pg.count_batches().await.expect("count A"),
            1,
            "exactly one logical effect for tenant A's batch"
        );

        // --- Isolation: tenant B inserts; tenant A's read still sees only its own. ---
        let pg_b = PostgresClient::connect(&app_dsn)
            .await
            .expect("connect app_rw B");
        pg_b.set_tenant(&tenant_b).await.expect("set tenant B");
        pg_b.record_batch(&batch, 99).await.expect("B insert");
        // Same batch id, different tenant: A still sees exactly its own one row, B sees
        // exactly its own one row — RLS scopes each connection to its tenant GUC.
        assert_eq!(
            pg.count_batches().await.expect("A re-count"),
            1,
            "tenant A must NOT see tenant B's row"
        );
        assert_eq!(
            pg_b.count_batches().await.expect("B count"),
            1,
            "tenant B sees only its own row"
        );
    }

    /// Derive the unprivileged `app_rw` DSN from the admin DSN by replacing the
    /// userinfo (`user:password@`) with `app_rw:app_rw@`, keeping host/port/db. Used so
    /// the tenant-scoped connections are the non-superuser role RLS actually applies to;
    /// the admin DSN's superuser would bypass RLS and void the isolation guarantee.
    fn app_rw_dsn(admin_dsn: &str) -> String {
        // Split scheme from the rest, then strip any existing userinfo before the host.
        let (scheme, rest) = admin_dsn.split_once("://").expect("DSN must contain ://");
        let host_and_db = rest.rsplit_once('@').map_or(rest, |(_, after)| after);
        format!("{scheme}://app_rw:app_rw@{host_and_db}")
    }

    /// Swapping the userinfo in a `postgres://user:pw@host/db` DSN yields the same
    /// host/db with the `app_rw` role — the connection RLS is enforced on.
    #[test]
    fn app_rw_dsn_swaps_userinfo() {
        assert_eq!(
            app_rw_dsn("postgres://fieldloop:fieldloop@localhost:5432/fieldloop"),
            "postgres://app_rw:app_rw@localhost:5432/fieldloop"
        );
        // No userinfo present: still produces a valid app_rw DSN.
        assert_eq!(
            app_rw_dsn("postgres://localhost:5432/fieldloop"),
            "postgres://app_rw:app_rw@localhost:5432/fieldloop"
        );
    }

    /// Render a built [`crate::tenant::TenantQuery`]'s bound values into its ClickHouse
    /// `{pN:Type}` markers, for the TEST read path only. Production binds these as HTTP
    /// query params; a test's values are trusted constants, so inlining them keeps the
    /// test a single self-contained SELECT without an HTTP param-passing dance. The
    /// builder already proved (in its own unit tests) that values are never inlined on
    /// the production path — this helper is test-local.
    fn bind_clickhouse_params(q: &crate::tenant::TenantQuery) -> String {
        use crate::tenant::ParamValue;
        let mut sql = q.sql.clone();
        for (i, p) in q.params.iter().enumerate() {
            let marker_str = match p {
                ParamValue::Str(s) => format!("'{}'", s.replace('\'', "''")),
                ParamValue::I64(v) => v.to_string(),
                ParamValue::F64(v) => v.to_string(),
                ParamValue::Bool(b) => i64::from(*b).to_string(),
            };
            let marker = format!("{{p{i}:{}}}", p.clickhouse_type());
            sql = sql.replace(&marker, &marker_str);
        }
        sql
    }
}
