//! The fail-closed, tenant-injecting query builder — the safety-critical piece.
//!
//! Every query that touches tenant data must be scoped to exactly one tenant, and
//! that isolation must NOT depend on a developer remembering to hand-write a
//! `WHERE tenant_id = ...` predicate (forgetting one is a silent cross-tenant data
//! leak). This builder makes the safe path the only path:
//!
//!   * It **REQUIRES a tenant id**. Finalizing a query without one returns an
//!     [`BuildError::MissingTenant`] error rather than emitting an unscoped query —
//!     so a tenant-less query is *not constructible*, it fails closed.
//!   * It injects `tenant_id` as a **bound parameter in a leading position**, so the
//!     scope predicate is always present and always the first parameter.
//!   * It **parameterizes every value**. Callers append `{param}` placeholders and
//!     push the values separately; the builder never string-interpolates a value into
//!     the SQL text. A value containing SQL metacharacters therefore lands in the
//!     parameter list, not in the query string, so injection is structurally
//!     impossible (the value is data, never code).
//!
//! The output is `(sql, params)`: a parameterized statement plus an ordered list of
//! bound values, ready to hand to whichever driver the live layer uses. Both the
//! ClickHouse `{name:Type}` form and an ordinal `$N`/`?` form can be produced from
//! the same builder, so the same tenant guarantee covers both stores.

use fieldloop_types::TenantId;

/// A value bound into a parameterized query. A small closed set of the column types
/// the schema actually uses, so a value is always carried as typed *data* in the
/// parameter list and never spliced into the SQL text as code.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// A string value (e.g. `tenant_id`, `metric_name`, a `policy_version`). Carried
    /// as data; SQL metacharacters in it are inert because it is never interpolated.
    Str(String),
    /// A 64-bit signed integer (e.g. an `outcome_ts` nanosecond bound, a `delay_ms`).
    I64(i64),
    /// A 64-bit float (e.g. a confidence threshold).
    F64(f64),
    /// A boolean value.
    Bool(bool),
}

impl ParamValue {
    /// The ClickHouse parameter type name (`{name:Type}`) for this value, so a
    /// generated ClickHouse query can declare each bind param with the right type.
    #[must_use]
    pub fn clickhouse_type(&self) -> &'static str {
        match self {
            ParamValue::Str(_) => "String",
            ParamValue::I64(_) => "Int64",
            ParamValue::F64(_) => "Float64",
            ParamValue::Bool(_) => "Bool",
        }
    }
}

impl From<&str> for ParamValue {
    fn from(s: &str) -> Self {
        ParamValue::Str(s.to_string())
    }
}
impl From<String> for ParamValue {
    fn from(s: String) -> Self {
        ParamValue::Str(s)
    }
}
impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::I64(v)
    }
}
impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::F64(v)
    }
}
impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        ParamValue::Bool(v)
    }
}

/// Error from finalizing a tenant-scoped query. The whole point of the builder is
/// that the unsafe outcomes are *errors*, not silently-emitted queries.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuildError {
    /// A query was finalized without a tenant id. This is the fail-closed guard: a
    /// query with no tenant scope must never be emittable, because an unscoped query
    /// can read across tenants. The caller must set a tenant before building.
    #[error("refusing to build a query with no tenant_id: every query must be tenant-scoped")]
    MissingTenant,
    /// A `{param}` placeholder was referenced in the SQL fragment for which no value
    /// was pushed (or vice-versa). Caught at build time so a malformed parameter list
    /// cannot be sent to the driver. The tenant placeholder (`{tenant}`) is NOT
    /// counted here — it may legitimately appear more than once (e.g. a subquery and
    /// an outer filter both scoped to the same tenant) yet always binds the single
    /// leading tenant value.
    #[error(
        "parameter count mismatch: {placeholders} `{{param}}` placeholders but {values} pushed values"
    )]
    ParamCountMismatch {
        /// Number of `{param}` placeholders found in the assembled fragment.
        placeholders: usize,
        /// Number of ordinary values pushed (NOT counting the leading tenant value).
        values: usize,
    },
    /// The SQL body referenced no `{tenant}` placeholder at all. The tenant is bound
    /// and injected, but the body must still SAY where the scope predicate goes — an
    /// unscoped body is rejected so the tenant filter can never be silently absent.
    #[error(
        "the query body has no {{tenant}} placeholder: every query must apply the tenant scope"
    )]
    NoTenantPlaceholder,
}

/// A finalized, tenant-scoped query: the SQL text plus its ordered bound values.
///
/// The two together are what a driver needs: the SQL contains only placeholders, the
/// values ride alongside as data. The tenant value is always present and always the
/// first bound value, so the scope can never be dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct TenantQuery {
    /// The parameterized SQL. Contains placeholders only — no interpolated values.
    pub sql: String,
    /// The ordered bound values. `params[0]` is always the `tenant_id`.
    pub params: Vec<ParamValue>,
}

/// Builds a tenant-scoped, fully-parameterized query.
///
/// Usage: set the tenant (required), provide a SQL body that opens with the leading
/// tenant predicate placeholder, push the remaining values in placeholder order, then
/// [`TenantQueryBuilder::build`]. The leading tenant parameter is injected for you, so
/// the body never has to (and never should) hand-write the tenant value.
///
/// The placeholder convention is `{param}` for an ordinary value and `{tenant}` for
/// the injected leading tenant scope. At build time they are rewritten left-to-right
/// into the chosen dialect's positional/named markers, and the value list is checked
/// against the placeholder count so a mismatch fails closed.
#[derive(Debug, Clone)]
pub struct TenantQueryBuilder {
    tenant: Option<TenantId>,
    body: String,
    values: Vec<ParamValue>,
    dialect: Dialect,
}

/// Which placeholder syntax to emit. The tenant guarantee is identical for both; only
/// the marker text differs, because ClickHouse and Postgres spell bind params
/// differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// ClickHouse `{name:Type}` named parameters. The tenant becomes `{p0:String}`.
    ClickHouse,
    /// Postgres `$N` ordinal parameters. The tenant becomes `$1`.
    Postgres,
}

impl TenantQueryBuilder {
    /// Start a builder for the given dialect with no tenant and no body yet. The
    /// tenant is intentionally absent at construction so the only way to get a tenant
    /// in is the explicit [`TenantQueryBuilder::tenant`] call — a build without it
    /// fails closed.
    #[must_use]
    pub fn new(dialect: Dialect) -> Self {
        Self {
            tenant: None,
            body: String::new(),
            values: Vec::new(),
            dialect,
        }
    }

    /// Set the required tenant scope. Until this is called, [`TenantQueryBuilder::build`]
    /// returns [`BuildError::MissingTenant`].
    #[must_use]
    pub fn tenant(mut self, tenant: TenantId) -> Self {
        self.tenant = Some(tenant);
        self
    }

    /// Provide the SQL body. It must reference the leading tenant scope as the
    /// `{tenant}` placeholder (so the scope is visibly part of the query the caller
    /// wrote) and any further values as `{param}` placeholders, in the same order the
    /// values are pushed. Values are NEVER written inline — that is what keeps a value
    /// with SQL metacharacters out of the SQL text.
    #[must_use]
    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Push the next ordinary bound value (in `{param}` order). The value is carried
    /// as typed data; it is never spliced into the SQL string.
    #[must_use]
    pub fn push(mut self, value: impl Into<ParamValue>) -> Self {
        self.values.push(value.into());
        self
    }

    /// Finalize into a `(sql, params)` [`TenantQuery`], injecting the tenant as the
    /// leading bound parameter.
    ///
    /// Fails closed: with no tenant set this returns [`BuildError::MissingTenant`]
    /// rather than an unscoped query. It also rejects a value/placeholder count
    /// mismatch so a malformed parameter list cannot reach the driver.
    pub fn build(self) -> Result<TenantQuery, BuildError> {
        // Fail-closed guard: no tenant => no query. This is the core safety property.
        // Borrow (not move) the tenant: the placeholder-rewrite loop below also borrows
        // `self`, so taking it by reference keeps both borrows compatible.
        let tenant = self.tenant.as_ref().ok_or(BuildError::MissingTenant)?;

        // The full ordered value list: tenant ALWAYS first, then the caller's values.
        let mut params: Vec<ParamValue> = Vec::with_capacity(self.values.len() + 1);
        params.push(ParamValue::Str(tenant.as_str().to_string()));
        params.extend(self.values.iter().cloned());

        // Rewrite placeholders left-to-right into dialect markers. `{tenant}` is the
        // leading scope (parameter index 0); each `{param}` consumes the next index.
        let mut sql = String::with_capacity(self.body.len() + 16);
        let mut next_index = 1usize; // 1-based index for ordinary params; 0 = tenant.
        // Counted separately: `{tenant}` may repeat (subquery + outer filter) and all
        // repeats bind the same leading value, so only `{param}` count is matched
        // against the pushed values. `tenant_seen` enforces the scope is present.
        let mut param_count = 0usize;
        let mut tenant_seen = false;
        let mut chars = self.body.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' {
                // Read the placeholder name up to the closing '}'.
                let mut name = String::new();
                for nc in chars.by_ref() {
                    if nc == '}' {
                        break;
                    }
                    name.push(nc);
                }
                match name.as_str() {
                    "tenant" => {
                        sql.push_str(&self.marker(0));
                        tenant_seen = true;
                    }
                    "param" => {
                        sql.push_str(&self.marker(next_index));
                        next_index += 1;
                        param_count += 1;
                    }
                    other => {
                        // Unknown placeholder: re-emit it verbatim. This lets the
                        // generated SQL contain literal `{name:Type}` typed markers
                        // that some queries build directly, without the builder
                        // mis-counting them as values.
                        sql.push('{');
                        sql.push_str(other);
                        sql.push('}');
                    }
                }
            } else {
                sql.push(c);
            }
        }

        // The body must apply the tenant scope somewhere — an unscoped body is
        // rejected so the tenant filter can never be silently absent.
        if !tenant_seen {
            return Err(BuildError::NoTenantPlaceholder);
        }

        // The number of `{param}` placeholders must equal the number of pushed values,
        // else the value list and the SQL disagree — fail closed rather than send it.
        if param_count != self.values.len() {
            return Err(BuildError::ParamCountMismatch {
                placeholders: param_count,
                values: self.values.len(),
            });
        }

        Ok(TenantQuery { sql, params })
    }

    /// The dialect marker text for the given 0-based parameter index.
    fn marker(&self, index: usize) -> String {
        match self.dialect {
            // ClickHouse named params; type is taken from the value at this index so
            // the declared `{pN:Type}` matches the bound value's type.
            Dialect::ClickHouse => {
                let ty = if index == 0 {
                    // index 0 is always the tenant string.
                    "String"
                } else {
                    self.values[index - 1].clickhouse_type()
                };
                format!("{{p{index}:{ty}}}")
            }
            // Postgres ordinal params are 1-based.
            Dialect::Postgres => format!("${}", index + 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fail-closed: building with no tenant set is an error, never an unscoped query.
    #[test]
    fn no_tenant_is_an_error() {
        let err = TenantQueryBuilder::new(Dialect::ClickHouse)
            .body("SELECT 1 WHERE tenant_id = {tenant}")
            .build()
            .unwrap_err();
        assert_eq!(err, BuildError::MissingTenant);
    }

    /// With a tenant set, the tenant is always present as the leading (index-0) bound
    /// parameter, and the leading marker references it.
    #[test]
    fn tenant_is_always_the_leading_bound_param() {
        let q = TenantQueryBuilder::new(Dialect::ClickHouse)
            .tenant(TenantId::new("acme"))
            .body("SELECT 1 WHERE tenant_id = {tenant} AND metric_name = {param}")
            .push("reward")
            .build()
            .expect("tenant set, should build");

        // params[0] is the tenant, always.
        assert_eq!(q.params[0], ParamValue::Str("acme".to_string()));
        // The leading marker is the tenant param p0.
        assert!(
            q.sql.contains("tenant_id = {p0:String}"),
            "tenant must be a leading bound param: {}",
            q.sql
        );
        // The ordinary value got its own later marker.
        assert!(q.sql.contains("metric_name = {p1:String}"), "{}", q.sql);
        assert_eq!(q.params[1], ParamValue::Str("reward".to_string()));
    }

    /// The injection guard: a value full of SQL metacharacters ends up as a *bound
    /// value*, never interpolated into the SQL text.
    #[test]
    fn malicious_value_is_a_parameter_not_interpolated() {
        let nasty = "'; DROP TABLE Rollout; --";
        let q = TenantQueryBuilder::new(Dialect::Postgres)
            .tenant(TenantId::new("acme"))
            .body("SELECT * FROM Feedback WHERE tenant_id = {tenant} AND metric_name = {param}")
            .push(nasty)
            .build()
            .expect("should build");

        // The dangerous text is carried as DATA in the params list...
        assert!(q.params.contains(&ParamValue::Str(nasty.to_string())));
        // ...and does NOT appear anywhere in the SQL string.
        assert!(
            !q.sql.contains("DROP TABLE"),
            "value must never be interpolated into SQL: {}",
            q.sql
        );
        // Postgres ordinal markers: tenant is $1, the value is $2.
        assert!(q.sql.contains("tenant_id = $1"), "{}", q.sql);
        assert!(q.sql.contains("metric_name = $2"), "{}", q.sql);
    }

    /// A placeholder with no matching value (or vice-versa) is rejected at build time,
    /// so a mismatched parameter list never reaches the driver.
    #[test]
    fn param_count_mismatch_is_rejected() {
        // Two `{param}` placeholders but only one pushed value (+ tenant) => mismatch.
        let err = TenantQueryBuilder::new(Dialect::Postgres)
            .tenant(TenantId::new("acme"))
            .body("SELECT 1 WHERE tenant_id = {tenant} AND a = {param} AND b = {param}")
            .push("only-one")
            .build()
            .unwrap_err();
        match err {
            BuildError::ParamCountMismatch {
                placeholders,
                values,
            } => {
                assert_eq!(placeholders, 2); // two `{param}` placeholders
                assert_eq!(values, 1); // one pushed value (tenant not counted)
            }
            other => panic!("expected count mismatch, got {other:?}"),
        }
    }

    /// A body that repeats `{tenant}` (e.g. a subquery and an outer filter both scoped
    /// to the same tenant) is valid: every repeat binds the single leading tenant
    /// value, and only `{param}` placeholders are matched against pushed values.
    #[test]
    fn repeated_tenant_placeholder_binds_one_value() {
        let q = TenantQueryBuilder::new(Dialect::Postgres)
            .tenant(TenantId::new("acme"))
            .body(
                "SELECT * FROM a WHERE tenant_id = {tenant} AND id IN \
                 (SELECT id FROM b WHERE tenant_id = {tenant} AND m = {param})",
            )
            .push("reward")
            .build()
            .expect("repeated tenant is fine, one bound value");
        // Exactly one tenant value + one pushed value.
        assert_eq!(q.params.len(), 2);
        assert_eq!(q.params[0], ParamValue::Str("acme".into()));
        // Both tenant predicates resolved to the same leading $1 marker.
        assert_eq!(q.sql.matches("tenant_id = $1").count(), 2, "{}", q.sql);
        assert!(q.sql.contains("m = $2"), "{}", q.sql);
    }

    /// A body with no `{tenant}` placeholder at all is rejected: the tenant is bound,
    /// but the body must still say where the scope predicate applies.
    #[test]
    fn missing_tenant_placeholder_is_rejected() {
        let err = TenantQueryBuilder::new(Dialect::Postgres)
            .tenant(TenantId::new("acme"))
            .body("SELECT 1 FROM t WHERE x = {param}")
            .push("v")
            .build()
            .unwrap_err();
        assert_eq!(err, BuildError::NoTenantPlaceholder);
    }

    /// Unknown placeholders (literal `{name:Type}` typed markers in a hand-built
    /// query) are passed through verbatim and not mis-counted as values, so a query
    /// can embed pre-typed ClickHouse markers alongside the injected ones.
    #[test]
    fn unknown_placeholders_pass_through_untouched() {
        let q = TenantQueryBuilder::new(Dialect::ClickHouse)
            .tenant(TenantId::new("acme"))
            .body("SELECT toUInt128({some_uuid:String}) WHERE tenant_id = {tenant}")
            .build()
            .expect("should build");
        // The literal typed marker survives; only `{tenant}` was rewritten.
        assert!(q.sql.contains("{some_uuid:String}"), "{}", q.sql);
        assert!(q.sql.contains("tenant_id = {p0:String}"), "{}", q.sql);
    }

    /// Numeric and boolean values get the right ClickHouse parameter type, so the
    /// generated typed markers match the bound values.
    #[test]
    fn typed_values_declare_the_right_clickhouse_type() {
        let q = TenantQueryBuilder::new(Dialect::ClickHouse)
            .tenant(TenantId::new("acme"))
            .body(
                "WHERE tenant_id = {tenant} AND ts >= {param} AND conf >= {param} AND ok = {param}",
            )
            .push(1_700_000_000_000_000_000i64)
            .push(0.95f64)
            .push(true)
            .build()
            .expect("should build");
        assert!(q.sql.contains("ts >= {p1:Int64}"), "{}", q.sql);
        assert!(q.sql.contains("conf >= {p2:Float64}"), "{}", q.sql);
        assert!(q.sql.contains("ok = {p3:Bool}"), "{}", q.sql);
    }
}
