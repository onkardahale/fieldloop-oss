# ADR 0001 — Tenant isolation: a fail-closed query builder plus Postgres RLS

Status: accepted

## Context

The platform is multi-tenant: one deployment serves many fleets, and one fleet must never read
another's rollouts, outcomes, or deploy decisions. Two failure modes are unacceptable: a query that
forgets its tenant filter (cross-tenant read), and a value string-interpolated into SQL (injection).

## Decision

Two layers, both fail-closed:

1. **A query builder that cannot emit an unscoped query.** Every query is bound to exactly one
   tenant via a *parameter*, and building one with no tenant is an error, not an empty filter. Values
   are parameterized — never interpolated — so a value can never become SQL.
2. **Postgres row-level security** on the control-plane tables, scoped by a per-connection tenant
   GUC, so even a query that slipped the filter returns nothing.

The application/data-plane connection MUST be a non-superuser, non-`BYPASSRLS` role: RLS is silently
void for those roles even with `FORCE ROW LEVEL SECURITY`. Migrations may run as the owner; per-tenant
reads/writes must not.

## Consequences

- Isolation is enforced in two independent places; a bug in one is caught by the other.
- The cost is discipline: the data-plane role's privileges are load-bearing and must be reviewed.
- Analytics SQL generated outside the builder (e.g. demo/admin tools) must escape the tenant itself —
  a gap that has bitten once and is now covered by tests.
