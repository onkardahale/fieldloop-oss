//! Tenant identity.
//!
//! Tenant identity is `(tenant_id, robot_id)` everywhere, never `robot_id` alone:
//! a robot id is unique only within its tenant, so a bare robot id is ambiguous and
//! a cross-tenant mix-up would be a data-isolation breach. Storage-layer isolation
//! (row policies / RLS) does the real enforcement, but the type system should never
//! even let a robot id travel without its tenant.
//!
//! We bundle the pair into one [`RobotIdentity`] type so a function signature
//! cannot take a `robot_id` without also carrying its `tenant_id`. The tenant
//! always travels with the robot, which is what lets the attribution layer treat
//! any join across two different tenants as a hard error rather than a low-quality
//! match.

use serde::{Deserialize, Serialize};

/// Tenant id. Newtype over `String` (stored `LowCardinality(String)`), so it is
/// never confused with a `robot_id` or any other low-cardinality label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TenantId(pub String);

impl TenantId {
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Robot id. A `robot_id` is **only** meaningful paired with its [`TenantId`] —
/// it is unique only within a tenant, so two tenants may reuse the same robot id.
/// Always carry it inside a [`RobotIdentity`], never alone.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RobotId(pub String);

impl RobotId {
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RobotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The canonical tenant identity carried on every Rollout and OutcomeEvent: the
/// `(tenant_id, robot_id)` pair. Inlined (`#[serde(flatten)]` at the use site) so
/// the wire/storage form keeps the two columns flat, matching the storage schema —
/// while Rust code can only ever construct and pass the pair together, so a robot
/// id can never be handled without its tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RobotIdentity {
    pub tenant_id: TenantId,
    pub robot_id: RobotId,
}

impl RobotIdentity {
    #[must_use]
    pub fn new(tenant_id: TenantId, robot_id: RobotId) -> Self {
        Self {
            tenant_id,
            robot_id,
        }
    }

    /// True iff two identities share a tenant. The attribution layer uses this to
    /// reject any join between two different tenants as a hard error — a
    /// cross-tenant binding is a data-isolation breach, never just a
    /// low-confidence row.
    #[must_use]
    pub fn same_tenant(&self, other: &RobotIdentity) -> bool {
        self.tenant_id == other.tenant_id
    }
}
