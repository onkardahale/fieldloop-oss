//! Policy version + provenance.
//!
//! `policy_version` threads through every row so provenance is bidirectional: from
//! any row you can find the policy that produced it, and from any policy you can
//! find the rows it touched. A robot's self-reported `policy_version` is only a
//! *claim*; at ingest it is reconciled against the server-side deployment ledger,
//! because what a robot says it is running is not trusted as truth.
//!
//! The version string format is `name@vsemver+sha256[:12]`, a content hash over
//! `(dataset_commit_ids, train_config, base_weights_hash)`. We therefore model
//! [`PolicyVersion`] as a **content-hash string newtype**, NOT a UUID — its
//! identity *is* the hash, so two builds with identical inputs collide by design.

use serde::{Deserialize, Serialize};

/// A policy version identifier — the content-hash string that threads through
/// every Rollout and Feedback row, giving bidirectional provenance.
///
/// Format: `name@vsemver+sha256[:12]`, a hash over `(dataset_commit_ids,
/// train_config, base_weights_hash)`. Kept as a validated-elsewhere string here
/// (this crate is types-only; charset/format validation happens at config load).
/// It is a newtype rather than a bare `String` so it can't be confused with a
/// `task_id`, `arm_label`, or free text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PolicyVersion(pub String);

impl PolicyVersion {
    /// Wrap a policy-version string. Format validation is a load-time concern;
    /// this type only asserts "this is a policy version".
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string (e.g. to write the storage column).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PolicyVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The artifact format a policy bundle is admitted as. A closed enum, not a
/// free-form string, so the trusted load path is exhaustive: only known-safe
/// formats can be matched. Weights load safetensors-only and in a sandbox, because
/// an arbitrary model file is hostile until proven otherwise; anything not in this
/// enum is shunted to the untrusted tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFormat {
    /// The only format admitted to the trusted load path.
    Safetensors,
}

/// A registered policy version with its provenance (the policy registry row).
///
/// This is the *registry record* — the immutable description of a trained policy,
/// and the anchor for bidirectional provenance. The runtime deployment claim/truth
/// (which robot ran it when) lives in the separate deployment ledger, modeled
/// elsewhere when that subsystem is built.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyVersionRecord {
    /// The content-hash policy version id — threads through every row.
    pub policy_version: PolicyVersion,
    /// SHA-256 of the model weights. Distinct from `policy_version`'s embedded
    /// short hash; this is the full digest the deployment ledger reconciles a
    /// robot's self-report against, and that the bundle signature covers.
    pub model_sha256: String,
    /// Artifact format — closed enum, `safetensors` only on the trusted path.
    pub artifact_format: ArtifactFormat,
    /// Bundle signature (over a Merkle root covering weights + config + adapter
    /// selector). Must verify before deploy, so a tampered bundle can't reach the
    /// fleet. Optional here because an unsigned in-flight candidate exists before
    /// admission.
    pub signature: Option<String>,
    /// Embodiment this policy targets (the robot type / action-space class).
    /// Cross-referenced against the embodiment registry at config load. Carried as
    /// a validated string in this types crate.
    pub embodiment: String,
    /// Provenance inputs the content hash is computed over — the anchor for the
    /// "which episodes/commits did this policy come from" lineage query.
    pub provenance: PolicyProvenance,
}

/// The provenance inputs a [`PolicyVersion`] hash is derived from.
///
/// These give bidirectional provenance: from these fields you can walk policy →
/// dataset commits → episodes, and the lineage spine walks the reverse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyProvenance {
    /// lakeFS dataset commit ids this policy was trained on.
    pub dataset_commit_ids: Vec<String>,
    /// SHA-256 of the base weights this policy was fine-tuned from (empty/None
    /// for a from-scratch policy).
    pub base_weights_sha256: Option<String>,
    /// Content hash of the training config (hyperparameters, recipe).
    pub train_config_hash: String,
}
