//! `Rollout` — the atom (immutable, append-only event row).
//!
//! One deployed-policy inference step. Minted on the robot inside the 50Hz control
//! loop (so construction must stay cheap and non-blocking), it is the left side of
//! the JOIN: an [`crate::OutcomeEvent`] is later attributed back to a Rollout to
//! produce [`crate::Feedback`].
//!
//! Field grouping reflects the four design rules: identity & ordering; the two
//! clocks (ids carry an advisory timestamp, the monotonic clock is authoritative);
//! policy attribution (a robot's self-report is a claim until the gateway
//! reconciles it); and payload-as-pointers (sensor bytes stay in the customer's
//! store).

use serde::{Deserialize, Serialize};

use crate::clock::{MonoClock, ServerAnchor};
use crate::ids::{EpisodeId, EvalRunId, RolloutId};
use crate::payload::{BoundedBlob, PayloadRef};
use crate::policy::PolicyVersion;
use crate::pose::Se3Pose;
use crate::tenant::RobotIdentity;

/// Server-derived trust after deployment-ledger reconciliation.
///
/// The robot self-reports `policy_version`/`model_hash`; that is only a *claim*.
/// The ingest gateway reconciles it against the server-side deployment ledger (the
/// record of which policy was actually deployed where) and sets this. Safety-
/// regression eval consumes only `Trusted` rows, so an unverifiable policy claim
/// can never feed a safety decision. Closed enum — there is no "maybe" trust state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    /// Reconciled against the ledger and matched. Eligible for safety eval.
    Trusted,
    /// Unreconciled or mismatched self-report. Captured anyway — deploying a new
    /// policy must never brick capture on the robots running it, so an
    /// unknown-but-well-formed policy version is still recorded — but excluded from
    /// the safety path.
    Untrusted,
}

/// How far an inline payload field has been schema-validated. Closed enum: an
/// unchecked field is *labeled* unchecked rather than silently trusted, because an
/// open-SDK field is hostile until validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaConformance {
    /// Not yet validated against any schema.
    Unchecked,
    /// Validated by the edge SDK/sidecar before upload.
    EdgeValidated,
    /// Validated by a plane-side consumer (e.g. ingest-gateway re-validation).
    ConsumerValidated,
}

/// The eval context for a rollout — the eval run id paired with a blind arm label.
///
/// Modeled as an `Option<EvalContext>` on the Rollout: `None` is the normal
/// production case (no eval window), `Some` carries the blinded A/B context. Pairing
/// the two in one struct makes "has an arm_label but no eval_run_id" — and
/// vice-versa — unrepresentable, so a rollout is either fully in an eval or fully
/// out of one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalContext {
    /// The eval run this rollout belongs to.
    pub eval_run_id: EvalRunId,
    /// Blind A/B label. Deliberately **not** the `policy_version` — the gate must
    /// not be able to tell which arm is the candidate, so the arm is labeled
    /// opaquely here and the real policy version is gated behind access control.
    pub arm_label: String,
}

/// Whether a rollout is a recording of a real fielded inference step or a piece of
/// synthetic data (sim eval, or a Cosmos-style augmentation of a real failure).
///
/// Synthetic data is useful — you can amplify one real failure into many variations,
/// or pre-screen a candidate policy in sim before touching hardware — but it must
/// never be mistaken for real field evidence. A sim is an approximation of the world,
/// and the residual sim-to-real gap means a behavior that looks safe in synthetic data
/// can still fail on a physical robot. So provenance is carried explicitly on every
/// rollout: curation can weight or separate synthetic data, and the safety path can
/// exclude it outright.
///
/// `Default` is `Real`: an old robot or an old stored row that predates this field
/// deserializes as real (see the `#[serde(default)]` on [`Rollout::provenance`]). That
/// is what makes adding this field a safe, backward-compatible append rather than a
/// breaking schema change — but it also means a synthetic source MUST explicitly mark
/// itself synthetic, since silence reads as real.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Provenance {
    /// A recording of a genuine deployed-policy inference step on a real robot in the
    /// field. The only provenance admissible for a safety verdict.
    Real,
    /// Synthetic data produced by a generator (e.g. `"cosmos-3"`, `"isaac-sim"`). The
    /// `generator` names what produced it, so a downstream consumer can audit, weight,
    /// or filter by source. Never admissible for a safety verdict.
    Synthetic {
        /// The generator that produced this data (e.g. `"cosmos-3"`, `"isaac-sim"`),
        /// recorded so synthetic data is auditable and filterable by its source.
        generator: String,
    },
}

impl Default for Provenance {
    /// Defaults to `Real` so an old robot or an old stored row that has no provenance
    /// field deserializes as real field evidence — the append stays backward
    /// compatible. A synthetic source must opt in explicitly via [`Provenance::synthetic`].
    fn default() -> Self {
        Provenance::Real
    }
}

impl Provenance {
    /// Construct a synthetic provenance tagging the generator that produced the data.
    #[must_use]
    pub fn synthetic(generator: impl Into<String>) -> Self {
        Provenance::Synthetic {
            generator: generator.into(),
        }
    }

    /// True iff this is real field evidence.
    #[must_use]
    pub fn is_real(&self) -> bool {
        matches!(self, Provenance::Real)
    }

    /// True iff this is synthetic data (sim / augmentation).
    #[must_use]
    pub fn is_synthetic(&self) -> bool {
        matches!(self, Provenance::Synthetic { .. })
    }

    /// True iff this provenance may back a safety verdict — `Real` only.
    ///
    /// Synthetic data must never satisfy a safety verdict: a sim is an approximation,
    /// and the sim-to-real gap means a behavior that passes a safety check in synthetic
    /// data can still fail on a physical robot. So the safety path admits real field
    /// evidence only; synthetic rollouts are barred here regardless of any other signal.
    #[must_use]
    pub fn admissible_for_safety(&self) -> bool {
        matches!(self, Provenance::Real)
    }
}

/// `Rollout` — one deployed-policy inference step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rollout {
    // ---- identity & ordering ------------------------------------------------
    /// UUIDv7; uniqueness + coarse sort ONLY. Its embedded timestamp is advisory
    /// (robot wall-clocks drift) and must never drive attribution.
    pub id: RolloutId,
    /// `(tenant_id, robot_id)` — never robot alone, since a robot id is unique only
    /// within a tenant.
    #[serde(flatten)]
    pub robot: RobotIdentity,
    /// The trajectory this step belongs to; the episode is the implicit grouping
    /// over this id, computed later rather than opened up-front.
    pub episode_id: EpisodeId,
    /// Step ordinal within the episode.
    pub step_index: u32,

    // ---- the two clocks ----------------------------------------------------
    /// `(boot_id, mono_ns, ts_wall_ns)` — the monotonic `(boot_id, mono_ns)` is the
    /// skew-free attribution authority; `ts_wall_ns` is an advisory wall estimate.
    #[serde(flatten)]
    pub clock: MonoClock,
    /// Server-set ingest time + boot anchor (trusted; set by the gateway). `Option`
    /// because it does not exist until the row reaches the gateway — the robot-side
    /// in-flight record has `None`.
    pub server_anchor: Option<ServerAnchor>,

    // ---- policy attribution (a claim until the gateway reconciles it) ------
    /// Self-reported policy version; reconciled against the deployment ledger at
    /// ingest, since a robot's claim about what it runs is not trusted as truth.
    pub policy_version: PolicyVersion,
    /// Self-reported weights hash, reconciled alongside `policy_version`.
    pub model_hash: String,
    /// SERVER-derived trust after ledger reconciliation. `Option` because it is
    /// unset until the gateway reconciles (robot-side record has `None`).
    pub trust: Option<Trust>,
    /// Embodiment (robot type / action-space class) — drives per-embodiment adapter
    /// dispatch and attribution-calibration buckets.
    pub embodiment: String,
    /// The task being attempted.
    pub task_id: String,
    /// Eval context, present only inside an A/B eval window. `None` in normal
    /// production. See [`EvalContext`].
    pub eval: Option<EvalContext>,

    // ---- payload = pointers, never bytes -----------------------------------
    /// Pointer to the observation bytes in the customer's own object store
    /// (mcap segment + msg range). Bytes never enter the metadata plane.
    pub observation_ref: PayloadRef,
    /// Pointer to the action bytes in the customer's own object store.
    pub action_ref: PayloadRef,
    /// Small inline context (JSON), byte-bounded at the SDK and re-validated at the
    /// gateway because it is an open-SDK field. Not a place for sensor bytes.
    pub context: BoundedBlob,
    /// How far `context` (and other inline fields) have been schema-validated.
    pub schema_conformance: SchemaConformance,

    // ---- timing & tags ------------------------------------------------------
    /// Inference wall-time in microseconds (the SDK's own measurement; ops
    /// telemetry, not an attribution input).
    pub inference_us: u32,
    /// Open free-form tags. Byte/rate-bounded at the gateway — an open-SDK field,
    /// untrusted until validated.
    #[serde(default)]
    pub tags: std::collections::BTreeMap<String, String>,

    // ---- real-vs-synthetic provenance --------------------------------------
    /// Whether this rollout is real field evidence or synthetic data (sim / Cosmos
    /// augmentation). `#[serde(default)]` so an old robot or an old stored row with no
    /// provenance field deserializes as `Real` — making this a backward-compatible
    /// append to the otherwise frozen schema, not a breaking change. Curation weights
    /// or separates synthetic data by this field, and the safety path excludes it (see
    /// [`Provenance::admissible_for_safety`]).
    #[serde(default)]
    pub provenance: Provenance,

    // ---- spatial & causal attribution channels -----------------------------
    /// Where this inference step happened, as a rigid-body pose. This is the rollout
    /// side of the spatial co-location test: a cross-boot outcome carrying a pose binds
    /// to the rollout whose pose is within an embodiment epsilon of it. `None` (the
    /// default, and the common case for a non-localized arm) means the rollout offers
    /// no pose channel and so can never be a spatial candidate — the tier skips it with
    /// that reason rather than inventing a location. `#[serde(default)]` keeps the
    /// field a backward-compatible append.
    #[serde(default)]
    pub pose: Option<Se3Pose>,
    /// The reference frame `pose` is expressed in. The spatial tier compares an outcome
    /// to this rollout only when their `frame_id`s match, so two poses in different
    /// frames (a robot-base frame vs a map frame) are never compared as if co-located.
    /// Empty (the default) means no frame declared, which keeps the rollout out of the
    /// spatial tier.
    #[serde(default)]
    pub frame_id: String,
    /// The station / cell this rollout ran at, for causal line-topology attribution. A
    /// downstream outcome's [`crate::outcome::DownstreamEdge`] names an
    /// `upstream_station_id`; the causal tier binds to the rollouts whose `station_id`
    /// matches it within the causal-lag window. Empty (the default) means the rollout
    /// declares no station and so is never a causal candidate. `#[serde(default)]`
    /// keeps this a backward-compatible append.
    #[serde(default)]
    pub station_id: String,
    /// Named scalar signal channels the robot reports alongside the action (e.g.
    /// `gripper_force`, `perception_confidence`). The typed raw material for incident
    /// diagnostics: ranking which signals deviate in a failure cohort versus the baseline. Empty
    /// (the default) means the robot reports no structured signals. `#[serde(default)]` keeps this
    /// a backward-compatible append, so an old record without the field reads back as empty.
    #[serde(default)]
    pub signals: std::collections::BTreeMap<String, f64>,
}

impl Rollout {
    /// Minimal constructor for the robot-side (pre-ingest) record: mints a fresh
    /// [`RolloutId`], leaves the server-set fields (`server_anchor`, `trust`)
    /// `None`, and defaults non-eval production context. The gateway fills in the
    /// trusted fields at ingest.
    ///
    /// This is a builder *stub* — this crate carries no business logic, so real
    /// construction happens in the capture SDK.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        robot: RobotIdentity,
        episode_id: EpisodeId,
        step_index: u32,
        clock: MonoClock,
        policy_version: PolicyVersion,
        model_hash: String,
        embodiment: String,
        task_id: String,
        observation_ref: PayloadRef,
        action_ref: PayloadRef,
        context: BoundedBlob,
        inference_us: u32,
    ) -> Self {
        Self {
            id: RolloutId::new(),
            robot,
            episode_id,
            step_index,
            clock,
            server_anchor: None,
            policy_version,
            model_hash,
            trust: None,
            embodiment,
            task_id,
            eval: None,
            observation_ref,
            action_ref,
            context,
            schema_conformance: SchemaConformance::Unchecked,
            inference_us,
            tags: std::collections::BTreeMap::new(),
            // A rollout built by the capture SDK is, by definition, a real fielded step.
            // Synthetic construction goes through `with_provenance`, never this path, so
            // synthetic data can never be minted as real by accident.
            provenance: Provenance::Real,
            // No localization and no station label on the minimal record: the rollout
            // is invisible to the spatial and causal tiers until a localized SDK fills
            // these in, so the default path is unchanged.
            pose: None,
            frame_id: String::new(),
            station_id: String::new(),
            signals: std::collections::BTreeMap::new(),
        }
    }

    /// Attach a spatial pose in a named frame, returning the modified rollout — the
    /// rollout side of the spatial co-location test. A builder so the common
    /// non-localized path's signature never changes.
    #[must_use]
    pub fn with_pose(mut self, pose: Se3Pose, frame_id: impl Into<String>) -> Self {
        self.pose = Some(pose);
        self.frame_id = frame_id.into();
        self
    }

    /// Tag this rollout with the station it ran at, returning the modified rollout —
    /// the upstream label the causal tier matches a downstream outcome's edge against.
    #[must_use]
    pub fn at_station(mut self, station_id: impl Into<String>) -> Self {
        self.station_id = station_id.into();
        self
    }

    /// Attach a named scalar signal (e.g. `gripper_force`), returning the modified rollout. A
    /// builder so a rollout can accrue signal channels without the constructor growing an
    /// open-ended argument, and the common no-signals path keeps its signature.
    #[must_use]
    pub fn with_signal(mut self, name: impl Into<String>, value: f64) -> Self {
        self.signals.insert(name.into(), value);
        self
    }

    /// Tag this rollout with an explicit provenance, returning the modified rollout.
    ///
    /// A builder for synthetic construction: `Rollout::new(...)` always produces `Real`
    /// (the safe default), and a synthetic-data producer calls
    /// `.with_provenance(Provenance::synthetic("cosmos-3"))` to mark it. Kept as a
    /// separate step so the common, real-capture path's signature never changes and
    /// marking data synthetic is always a deliberate, visible act.
    #[must_use]
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonoClock;
    use crate::ids::BootId;
    use crate::payload::{BoundedBlob, PayloadRef};
    use crate::pose::Se3Pose;
    use crate::tenant::{RobotId, RobotIdentity, TenantId};

    fn sample() -> Rollout {
        Rollout::new(
            RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1")),
            EpisodeId::new(),
            0,
            MonoClock::new(BootId::new(), 1, 1),
            PolicyVersion::new("pick@v1+abc"),
            "sha256:w".into(),
            "ur5e".into(),
            "pick".into(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            10,
        )
    }

    /// Headline backward-compat: a `Rollout` JSON that has NO `provenance` field — an
    /// old robot or a row stored before this append — deserializes as `Real`. This is
    /// what makes the field a safe append to the frozen schema.
    #[test]
    fn old_json_without_provenance_defaults_to_real() {
        let mut json = serde_json::to_value(sample()).unwrap();
        // Simulate an old payload by stripping the field entirely.
        json.as_object_mut().unwrap().remove("provenance");
        assert!(json.get("provenance").is_none());
        let back: Rollout = serde_json::from_value(json).unwrap();
        assert_eq!(back.provenance, Provenance::Real);
        assert!(back.provenance.is_real());
    }

    /// An old `Rollout` JSON with none of the spatial/causal fields deserializes with
    /// `pose = None`, an empty `frame_id`, and an empty `station_id` — so the rollout
    /// is simply invisible to the spatial and causal tiers, never bound on an invented
    /// location or station. This keeps the three new fields a safe schema append.
    #[test]
    fn old_json_without_spatial_causal_fields_defaults_empty() {
        let mut json = serde_json::to_value(sample()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("pose");
        obj.remove("frame_id");
        obj.remove("station_id");
        let back: Rollout = serde_json::from_value(json).unwrap();
        assert!(back.pose.is_none());
        assert!(back.frame_id.is_empty());
        assert!(back.station_id.is_empty());
    }

    /// The pose, frame, and station round-trip through JSON via the builders, so the
    /// rollout side of the spatial co-location test and the causal station match
    /// survive storage.
    #[test]
    fn rollout_pose_and_station_round_trip() {
        let r = sample()
            .with_pose(Se3Pose::at(3.0, 4.0, 0.0), "map")
            .at_station("pick_cell");
        let back: Rollout = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.pose, Some(Se3Pose::at(3.0, 4.0, 0.0)));
        assert_eq!(back.frame_id, "map");
        assert_eq!(back.station_id, "pick_cell");
    }

    /// A synthetic rollout round-trips: the `Synthetic { generator }` is preserved
    /// through JSON, so the generator stays auditable downstream.
    #[test]
    fn synthetic_provenance_round_trips() {
        let r = sample().with_provenance(Provenance::synthetic("cosmos-3"));
        assert!(r.provenance.is_synthetic());
        let back: Rollout = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.provenance, Provenance::synthetic("cosmos-3"));
    }

    /// The safety rule: real is admissible, synthetic never is — regardless of the
    /// generator. Safety must rest on real field evidence only (sim-to-real gap).
    #[test]
    fn admissible_for_safety_real_only() {
        assert!(Provenance::Real.admissible_for_safety());
        assert!(!Provenance::synthetic("isaac-sim").admissible_for_safety());
        assert!(!Provenance::synthetic("cosmos-3").admissible_for_safety());
    }
}
