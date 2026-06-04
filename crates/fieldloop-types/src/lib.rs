//! # `fieldloop-types` — the frozen canonical schema
//!
//! The one schema every component agrees on: the data threading the loop — capture mints it, the
//! sidecar ships it, ingest writes it, the join binds it, curation/training/eval consume it.
//!
//! FROZEN: a breaking change ripples to robots already running in the field, so treat additions as
//! append-only and a breaking change as a fleet-wide migration.
//!
//! Contract baked into the types:
//! - The hot 50Hz control loop never blocks: a [`Rollout`] is flat and cheap, ids are lock-free
//!   `Uuid::now_v7()` mints ([`ids`]).
//! - Two clocks, one id: ids are UUIDv7 for uniqueness + coarse sort only (embedded time advisory);
//!   attribution keys off the monotonic `(boot_id, mono_ns)` clock, cross-boot off server-anchored
//!   ingest time ([`clock`]).
//! - Metadata plane, not data plane: rows carry [`payload::PayloadRef`] pointers, never sensor
//!   bytes; only small bounded inline fields ([`payload::BoundedBlob`]) ride along.
//! - Provenance: `policy_version` is on every row; it is a robot self-reported *claim*, and
//!   [`rollout::Trust`] is set only after the gateway reconciles it against the deploy ledger.
//! - Tenant identity is the indivisible `(tenant_id, robot_id)` ([`tenant::RobotIdentity`]), never
//!   `robot_id` alone — a cross-tenant mix-up cannot be constructed.
//! - Untrusted by default: open SDK fields are bounded blobs flagged for re-validation, "kinds" are
//!   closed enums (no free strings on the trusted path), trust/conformance are explicit.
//!
//! Distinct id newtypes ([`ids::RolloutId`] vs [`ids::EpisodeId`]) make a wrong id a compile error,
//! and polymorphic columns are sum types ([`feedback::FeedbackTarget`], [`feedback::FeedbackValue`]).
//! `Feedback.dedup_key` is a plain string, not a hash-seeded v7, so latest-write-wins ordering holds.
//!
//! Types + minimal constructors only — no business logic, no I/O.

// Lint posture: a frozen schema crate should be loud about under-documentation
// and sloppy public surface, but it is not a place for runtime panics.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod clock;
pub mod feedback;
pub mod ids;
pub mod outcome;
pub mod payload;
pub mod policy;
pub mod pose;
pub mod rollout;
pub mod tenant;

// ---- Re-exports of the key types, so consumers can `use fieldloop_types::X`. ----

pub use clock::{MonoClock, ServerAnchor};
pub use feedback::{
    FailureClass, Feedback, FeedbackSource, FeedbackTarget, FeedbackValue, JoinMethod, LabelKind,
};
pub use ids::{BootId, EpisodeId, EvalRunId, FeedbackId, IdParseError, OutcomeId, RolloutId};
pub use outcome::{DownstreamEdge, OutcomeEvent, OutcomeKind};
pub use payload::{BoundedBlob, ByteRange, PayloadRef};
pub use policy::{ArtifactFormat, PolicyProvenance, PolicyVersion, PolicyVersionRecord};
pub use pose::Se3Pose;
pub use rollout::{EvalContext, Provenance, Rollout, SchemaConformance, Trust};
pub use tenant::{RobotId, RobotIdentity, TenantId};

// ---------------------------------------------------------------------------
// Self-contained verification: these tests exercise only this crate's own public
// API plus `serde_json`, so the crate is a closed loop — its types round-trip
// through their documented `Display`/`FromStr` and serde forms without any
// external schema, DB, or harness.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    use crate::clock::MonoClock;
    use crate::ids::BootId;
    use crate::payload::{BoundedBlob, PayloadRef};

    /// (a) An id newtype round-trips through `Display` -> `FromStr`, and the
    /// `FromStr` of an invalid string reports the newtype that failed.
    #[test]
    fn rollout_id_round_trips_through_display_and_from_str() {
        let id = RolloutId::new();
        let text = id.to_string();
        let parsed = RolloutId::from_str(&text).expect("a freshly-minted id must re-parse");
        assert_eq!(id, parsed, "Display -> FromStr must be lossless");

        // The error names the newtype that failed, so a parse failure is diagnosable.
        let err = RolloutId::from_str("definitely-not-a-uuid").unwrap_err();
        assert_eq!(err.kind, "RolloutId");
    }

    /// (a) Every id newtype is `#[serde(transparent)]`, so its JSON form is exactly
    /// the inner UUID string and it round-trips through `serde_json`.
    #[test]
    fn id_newtypes_round_trip_through_serde_json() {
        let id = EpisodeId::new();
        let json = serde_json::to_string(&id).expect("serialize id");
        // Transparent: the JSON is just the quoted UUID string, nothing more.
        assert_eq!(json, format!("\"{id}\""));
        let back: EpisodeId = serde_json::from_str(&json).expect("deserialize id");
        assert_eq!(id, back);

        // A distinct newtype is wire-compatible with a bare UUID but type-distinct
        // in Rust: an OutcomeId serializes to the same shape and round-trips too.
        let oid = OutcomeId::new();
        let oback: OutcomeId = serde_json::from_str(&serde_json::to_string(&oid).unwrap()).unwrap();
        assert_eq!(oid, oback);
    }

    fn sample_feedback() -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(RolloutId::new()),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "task_success".to_string(),
            value: FeedbackValue::Boolean { value: true },
            join_method: JoinMethod::Temporal,
            join_confidence: 0.87,
            join_version: "join-v3".to_string(),
            calibration_version: "calib-2026-01".to_string(),
            source_outcome_id: Some(OutcomeId::new()),
            delay_ms: Some(420),
            retracted: false,
            dedup_key: "out:abc|tgt:def|join-v3|digest:001".to_string(),
            outcome_ts_ns: 1_700_000_000_000_000_000,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    /// (b) A `Feedback` value carrying a `FeedbackTarget`, a `FeedbackValue`, and a
    /// `LabelKind` round-trips through `serde_json` to JSON and back equal.
    #[test]
    fn feedback_round_trips_through_serde_json_equal() {
        let fb = sample_feedback();
        let json = serde_json::to_string(&fb).expect("serialize Feedback");
        let back: Feedback = serde_json::from_str(&json).expect("deserialize Feedback");
        assert_eq!(
            fb, back,
            "Feedback must survive a JSON round-trip unchanged"
        );
    }

    /// (b) The round-trip also holds for the episode-grain target and the other
    /// value shapes, so the sum-type tagging is stable across variants.
    #[test]
    fn feedback_value_and_target_variants_round_trip() {
        let mut fb = sample_feedback();
        fb.target = FeedbackTarget::Episode(EpisodeId::new());
        fb.label_kind = LabelKind::EpisodeReturn;
        fb.value = FeedbackValue::Float { value: 1.5 };
        let back: Feedback = serde_json::from_str(&serde_json::to_string(&fb).unwrap()).unwrap();
        assert_eq!(fb, back);

        fb.value = FeedbackValue::FailureClass {
            class: FailureClass::Perception,
        };
        let back2: Feedback = serde_json::from_str(&serde_json::to_string(&fb).unwrap()).unwrap();
        assert_eq!(fb, back2);

        // synthetic_absence: no source outcome, no delay — the Option `None`s
        // survive the round-trip rather than collapsing to a sentinel.
        fb.join_method = JoinMethod::SyntheticAbsence;
        fb.source_outcome_id = None;
        fb.delay_ms = None;
        let back3: Feedback = serde_json::from_str(&serde_json::to_string(&fb).unwrap()).unwrap();
        assert_eq!(fb, back3);
        assert!(back3.source_outcome_id.is_none());
    }

    /// (c) A `Rollout` serializes to JSON without panicking, via the crate's own
    /// `Rollout::new` builder stub.
    #[test]
    fn rollout_serializes_to_json() {
        let rollout = Rollout::new(
            RobotIdentity::new(TenantId::new("acme"), RobotId::new("robot-7")),
            EpisodeId::new(),
            0,
            MonoClock::new(BootId::new(), 1_000_000, 1_700_000_000_000_000_000),
            PolicyVersion::new("pick@v1.2.0+abc123def456"),
            "sha256:deadbeef".to_string(),
            "ur5e".to_string(),
            "bin_pick".to_string(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            850,
        );

        let json = serde_json::to_string(&rollout).expect("Rollout must serialize");
        assert!(!json.is_empty());
        // The pre-ingest record leaves the server-set fields unset.
        assert!(rollout.server_anchor.is_none());
        assert!(rollout.trust.is_none());
        // Round-trips back to an equal value as a bonus sanity check.
        let back: Rollout = serde_json::from_str(&json).expect("Rollout must deserialize");
        assert_eq!(rollout, back);
    }

    /// The safety-eligibility helper guards on confidence and retraction, the
    /// confidence half of the safety gate.
    #[test]
    fn safety_eligibility_requires_full_confidence_and_not_retracted() {
        let mut fb = sample_feedback();
        fb.join_confidence = 1.0;
        assert!(fb.is_safety_eligible_confidence());

        fb.retracted = true;
        assert!(!fb.is_safety_eligible_confidence());

        fb.retracted = false;
        fb.join_confidence = 0.999;
        assert!(!fb.is_safety_eligible_confidence());
    }
}
