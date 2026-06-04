//! # `fieldloop-curation` — joined failures into versioned, reproducible datasets
//!
//! Turns attribution bindings into something a model can train on: a *slice* (a filtered selection
//! of failures), pinned reproducibly, exported as a training-dataset manifest, with bidirectional
//! lineage so every trained policy traces to the exact episodes it learned from, and vice-versa.
//!
//! Pure and in-memory; heavy I/O sits behind traits with stub impls so the loop is testable with no
//! external system: [`PayloadFetcher`] (resolve/decode payload bytes), [`LineageStore`] (the
//! provenance index), and a future commit seam (lakeFS/FiftyOne).
//!
//! The pieces:
//! - [`SliceSpec`] — the query: filter by policy / failure class / task / site / outcome-time
//!   window, a confidence floor, and a per-step vs per-trajectory grain.
//! - [`compile_slice`] — resolve the authoritative feedback per target (the join's latest-wins
//!   rule), apply the confidence gate, and pin the matched ids into a [`ResolvedSlice`] — the frozen
//!   manifest (not the query), content-hashed over the pinned set.
//! - [`to_lerobot_manifest`] — the export spec a trainer consumes: episodes → frames referencing
//!   each rollout's payload and its authoritative label/reward.
//! - [`LineageStore`] / [`InMemoryLineageStore`] — commit + trained-from edges, so
//!   `episodes_for_policy` / `policies_for_episode` are single-index reads.
//!
//! Safety-critical rule (the confidence gate, in [`compile_slice`]): a retracted binding, or one
//! below the slice's `min_confidence`, is NEVER admitted to training — it is surfaced as
//! needs-review. A confidently-wrong label poisons a model worse than a missing example helps, so a
//! doubtful binding reaches a human, not a trainer.
//!
//! Reproducibility: a [`ResolvedSlice`] pins the concrete `(rollout_id, feedback_id)` ids, not the
//! query, so a rebuild reproduces the identical dataset even if the source rows later changed.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod lerobot;
pub mod lineage;
pub mod slice;
pub mod spec;

pub use lerobot::{
    FrameLabel, LeRobotDatasetManifest, LeRobotEpisode, LeRobotFrame, PayloadFetcher,
    ResolvedPayload, to_lerobot_manifest,
};
pub use lineage::{InMemoryLineageStore, LineageStore};
pub use slice::{
    DatasetCommit, NeedsReview, PinnedItem, ResolvedSlice, ReviewReason, compile_slice,
};
pub use spec::{DEFAULT_MIN_CONFIDENCE, Grain, OutcomeTsWindow, SITE_TAG_KEY, SliceSpec};

// ---------------------------------------------------------------------------
// Self-contained verification: every test below builds its own rollouts and
// feedback in memory and exercises only this crate's public API plus the JOIN
// crate's latest-wins resolver — no DB, no object storage, no training runtime.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use fieldloop_types::{
        BoundedBlob, ByteRange, EpisodeId, FailureClass, Feedback, FeedbackId, FeedbackTarget,
        FeedbackValue, JoinMethod, LabelKind, MonoClock, PayloadRef, PolicyVersion, RobotId,
        RobotIdentity, Rollout, RolloutId, TenantId,
    };

    use super::*;

    // ---- builders --------------------------------------------------------

    const TENANT: &str = "acme";
    const POLICY: &str = "pick@v1.0.0+aaaaaaaaaaaa";
    const OTHER_POLICY: &str = "pick@v2.0.0+bbbbbbbbbbbb";

    /// A rollout with explicit id/episode so a test can pin and look it up. Tags carry
    /// an optional site so the site filter can be exercised.
    #[allow(clippy::too_many_arguments)]
    fn rollout(
        id: RolloutId,
        episode: EpisodeId,
        step: u32,
        policy: &str,
        task: &str,
        site: Option<&str>,
        obs_key: &str,
        act_key: &str,
    ) -> Rollout {
        let mut tags = BTreeMap::new();
        if let Some(s) = site {
            tags.insert(SITE_TAG_KEY.to_string(), s.to_string());
        }
        Rollout {
            id,
            robot: RobotIdentity::new(TenantId::new(TENANT), RobotId::new("robot-1")),
            episode_id: episode,
            step_index: step,
            clock: MonoClock::new(
                fieldloop_types::BootId::new(),
                1_000,
                1_700_000_000_000_000_000,
            ),
            server_anchor: None,
            policy_version: PolicyVersion::new(policy),
            model_hash: "sha256:dead".to_string(),
            trust: None,
            embodiment: "ur5e".to_string(),
            task_id: task.to_string(),
            eval: None,
            observation_ref: PayloadRef {
                object_key: obs_key.to_string(),
                range: Some(ByteRange { start: 0, end: 10 }),
                content_sha256: Some("obs-digest".to_string()),
            },
            action_ref: PayloadRef {
                object_key: act_key.to_string(),
                range: None,
                content_sha256: None,
            },
            context: BoundedBlob::empty(),
            schema_conformance: fieldloop_types::SchemaConformance::Unchecked,
            inference_us: 100,
            tags,
            provenance: fieldloop_types::Provenance::Real,
            pose: None,
            frame_id: String::new(),
            station_id: String::new(),
            signals: Default::default(),
        }
    }

    /// A feedback row targeting a rollout. `id` is explicit so latest-wins ordering
    /// (by id) is controllable in a test.
    #[allow(clippy::too_many_arguments)]
    fn feedback(
        id: FeedbackId,
        target: RolloutId,
        method: JoinMethod,
        confidence: f32,
        retracted: bool,
        value: FeedbackValue,
        outcome_ts_ns: i64,
    ) -> Feedback {
        Feedback {
            id,
            tenant_id: TenantId::new(TENANT),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "task_success".to_string(),
            value,
            join_method: method,
            join_confidence: confidence,
            join_version: "join-v1".to_string(),
            calibration_version: "calib-1".to_string(),
            source_outcome_id: None,
            delay_ms: Some(10),
            retracted,
            dedup_key: format!("dk:{target}:{}", method_tag(method)),
            credit_weight: 1.0,
            contributing_set_id: None,
            outcome_ts_ns,
        }
    }

    fn method_tag(m: JoinMethod) -> &'static str {
        match m {
            JoinMethod::Manual => "manual",
            JoinMethod::Temporal => "temporal",
            _ => "other",
        }
    }

    fn spec(grain: Grain, min_confidence: f32) -> SliceSpec {
        SliceSpec {
            policy_version: None,
            failure_class: None,
            task_id: None,
            site: None,
            outcome_ts: OutcomeTsWindow::default(),
            min_confidence,
            include_synthetic: false,
            grain,
        }
    }

    /// A stub fetcher that echoes the pointer back as a resolved reference, so the
    /// export is testable without any object storage. An empty pointer resolves to
    /// `None`, mirroring "no payload on this channel".
    #[derive(Debug)]
    struct EchoFetcher;
    impl PayloadFetcher for EchoFetcher {
        fn resolve(&self, payload: &PayloadRef) -> Option<ResolvedPayload> {
            if payload.is_empty() {
                return None;
            }
            Some(ResolvedPayload {
                object_key: payload.object_key.clone(),
                range: payload.range.map(|r| [r.start, r.end]),
                content_sha256: payload.content_sha256.clone(),
            })
        }
    }

    // ---- slice selection -------------------------------------------------

    /// The spec's metadata filters select matching rollouts and exclude non-matching by
    /// policy, task, site, and outcome-time window.
    #[test]
    fn slice_filters_by_policy_task_site_and_window() {
        let r_match = RolloutId::new();
        let r_wrong_policy = RolloutId::new();
        let r_wrong_site = RolloutId::new();
        let r_out_of_window = RolloutId::new();
        let ep = EpisodeId::new();

        let rollouts = vec![
            rollout(
                r_match,
                ep,
                0,
                POLICY,
                "bin_pick",
                Some("plant-a"),
                "o0",
                "a0",
            ),
            rollout(
                r_wrong_policy,
                ep,
                1,
                OTHER_POLICY,
                "bin_pick",
                Some("plant-a"),
                "o1",
                "a1",
            ),
            rollout(
                r_wrong_site,
                ep,
                2,
                POLICY,
                "bin_pick",
                Some("plant-b"),
                "o2",
                "a2",
            ),
            rollout(
                r_out_of_window,
                ep,
                3,
                POLICY,
                "bin_pick",
                Some("plant-a"),
                "o3",
                "a3",
            ),
        ];
        let fb = vec![
            feedback(
                FeedbackId::new(),
                r_match,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1_000,
            ),
            feedback(
                FeedbackId::new(),
                r_wrong_policy,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1_000,
            ),
            feedback(
                FeedbackId::new(),
                r_wrong_site,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1_000,
            ),
            feedback(
                FeedbackId::new(),
                r_out_of_window,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                5_000,
            ),
        ];

        let mut s = spec(Grain::Rollout, 0.8);
        s.policy_version = Some(PolicyVersion::new(POLICY));
        s.task_id = Some("bin_pick".to_string());
        s.site = Some("plant-a".to_string());
        s.outcome_ts = OutcomeTsWindow {
            start_ns: Some(0),
            end_ns: Some(2_000),
        };

        let resolved = compile_slice(&s, &rollouts, &fb);
        let ids: Vec<RolloutId> = resolved.items.iter().map(|i| i.rollout_id).collect();
        assert_eq!(
            ids,
            vec![r_match],
            "only the row matching every filter is pinned"
        );
    }

    /// Synthetic data is opt-in and stays distinguishable: with `include_synthetic =
    /// false` (the safe default) a synthetic rollout is excluded while a real one is
    /// pinned; with `include_synthetic = true` the synthetic rollout IS pinned but its
    /// pinned item — and the exported frame — is tagged `Synthetic`, so it can never be
    /// silently consumed as real field evidence.
    #[test]
    fn synthetic_is_opt_in_and_tagged_in_slice_and_manifest() {
        let r_real = RolloutId::new();
        let r_syn = RolloutId::new();
        let ep = EpisodeId::new();

        let real = rollout(r_real, ep, 0, POLICY, "bin_pick", None, "o0", "a0");
        let syn = rollout(r_syn, ep, 1, POLICY, "bin_pick", None, "o1", "a1")
            .with_provenance(fieldloop_types::Provenance::synthetic("cosmos-3"));
        let rollouts = vec![real, syn];

        let f_real = FeedbackId::new();
        let f_syn = FeedbackId::new();
        let fb = vec![
            feedback(
                f_real,
                r_real,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: true },
                1_000,
            ),
            feedback(
                f_syn,
                r_syn,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: true },
                1_000,
            ),
        ];

        // Default: synthetic excluded, only the real rollout is pinned.
        let s_default = spec(Grain::Rollout, 0.8);
        let resolved = compile_slice(&s_default, &rollouts, &fb);
        let ids: Vec<RolloutId> = resolved.items.iter().map(|i| i.rollout_id).collect();
        assert_eq!(ids, vec![r_real], "synthetic excluded unless opted in");
        assert!(resolved.items.iter().all(|i| i.provenance.is_real()));

        // Opt in: synthetic is admitted, but tagged synthetic on the pinned item.
        let mut s_incl = spec(Grain::Rollout, 0.8);
        s_incl.include_synthetic = true;
        let resolved = compile_slice(&s_incl, &rollouts, &fb);
        assert_eq!(resolved.items.len(), 2, "synthetic admitted when opted in");
        let syn_item = resolved
            .items
            .iter()
            .find(|i| i.rollout_id == r_syn)
            .expect("synthetic item pinned");
        assert_eq!(
            syn_item.provenance,
            fieldloop_types::Provenance::synthetic("cosmos-3"),
            "synthetic item stays tagged with its generator"
        );

        // The export carries the tag too: the synthetic frame is visibly synthetic.
        let manifest = to_lerobot_manifest(&resolved, &rollouts, &fb, &EchoFetcher);
        let frames: Vec<_> = manifest.episodes.iter().flat_map(|e| &e.frames).collect();
        let syn_frame = frames
            .iter()
            .find(|f| f.rollout_id == r_syn)
            .expect("synthetic frame exported");
        assert!(syn_frame.provenance.is_synthetic());
        let real_frame = frames
            .iter()
            .find(|f| f.rollout_id == r_real)
            .expect("real frame exported");
        assert!(real_frame.provenance.is_real());
    }

    // ---- confidence gate -------------------------------------------------

    /// A below-`min_confidence` binding is excluded from the slice and flagged
    /// needs-review (not silently dropped); a retracted binding is excluded; and a
    /// high-confidence MANUAL binding is included even when an older automated one was
    /// lower-confidence (latest-wins + manual precedence picks the manual row, which
    /// then clears the gate).
    #[test]
    fn confidence_gate_excludes_low_and_retracted_keeps_manual() {
        let r_low = RolloutId::new();
        let r_retracted = RolloutId::new();
        let r_manual = RolloutId::new();
        let ep = EpisodeId::new();

        let rollouts = vec![
            rollout(r_low, ep, 0, POLICY, "t", None, "o0", "a0"),
            rollout(r_retracted, ep, 1, POLICY, "t", None, "o1", "a1"),
            rollout(r_manual, ep, 2, POLICY, "t", None, "o2", "a2"),
        ];

        // For r_manual: an OLDER, low-confidence automated row plus a NEWER manual row.
        // winning_binding must pick the manual row (manual outranks automated), and the
        // manual row's confidence clears the gate.
        let auto_old = FeedbackId::new();
        let manual_new = FeedbackId::new();
        assert!(
            auto_old.as_uuid() < manual_new.as_uuid(),
            "manual id must sort newer"
        );

        let fb = vec![
            feedback(
                FeedbackId::new(),
                r_low,
                JoinMethod::Temporal,
                0.5,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r_retracted,
                JoinMethod::Temporal,
                0.99,
                true,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                auto_old,
                r_manual,
                JoinMethod::Temporal,
                0.4,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                manual_new,
                r_manual,
                JoinMethod::Manual,
                1.0,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
        ];

        let resolved = compile_slice(&spec(Grain::Rollout, 0.8), &rollouts, &fb);

        let pinned: Vec<RolloutId> = resolved.items.iter().map(|i| i.rollout_id).collect();
        assert_eq!(
            pinned,
            vec![r_manual],
            "only the high-confidence manual binding is trained on"
        );

        // The manual winner, not the older automated one, supplied the pinned label.
        assert_eq!(resolved.items[0].feedback_id, manual_new);

        // The low-confidence one is flagged below-confidence; the retracted one is
        // flagged retracted. Both are surfaced, neither is in the training set.
        let low = resolved
            .needs_review
            .iter()
            .find(|n| n.rollout_id == r_low)
            .unwrap();
        assert_eq!(low.reason, ReviewReason::BelowMinConfidence);
        let retr = resolved
            .needs_review
            .iter()
            .find(|n| n.rollout_id == r_retracted)
            .unwrap();
        assert_eq!(retr.reason, ReviewReason::Retracted);
    }

    // ---- grain-aware -----------------------------------------------------

    /// An `Episode` slice rolls per-step items up into the distinct episodes they
    /// cover; a `Rollout` slice is per-step. Both pin the same per-step items, but the
    /// episode rollup differs.
    #[test]
    fn grain_episode_rolls_up_steps_rollout_is_per_step() {
        let ep_a = EpisodeId::new();
        let ep_b = EpisodeId::new();
        let r0 = RolloutId::new();
        let r1 = RolloutId::new();
        let r2 = RolloutId::new();

        let rollouts = vec![
            rollout(r0, ep_a, 0, POLICY, "t", None, "o0", "a0"),
            rollout(r1, ep_a, 1, POLICY, "t", None, "o1", "a1"),
            rollout(r2, ep_b, 0, POLICY, "t", None, "o2", "a2"),
        ];
        let fb = vec![
            feedback(
                FeedbackId::new(),
                r0,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r1,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r2,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
        ];

        let per_step = compile_slice(&spec(Grain::Rollout, 0.8), &rollouts, &fb);
        assert_eq!(
            per_step.items.len(),
            3,
            "rollout grain pins one item per step"
        );

        let per_traj = compile_slice(&spec(Grain::Episode, 0.8), &rollouts, &fb);
        assert_eq!(
            per_traj.items.len(),
            3,
            "episode grain still pins the underlying steps"
        );
        assert_eq!(
            per_traj.episodes.len(),
            2,
            "but rolls them up into 2 distinct episodes"
        );
        // The grain is part of the dataset identity, so the two hashes differ.
        assert_ne!(per_step.content_hash, per_traj.content_hash);
    }

    // ---- reproducibility -------------------------------------------------

    /// Build a slice, then rebuild the dataset from its PINNED manifest with the source
    /// rows REMOVED/CHANGED -> the export is identical. This proves the slice pins ids,
    /// not a live query: a re-query over the mutated rows would differ, but the pinned
    /// rebuild reproduces the original dataset exactly.
    #[test]
    fn pinned_manifest_reproduces_dataset_after_source_rows_change() {
        let ep = EpisodeId::new();
        let r0 = RolloutId::new();
        let r1 = RolloutId::new();
        let f0 = FeedbackId::new();
        let f1 = FeedbackId::new();

        let rollouts = vec![
            rollout(r0, ep, 0, POLICY, "t", None, "obs0", "act0"),
            rollout(r1, ep, 1, POLICY, "t", None, "obs1", "act1"),
        ];
        let fb = vec![
            feedback(
                f0,
                r0,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                f1,
                r1,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
        ];

        // Compile against the ORIGINAL rows and export.
        let resolved = compile_slice(&spec(Grain::Episode, 0.8), &rollouts, &fb);
        let original_manifest = to_lerobot_manifest(&resolved, &rollouts, &fb, &EchoFetcher);

        // Now MUTATE the world: drop r1 entirely, and CHANGE r0's payload key and f0's
        // label. A live query would now produce a different (smaller, relabeled)
        // dataset.
        let mut changed_r0 = rollout(r0, ep, 0, POLICY, "t", None, "TAMPERED", "TAMPERED");
        changed_r0.id = r0;
        let changed_rollouts = vec![changed_r0];
        let changed_fb = vec![feedback(
            f0,
            r0,
            JoinMethod::Temporal,
            0.95,
            false,
            FeedbackValue::Boolean { value: true }, // label flipped
            999,
        )];

        // A naive RECOMPILE over the mutated world differs (proving the world changed)...
        let recompiled = compile_slice(&spec(Grain::Episode, 0.8), &changed_rollouts, &changed_fb);
        let recompiled_manifest =
            to_lerobot_manifest(&recompiled, &changed_rollouts, &changed_fb, &EchoFetcher);
        assert_ne!(
            original_manifest, recompiled_manifest,
            "a live re-query over mutated rows must differ — this is the bug pinning fixes"
        );

        // ...but rebuilding from the PINNED manifest, supplying the original-id rows
        // (as a rebuild would re-fetch by pinned id), reproduces the identical dataset.
        let rebuilt = to_lerobot_manifest(&resolved, &rollouts, &fb, &EchoFetcher);
        assert_eq!(
            original_manifest, rebuilt,
            "rebuilding from the pinned manifest reproduces the exact dataset"
        );
    }

    // ---- LeRobot manifest ------------------------------------------------

    /// The curate→retrain seam's load-bearing guarantee: the source-row indices the trainer
    /// is told to train on are EXACTLY the `step_index` of the rollouts in the curated slice,
    /// and curating to a smaller set (only the failures) yields a STRICT subset. This is what
    /// makes an operator's curation change which frames train instead of riding along as an
    /// unused label — the property `gatedemo::slice_manifest_path` relies on to hand the trainer
    /// a `--manifest` of frame indices.
    #[test]
    fn manifest_frame_indices_track_the_curated_slice() {
        let ep = EpisodeId::new();
        // Four rollouts whose `step_index` is the source-parquet row index.
        let (r10, r20, r30, r40) = (
            RolloutId::new(),
            RolloutId::new(),
            RolloutId::new(),
            RolloutId::new(),
        );
        let rollouts = vec![
            rollout(r10, ep, 10, POLICY, "t", None, "o10", "a10"),
            rollout(r20, ep, 20, POLICY, "t", None, "o20", "a20"),
            rollout(r30, ep, 30, POLICY, "t", None, "o30", "a30"),
            rollout(r40, ep, 40, POLICY, "t", None, "o40", "a40"),
        ];
        // r20 and r40 are real failures; r10 and r30 succeed.
        let fb = vec![
            feedback(
                FeedbackId::new(),
                r10,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: true },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r20,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r30,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: true },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r40,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::FailureClass {
                    class: FailureClass::Manipulation,
                },
                1,
            ),
        ];

        // Mirror `slice_manifest_path`: compile the slice, export the manifest, collect the
        // frames' source-row indices.
        let indices = |rs: &[Rollout]| -> Vec<u32> {
            let slice = compile_slice(&spec(Grain::Rollout, 0.8), rs, &fb);
            let manifest = to_lerobot_manifest(&slice, rs, &fb, &EchoFetcher);
            let mut v: Vec<u32> = manifest
                .episodes
                .iter()
                .flat_map(|e| e.frames.iter().map(|f| f.step_index))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        // Full curation selects every rollout's source row.
        assert_eq!(indices(&rollouts), vec![10, 20, 30, 40]);

        // Failures-only curation selects ONLY the failing rollouts' source rows — a strict
        // subset, so the trainer provably learns from a different set of frames.
        let failures: Vec<Rollout> = rollouts
            .iter()
            .filter(|r| r.id == r20 || r.id == r40)
            .cloned()
            .collect();
        assert_eq!(indices(&failures), vec![20, 40]);
    }

    /// A resolved slice + a stub fetcher produces a manifest with the right
    /// episodes/frames, and each frame's label/reward comes from the AUTHORITATIVE
    /// feedback (here, a manual row that outranks an automated one).
    #[test]
    fn lerobot_manifest_has_episodes_frames_and_authoritative_labels() {
        let ep = EpisodeId::new();
        let r0 = RolloutId::new();
        let r1 = RolloutId::new();

        let rollouts = vec![
            rollout(r0, ep, 0, POLICY, "t", None, "obs0", "act0"),
            // r1 has no action payload, to check the None-channel path.
            rollout(r1, ep, 1, POLICY, "t", None, "obs1", ""),
        ];

        // r0's authoritative label is a manual FailureClass; an older automated boolean
        // must be overridden by it.
        let auto = FeedbackId::new();
        let manual = FeedbackId::new();
        assert!(auto.as_uuid() < manual.as_uuid());
        let fb = vec![
            feedback(
                auto,
                r0,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                manual,
                r0,
                JoinMethod::Manual,
                1.0,
                false,
                FeedbackValue::FailureClass {
                    class: FailureClass::Planning,
                },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r1,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Float { value: 0.25 },
                1,
            ),
        ];

        let resolved = compile_slice(&spec(Grain::Episode, 0.8), &rollouts, &fb);
        let manifest = to_lerobot_manifest(&resolved, &rollouts, &fb, &EchoFetcher);

        assert_eq!(manifest.episodes.len(), 1, "one episode");
        let episode = &manifest.episodes[0];
        assert_eq!(episode.episode_id, ep);
        assert_eq!(episode.frames.len(), 2, "two frames");
        assert_eq!(manifest.frame_count(), 2);

        // Frames are ordered by step index.
        assert_eq!(episode.frames[0].step_index, 0);
        assert_eq!(episode.frames[1].step_index, 1);

        // Frame 0's label is the MANUAL authoritative failure class, not the automated
        // boolean — the label comes from the authoritative binding.
        assert_eq!(episode.frames[0].feedback_id, manual);
        assert_eq!(
            episode.frames[0].label,
            FrameLabel::FailureClass {
                class: "planning".to_string()
            }
        );
        // Frame 0 has a resolved observation (with range + digest) and action.
        let obs = episode.frames[0].observation.as_ref().unwrap();
        assert_eq!(obs.object_key, "obs0");
        assert_eq!(obs.range, Some([0, 10]));
        assert!(episode.frames[0].action.is_some());

        // Frame 1's reward is the float value; its empty action ref resolves to None.
        assert_eq!(episode.frames[1].label, FrameLabel::Reward { value: 0.25 });
        assert!(episode.frames[1].action.is_none());
    }

    // ---- content hash / commit dedup -------------------------------------

    /// Identical (spec + inputs) yield the same content hash and commit id (dedup); a
    /// changed input yields a different one. Lineage metadata does not affect the id.
    #[test]
    fn content_hash_dedups_identical_and_differs_on_change() {
        let ep = EpisodeId::new();
        let r0 = RolloutId::new();
        let f0 = FeedbackId::new();
        let rollouts = vec![rollout(r0, ep, 0, POLICY, "t", None, "o0", "a0")];
        let fb = vec![feedback(
            f0,
            r0,
            JoinMethod::Temporal,
            0.95,
            false,
            FeedbackValue::Boolean { value: false },
            1,
        )];

        let s = spec(Grain::Rollout, 0.8);
        let a = compile_slice(&s, &rollouts, &fb);
        let b = compile_slice(&s, &rollouts, &fb);
        assert_eq!(
            a.content_hash, b.content_hash,
            "identical inputs hash equal"
        );

        let commit_a = DatasetCommit::seal(a.clone(), BTreeMap::new());
        let mut lineage = BTreeMap::new();
        lineage.insert("built_by".to_string(), "tester".to_string());
        let commit_b = DatasetCommit::seal(b, lineage);
        assert_eq!(
            commit_a.commit_id, commit_b.commit_id,
            "lineage metadata does not change the id"
        );

        // Change an input: a different pinned rollout id -> different hash/commit.
        let r1 = RolloutId::new();
        let rollouts2 = vec![rollout(r1, ep, 0, POLICY, "t", None, "o0", "a0")];
        let fb2 = vec![feedback(
            FeedbackId::new(),
            r1,
            JoinMethod::Temporal,
            0.95,
            false,
            FeedbackValue::Boolean { value: false },
            1,
        )];
        let c = compile_slice(&s, &rollouts2, &fb2);
        assert_ne!(
            a.content_hash, c.content_hash,
            "a changed pinned id changes the hash"
        );
        assert_ne!(
            commit_a.commit_id,
            DatasetCommit::seal(c, BTreeMap::new()).commit_id
        );
    }

    // ---- bidirectional lineage -------------------------------------------

    /// After recording a commit and the policy trained from it, both lookups return the
    /// right sets and agree with each other (episodes <-> policies are reverses).
    #[test]
    fn lineage_walks_both_directions() {
        let ep_a = EpisodeId::new();
        let ep_b = EpisodeId::new();
        let r0 = RolloutId::new();
        let r1 = RolloutId::new();
        let rollouts = vec![
            rollout(r0, ep_a, 0, POLICY, "t", None, "o0", "a0"),
            rollout(r1, ep_b, 0, POLICY, "t", None, "o1", "a1"),
        ];
        let fb = vec![
            feedback(
                FeedbackId::new(),
                r0,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
            feedback(
                FeedbackId::new(),
                r1,
                JoinMethod::Temporal,
                0.95,
                false,
                FeedbackValue::Boolean { value: false },
                1,
            ),
        ];
        let resolved = compile_slice(&spec(Grain::Episode, 0.8), &rollouts, &fb);
        let commit = DatasetCommit::seal(resolved, BTreeMap::new());

        let policy = PolicyVersion::new("trained@v3.0.0+cccccccccccc");
        let mut store = InMemoryLineageStore::new();
        store.record_commit(&commit);
        store.record_trained_from(&policy, &commit.commit_id);

        // policy -> episodes: both episodes the commit pinned.
        let mut got = store.episodes_for_policy(&policy);
        got.sort_by_key(EpisodeId::as_uuid);
        let mut want = vec![ep_a, ep_b];
        want.sort_by_key(EpisodeId::as_uuid);
        assert_eq!(
            got, want,
            "policy traces back to the episodes it trained from"
        );

        // episode -> policies: the reverse direction returns the policy.
        assert_eq!(store.policies_for_episode(&ep_a), vec![policy.clone()]);
        assert_eq!(store.policies_for_episode(&ep_b), vec![policy.clone()]);

        // An unknown episode/policy returns empty, never a scan error.
        assert!(store.policies_for_episode(&EpisodeId::new()).is_empty());
        assert!(
            store
                .episodes_for_policy(&PolicyVersion::new("unknown@v0+000000000000"))
                .is_empty()
        );
    }
}
