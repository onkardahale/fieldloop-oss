//! # Cosmos → Fieldloop synthetic-ingest adapter
//!
//! NVIDIA Cosmos is a world-foundation-model stack that GENERATES synthetic robot
//! trajectories on a GPU cluster (the heavy generation step). That generation is
//! external GPU work and is deliberately out of scope here: this adapter does NOT call
//! Cosmos and does NOT run any model. It ingests a **pre-generated Cosmos export** —
//! a LeRobot-shaped dataset (episodes of steps with observation/action object keys and
//! a per-step success label) that a Cosmos generation job already wrote — and maps it
//! into Fieldloop's canonical [`Rollout`] type.
//!
//! The one guarantee this adapter exists to make: **every** rollout it produces is
//! tagged [`Provenance::Synthetic`] with the generator name (`"cosmos-3"`). It is
//! structurally impossible for this path to mint a synthetic step as `Real`, because
//! the rollout is built and then `with_provenance(synthetic)` is applied unconditionally
//! to each item — there is no branch that leaves a step untagged. That tag then rides
//! through the rest of the pipeline (store → curate → retrain → eval), where the safety
//! deploy-gate refuses any policy trained on it. The adapter's job is just the honest
//! tagging at the boundary; the barring is enforced downstream by the tag it sets.
//!
//! Shape mapping (LeRobot-style export → `Rollout`): one Cosmos episode becomes one
//! `EpisodeId` grouping; each step in the episode becomes one `Rollout` whose
//! `step_index` is the step ordinal, whose `observation_ref`/`action_ref` are the
//! object keys the Cosmos job wrote, and whose advisory monotonic clock is synthesized
//! from a fixed boot id plus a per-step monotonic counter (synthetic data has no real
//! robot boot, so the clock is advisory-only and never an attribution authority).

use serde::Deserialize;

use fieldloop_types::{
    BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, Provenance, RobotIdentity,
    Rollout,
};

/// The generator name stamped onto every rollout this adapter produces, so a downstream
/// consumer can audit, filter, or bar by the exact source. Kept as one constant because
/// this adapter ingests exactly one generator's export — Cosmos — and tagging anything
/// else `cosmos-3` would be a provenance lie.
pub const COSMOS_GENERATOR: &str = "cosmos-3";

/// One step of a Cosmos-generated, LeRobot-shaped synthetic episode, as it appears in a
/// pre-generated export. This is the *input* shape (a tiny serde fixture / in-memory
/// struct), NOT a Fieldloop type: it carries only what a LeRobot export records per
/// frame — the step ordinal, the observation/action object keys the Cosmos job wrote,
/// and the synthetic success label. It has no provenance field of its own: provenance is
/// not the dataset's to claim — the adapter assigns it at the boundary, so an export can
/// never smuggle in a `Real` tag.
#[derive(Debug, Clone, Deserialize)]
pub struct CosmosStep {
    /// Step ordinal within the episode (becomes the rollout's `step_index`).
    pub step_index: u32,
    /// Object key the Cosmos job wrote the observation bytes under (becomes the
    /// rollout's `observation_ref`). Empty means no separate observation blob.
    pub observation_key: String,
    /// Object key for the action bytes (becomes the rollout's `action_ref`).
    pub action_key: String,
    /// The synthetic success label the generator assigned to this step. Carried into the
    /// rollout's tags as advisory context (it is generated, not field-observed truth, so
    /// it never becomes a trusted feedback row here).
    pub success: bool,
}

/// One Cosmos-generated, LeRobot-shaped synthetic episode: a task label plus an ordered
/// list of steps. A whole pre-generated export is a `Vec<CosmosEpisode>`.
#[derive(Debug, Clone, Deserialize)]
pub struct CosmosEpisode {
    /// The task the synthetic trajectory attempts (becomes each rollout's `task_id`).
    pub task_id: String,
    /// The ordered steps of the trajectory.
    pub steps: Vec<CosmosStep>,
}

/// Convert a pre-generated Cosmos LeRobot-shaped export into Fieldloop [`Rollout`]s, each
/// tagged [`Provenance::Synthetic`] with generator `"cosmos-3"`.
///
/// `identity` is the `(tenant, robot)` the synthetic data is ingested under (synthetic
/// data is owned by a tenant like any other), `embodiment` is the robot type the Cosmos
/// trajectories target (drives the per-embodiment adapter downstream), and
/// `policy_version` is the policy label the generation was conditioned on (advisory: a
/// synthetic step is not a deployed-policy inference, so this is provenance context, not
/// a trust claim).
///
/// Every produced rollout is unconditionally `with_provenance(synthetic("cosmos-3"))`:
/// there is no code path through this function that yields an untagged or `Real` rollout,
/// which is the whole point — the pipeline downstream can treat the output as synthetic
/// without re-deriving that fact.
pub fn ingest_cosmos_export(
    export: &[CosmosEpisode],
    identity: &RobotIdentity,
    embodiment: &str,
    policy_version: &PolicyVersion,
) -> Vec<Rollout> {
    // Cosmos is one specific generator; delegate to the generator-parameterized ingest so
    // there is exactly one mapping from a synthetic export to tagged rollouts, and stamp
    // the fixed `cosmos-3` name — tagging a non-Cosmos export `cosmos-3` would be a
    // provenance lie, which is why the generic path takes the generator explicitly.
    ingest_synthetic_export(
        export,
        COSMOS_GENERATOR,
        identity,
        embodiment,
        policy_version,
    )
}

/// Convert a pre-generated, LeRobot-shaped synthetic export into Fieldloop [`Rollout`]s,
/// each tagged [`Provenance::Synthetic`] with the caller-supplied `generator` name.
///
/// This is the generator-agnostic core: it carries whatever synthetic source actually
/// produced the export (e.g. a Cosmos world-model job, or an NVIDIA GR00T/Isaac simulation
/// export) so the provenance tag names the *real* generator rather than a single hard-coded
/// label. `identity`/`embodiment`/`policy_version` are the same ingest context as the Cosmos
/// path. Every produced rollout is unconditionally `with_provenance(synthetic(generator))`:
/// there is no path here that yields an untagged or `Real` rollout, so the pipeline
/// downstream can treat the output as synthetic without re-deriving that fact.
pub fn ingest_synthetic_export(
    export: &[CosmosEpisode],
    generator: &str,
    identity: &RobotIdentity,
    embodiment: &str,
    policy_version: &PolicyVersion,
) -> Vec<Rollout> {
    // A fixed boot id for the whole synthetic batch: synthetic data has no real robot
    // boot, and the monotonic clock is advisory-only (never an attribution authority),
    // so a single deterministic anchor is correct and keeps the export reproducible.
    let boot = fieldloop_types::BootId::new();
    let mut mono_ns: u64 = 0;
    let mut out = Vec::new();

    for episode in export {
        // One synthetic episode = one `EpisodeId` grouping over its steps.
        let episode_id = EpisodeId::new();
        for step in &episode.steps {
            mono_ns += 1;
            // The advisory clock: monotonic counter for ordering, a fixed wall stamp
            // since synthetic data has no real wall-clock to report.
            let clock = MonoClock::new(boot, mono_ns, 1_700_000_000_000_000_000);

            let observation_ref = key_to_ref(&step.observation_key);
            let action_ref = key_to_ref(&step.action_key);

            let mut rollout = Rollout::new(
                identity.clone(),
                episode_id,
                step.step_index,
                clock,
                policy_version.clone(),
                // No real weights hash exists for a synthetic step; name the generator so
                // the model_hash field is honest about what produced the trajectory.
                format!("synthetic:{generator}"),
                embodiment.to_string(),
                episode.task_id.clone(),
                observation_ref,
                action_ref,
                BoundedBlob::empty(),
                // Synthetic steps have no measured inference latency; report zero rather
                // than fabricate a timing number that ops telemetry might trust.
                0,
            );
            // Carry the generator's advisory success label as a tag, not as trusted
            // feedback: it is generated, not field-observed, so it informs but does not
            // become a safety-relevant outcome here.
            rollout
                .tags
                .insert("synthetic_success".to_string(), step.success.to_string());
            rollout
                .tags
                .insert("synthetic_generator".to_string(), generator.to_string());

            // The load-bearing line: tag every rollout synthetic/<generator>, unconditionally.
            out.push(rollout.with_provenance(Provenance::synthetic(generator)));
        }
    }

    out
}

/// Map a Cosmos export object key to a [`PayloadRef`]: a non-empty key becomes a pointer
/// to that object; an empty key becomes a "no payload" pointer. Synthetic exports
/// reference whole objects (no sub-ranges) and carry no checksum here (the export is
/// trusted bytes the Cosmos job wrote, not an untrusted upload), so range/checksum stay
/// `None`.
fn key_to_ref(key: &str) -> PayloadRef {
    if key.is_empty() {
        PayloadRef::none()
    } else {
        PayloadRef {
            object_key: key.to_string(),
            range: None,
            content_sha256: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_types::{RobotId, TenantId};

    fn fixture() -> Vec<CosmosEpisode> {
        vec![
            CosmosEpisode {
                task_id: "bin-pick".to_string(),
                steps: vec![
                    CosmosStep {
                        step_index: 0,
                        observation_key: "cosmos/ep0/obs0.bin".to_string(),
                        action_key: "cosmos/ep0/act0.bin".to_string(),
                        success: true,
                    },
                    CosmosStep {
                        step_index: 1,
                        observation_key: "cosmos/ep0/obs1.bin".to_string(),
                        action_key: "cosmos/ep0/act1.bin".to_string(),
                        success: false,
                    },
                ],
            },
            CosmosEpisode {
                task_id: "place".to_string(),
                steps: vec![CosmosStep {
                    step_index: 0,
                    observation_key: "cosmos/ep1/obs0.bin".to_string(),
                    action_key: String::new(),
                    success: true,
                }],
            },
        ]
    }

    fn identity() -> RobotIdentity {
        RobotIdentity::new(TenantId::new("acme"), RobotId::new("sim-robot"))
    }

    /// EVERY produced rollout is tagged synthetic with generator `cosmos-3` — the single
    /// guarantee the adapter exists to make. A miss here would let synthetic data slip
    /// into the pipeline as real and reach a safety gate.
    #[test]
    fn tags_every_rollout_synthetic_cosmos3() {
        let policy = PolicyVersion::new("nav@v1+abcabcabcabc");
        let rollouts = ingest_cosmos_export(&fixture(), &identity(), "arm6dof", &policy);

        assert_eq!(
            rollouts.len(),
            3,
            "two episodes of 2 + 1 steps -> 3 rollouts"
        );
        for r in &rollouts {
            assert_eq!(
                r.provenance,
                Provenance::synthetic("cosmos-3"),
                "every cosmos rollout must be tagged synthetic/cosmos-3"
            );
            assert!(!r.provenance.admissible_for_safety());
        }
    }

    /// The LeRobot shape maps correctly: step ordinals, object keys, task, and the
    /// episode grouping all carry through, and an empty key becomes an empty pointer.
    #[test]
    fn maps_shape_correctly() {
        let policy = PolicyVersion::new("nav@v1+abcabcabcabc");
        let rollouts = ingest_cosmos_export(&fixture(), &identity(), "arm6dof", &policy);

        // First episode's two steps share one episode id; the third is a different episode.
        assert_eq!(rollouts[0].episode_id, rollouts[1].episode_id);
        assert_ne!(rollouts[0].episode_id, rollouts[2].episode_id);

        // Step ordinals and task labels carry through.
        assert_eq!(rollouts[0].step_index, 0);
        assert_eq!(rollouts[1].step_index, 1);
        assert_eq!(rollouts[0].task_id, "bin-pick");
        assert_eq!(rollouts[2].task_id, "place");

        // Object keys map to payload pointers; an empty action key becomes an empty ref.
        assert_eq!(
            rollouts[0].observation_ref.object_key,
            "cosmos/ep0/obs0.bin"
        );
        assert_eq!(rollouts[0].action_ref.object_key, "cosmos/ep0/act0.bin");
        assert!(
            rollouts[2].action_ref.is_empty(),
            "empty key -> empty pointer"
        );

        // The advisory success label rides through as a tag (advisory, not trusted truth).
        assert_eq!(
            rollouts[0]
                .tags
                .get("synthetic_success")
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(rollouts[0].embodiment, "arm6dof");
    }

    /// The input shape is plain serde, so a Cosmos export can arrive as JSON (the realistic
    /// ingest form) and deserialize into the same fixtures the in-memory path uses.
    #[test]
    fn export_deserializes_from_json() {
        let json = r#"[
            {"task_id":"bin-pick","steps":[
                {"step_index":0,"observation_key":"o0","action_key":"a0","success":true}
            ]}
        ]"#;
        let export: Vec<CosmosEpisode> = serde_json::from_str(json).expect("export parses");
        let policy = PolicyVersion::new("nav@v1+abcabcabcabc");
        let rollouts = ingest_cosmos_export(&export, &identity(), "arm6dof", &policy);
        assert_eq!(rollouts.len(), 1);
        assert!(rollouts[0].provenance.is_synthetic());
    }
}
