//! # `fieldloop-import` — MCAP log → attribution input
//!
//! The file-import path: read a finalized MCAP recording plus a small topic-mapping TOML
//! and produce the typed [`Rollout`]s and [`OutcomeEvent`]s the attribution engine
//! consumes — so a team can attribute a takeover or e-stop in logs they already have,
//! without first writing a capture integration.
//!
//! A generic MCAP carries no notion of tenant, robot, episode, or policy, so the mapping
//! supplies those once. This is the single-boot v1: one file is treated as one robot, one
//! boot, one episode — stated, not inferred, because a delta on the monotonic clock is
//! only meaningful within a single boot. The mapping also declares which topics carry
//! policy decisions and which carry outcomes (with the outcome kind), and which MCAP
//! timestamp to read as that monotonic clock. Topics named in neither role are ignored, so
//! the many sensor streams in a real log cost nothing and need no declaration.
//!
//! Pure and deterministic: the caller hands in bytes (this crate does no file I/O of its
//! own), and the same file always yields the same rollouts in the same order — `step_index`
//! is assigned in clock order so a re-import never renumbers a decision.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::collections::BTreeMap;

use serde::Deserialize;

use fieldloop_types::{
    BootId, BoundedBlob, EpisodeId, MonoClock, OutcomeEvent, OutcomeKind, PayloadRef,
    PolicyVersion, RobotId, RobotIdentity, Rollout, TenantId,
};

pub mod doctor;

#[cfg(test)]
mod test_util;

pub use doctor::{
    DEFAULT_CLOCK_SKEW_THRESHOLD_NS, DiagnosedRole, Diagnosis, TopicReport, diagnose,
};

/// Which MCAP timestamp to treat as the monotonic attribution clock. MCAP stamps every
/// message with both a `log_time` (when the recorder logged it) and a `publish_time`
/// (when the source node published it); under the single-boot assumption either is a
/// usable monotonic reference, but they can differ, so the choice is explicit rather than
/// guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockSource {
    /// The recorder's log timestamp — present on every MCAP message, so the safe default.
    #[default]
    LogTime,
    /// The publisher's timestamp, for stacks that set one meaningfully upstream.
    PublishTime,
}

/// The `[clock]` table. Absent entirely, it defaults to `log_time`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ClockConfig {
    /// Which timestamp drives the monotonic clock.
    #[serde(default)]
    pub source: ClockSource,
}

/// A topic whose every message is a policy decision — one [`Rollout`] per message.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionTopic {
    /// The MCAP topic name (e.g. `/policy/action`).
    pub topic: String,
}

/// A topic whose every message is an outcome of `outcome_kind`. The kind is typed against
/// the closed [`OutcomeKind`] taxonomy, so an unknown spelling is rejected by the TOML
/// parse (with the valid kinds listed) and the taxonomy stays defined in one place.
#[derive(Debug, Clone, Deserialize)]
pub struct OutcomeTopic {
    /// The MCAP topic name (e.g. `/safety/estop`).
    pub topic: String,
    /// The kind of outcome every message on this topic represents.
    pub outcome_kind: OutcomeKind,
}

/// The full topic-mapping: per-file identity constants plus the decision/outcome topic
/// declarations. Built and validated through [`MappingConfig::from_toml`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappingConfig {
    /// Owning tenant for every record decoded from this file.
    pub tenant_id: String,
    /// The robot that produced this file.
    pub robot_id: String,
    /// The single boot this file covers (UUID). Every record shares it, so the monotonic
    /// clock is comparable across the whole file.
    pub boot_id: String,
    /// The single episode this file covers (UUID).
    pub episode_id: String,
    /// Embodiment name; selects the attribution windows applied downstream.
    pub embodiment: String,
    /// Policy version stamped on every decision rollout.
    pub policy_version: String,
    /// Task id stamped on every decision rollout.
    pub task_id: String,
    /// Optional model hash; empty when the source stack records none.
    #[serde(default)]
    pub model_hash: String,
    /// Clock selection; defaults to `log_time`.
    #[serde(default)]
    pub clock: ClockConfig,
    /// Topics carrying policy decisions.
    #[serde(default)]
    pub decisions: Vec<DecisionTopic>,
    /// Topics carrying outcomes.
    #[serde(default)]
    pub outcomes: Vec<OutcomeTopic>,
}

/// The role a mapped topic plays, resolved once so message classification is a single map
/// lookup per message rather than a scan of both topic lists.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TopicRole {
    Decision,
    Outcome(OutcomeKind),
}

/// Why an import could not proceed. Separates a bad mapping (the user's config) from an
/// undecodable file (the bytes) so the caller knows which to fix.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// The mapping TOML failed to parse or violated the schema (unknown field, unknown
    /// outcome kind, wrong type, missing required key).
    #[error("invalid mapping: {0}")]
    Mapping(#[from] toml::de::Error),
    /// The bytes could not be decoded as a finalized MCAP file.
    #[error("could not decode MCAP: {0}")]
    Mcap(#[from] mcap::McapError),
    /// One topic was declared in more than one role (both decision and outcome, or twice
    /// in one list). A topic carries exactly one role, so the ambiguity is rejected rather
    /// than resolved by declaration order.
    #[error("topic `{topic}` is declared more than once; each topic may carry only one role")]
    DuplicateTopic {
        /// The doubly-declared topic.
        topic: String,
    },
    /// A UUID identity field (`boot_id` / `episode_id`) was not a valid UUID.
    #[error("field `{field}` is not a valid UUID: {message}")]
    BadId {
        /// The offending field name.
        field: &'static str,
        /// The parser's reason.
        message: String,
    },
}

/// The typed attribution inputs decoded from one MCAP file: decision rollouts (in
/// monotonic-clock order, `step_index` assigned by that order) and the outcomes to bind.
#[derive(Debug, Default)]
pub struct ImportedEvents {
    /// Decisions, one per decision-topic message, ordered by the monotonic clock.
    pub rollouts: Vec<Rollout>,
    /// Outcomes, one per outcome-topic message.
    pub outcomes: Vec<OutcomeEvent>,
}

impl MappingConfig {
    /// Parse and validate a mapping from TOML text. The one rule serde cannot express — a
    /// topic claimed by two roles — is checked here, so [`import_mcap`] can assume a
    /// conflict-free role map and never has to pick a winner.
    pub fn from_toml(text: &str) -> Result<Self, ImportError> {
        let config: MappingConfig = toml::from_str(text)?;
        // Build the role map eagerly to surface a double-claimed topic now, at load, not
        // silently at classification time.
        config.topic_roles()?;
        Ok(config)
    }

    /// Build the `topic -> role` lookup, rejecting any topic claimed more than once.
    pub(crate) fn topic_roles(&self) -> Result<BTreeMap<String, TopicRole>, ImportError> {
        let mut roles = BTreeMap::new();
        for d in &self.decisions {
            if roles.insert(d.topic.clone(), TopicRole::Decision).is_some() {
                return Err(ImportError::DuplicateTopic {
                    topic: d.topic.clone(),
                });
            }
        }
        for o in &self.outcomes {
            if roles
                .insert(o.topic.clone(), TopicRole::Outcome(o.outcome_kind))
                .is_some()
            {
                return Err(ImportError::DuplicateTopic {
                    topic: o.topic.clone(),
                });
            }
        }
        Ok(roles)
    }
}

/// Decode `mcap_bytes` into typed rollouts and outcomes under `mapping`.
///
/// Messages on decision topics become rollouts; messages on outcome topics become outcomes
/// of the mapped kind; messages on any other topic are ignored. The mapping's chosen MCAP
/// timestamp becomes the monotonic clock — `ts_wall_ns` is left advisory at 0 because
/// attribution decides on the skew-free monotonic delta, never the wall estimate. Rollouts
/// come back sorted by that clock with `step_index` assigned in that order, so a re-import
/// of the same file produces byte-identical numbering.
pub fn import_mcap(
    mcap_bytes: &[u8],
    mapping: &MappingConfig,
) -> Result<ImportedEvents, ImportError> {
    let roles = mapping.topic_roles()?;
    // Parse the identity UUIDs up front so a malformed id fails before any decoding work.
    let boot_id = mapping
        .boot_id
        .parse::<BootId>()
        .map_err(|e| ImportError::BadId {
            field: "boot_id",
            message: e.to_string(),
        })?;
    let episode_id = mapping
        .episode_id
        .parse::<EpisodeId>()
        .map_err(|e| ImportError::BadId {
            field: "episode_id",
            message: e.to_string(),
        })?;

    let robot = RobotIdentity::new(
        TenantId::new(mapping.tenant_id.clone()),
        RobotId::new(mapping.robot_id.clone()),
    );
    let policy_version = PolicyVersion::new(mapping.policy_version.clone());

    // Two-pass: outcomes are built as we see them, but decisions are collected as bare
    // stamps first so `step_index` can be assigned in clock order regardless of the order
    // the topics happen to interleave in the file.
    let mut decision_stamps: Vec<u64> = Vec::new();
    let mut outcomes: Vec<OutcomeEvent> = Vec::new();

    for message in mcap::MessageStream::new(mcap_bytes)? {
        let message = message?;
        let Some(role) = roles.get(message.channel.topic.as_str()) else {
            continue; // a sensor or other unmapped topic — ignored by design
        };
        let stamp = match mapping.clock.source {
            ClockSource::LogTime => message.log_time,
            ClockSource::PublishTime => message.publish_time,
        };
        match role {
            TopicRole::Decision => decision_stamps.push(stamp),
            TopicRole::Outcome(kind) => {
                let clock = MonoClock::new(boot_id, stamp, 0);
                outcomes.push(OutcomeEvent::new(
                    robot.clone(),
                    clock,
                    *kind,
                    BoundedBlob::empty(),
                ));
            }
        }
    }

    // Bare u64 stamps: equal keys are indistinguishable, so an unstable sort is still
    // deterministic. If a stamp ever carries a payload alongside it, this must become a
    // stable sort or the "re-import never renumbers" contract silently breaks.
    decision_stamps.sort_unstable();
    let mut rollouts = Vec::with_capacity(decision_stamps.len());
    for (index, stamp) in decision_stamps.into_iter().enumerate() {
        let step_index = u32::try_from(index)
            .expect("more than u32::MAX decisions in one file; step_index is u32 by schema");
        let clock = MonoClock::new(boot_id, stamp, 0);
        rollouts.push(Rollout::new(
            robot.clone(),
            episode_id,
            step_index,
            clock,
            policy_version.clone(),
            mapping.model_hash.clone(),
            mapping.embodiment.clone(),
            mapping.task_id.clone(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            0,
        ));
    }

    Ok(ImportedEvents { rollouts, outcomes })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAPPING: &str = r#"
        tenant_id = "acme"
        robot_id = "robot-01"
        boot_id = "00000000-0000-7000-8000-000000000001"
        episode_id = "00000000-0000-7000-8000-000000000002"
        embodiment = "six_dof_arm"
        policy_version = "pi-v1"
        task_id = "pick-place"

        [clock]
        source = "log_time"

        [[decisions]]
        topic = "/policy/action"

        [[outcomes]]
        topic = "/safety/estop"
        outcome_kind = "e_stop"
    "#;

    /// Delegate to the shared test writer with publish_time == log_time (no skew).
    fn write_mcap(messages: &[(&str, u64, &[u8])]) -> Vec<u8> {
        let with_publish: Vec<(&str, u64, u64, &[u8])> =
            messages.iter().map(|(t, l, d)| (*t, *l, *l, *d)).collect();
        crate::test_util::write_mcap(&with_publish)
    }

    #[test]
    fn imports_decisions_and_outcomes_from_real_mcap() {
        let bytes = write_mcap(&[
            ("/policy/action", 1_000, b"a0"),
            ("/sensor/depth", 1_500, b"ignored"), // unmapped topic -> dropped
            ("/policy/action", 2_000, b"a1"),
            ("/safety/estop", 3_000, b"ESTOP"),
        ]);
        let mapping = MappingConfig::from_toml(MAPPING).expect("valid mapping");
        let events = import_mcap(&bytes, &mapping).expect("import");

        assert_eq!(
            events.rollouts.len(),
            2,
            "two decision messages -> two rollouts"
        );
        assert_eq!(
            events.outcomes.len(),
            1,
            "one outcome message -> one outcome; the sensor topic is ignored"
        );

        // step_index assigned in clock order, mono_ns taken from log_time.
        assert_eq!(events.rollouts[0].step_index, 0);
        assert_eq!(events.rollouts[0].clock.mono_ns, 1_000);
        assert_eq!(events.rollouts[1].step_index, 1);
        assert_eq!(events.rollouts[1].clock.mono_ns, 2_000);

        // identity threaded from the mapping onto every rollout.
        assert_eq!(events.rollouts[0].embodiment, "six_dof_arm");
        assert_eq!(events.rollouts[0].task_id, "pick-place");

        // the outcome carries the mapped kind + its own clock.
        assert_eq!(events.outcomes[0].outcome_kind, OutcomeKind::EStop);
        assert_eq!(events.outcomes[0].clock.mono_ns, 3_000);
    }

    #[test]
    fn rejects_a_topic_claimed_by_two_roles() {
        let toml = r#"
            tenant_id = "t"
            robot_id = "r"
            boot_id = "00000000-0000-7000-8000-000000000001"
            episode_id = "00000000-0000-7000-8000-000000000002"
            embodiment = "arm"
            policy_version = "v"
            task_id = "task"
            [[decisions]]
            topic = "/shared"
            [[outcomes]]
            topic = "/shared"
            outcome_kind = "e_stop"
        "#;
        let err = MappingConfig::from_toml(toml).expect_err("topic in two roles is rejected");
        assert!(matches!(err, ImportError::DuplicateTopic { topic } if topic == "/shared"));
    }

    #[test]
    fn rejects_an_unknown_outcome_kind_at_parse() {
        let toml = r#"
            tenant_id = "t"
            robot_id = "r"
            boot_id = "00000000-0000-7000-8000-000000000001"
            episode_id = "00000000-0000-7000-8000-000000000002"
            embodiment = "arm"
            policy_version = "v"
            task_id = "task"
            [[outcomes]]
            topic = "/x"
            outcome_kind = "explosion"
        "#;
        // The closed taxonomy is the serde enum, so an unknown kind never reaches import.
        assert!(matches!(
            MappingConfig::from_toml(toml),
            Err(ImportError::Mapping(_))
        ));
    }

    #[test]
    fn rejects_a_malformed_boot_id() {
        let toml = r#"
            tenant_id = "t"
            robot_id = "r"
            boot_id = "not-a-uuid"
            episode_id = "00000000-0000-7000-8000-000000000002"
            embodiment = "arm"
            policy_version = "v"
            task_id = "task"
            [[decisions]]
            topic = "/d"
        "#;
        let mapping = MappingConfig::from_toml(toml).expect("parses; uuid checked at import");
        let err = import_mcap(&write_mcap(&[]), &mapping).expect_err("bad boot_id is rejected");
        assert!(matches!(err, ImportError::BadId { field, .. } if field == "boot_id"));
    }

    #[test]
    fn an_unknown_field_in_the_mapping_is_rejected() {
        let toml = r#"
            tenant_id = "t"
            robot_id = "r"
            boot_id = "00000000-0000-7000-8000-000000000001"
            episode_id = "00000000-0000-7000-8000-000000000002"
            embodiment = "arm"
            policy_version = "v"
            task_id = "task"
            taks_id = "typo"
        "#;
        assert!(matches!(
            MappingConfig::from_toml(toml),
            Err(ImportError::Mapping(_))
        ));
    }
}
