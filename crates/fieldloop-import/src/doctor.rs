//! `fieldloop doctor` core: check an MCAP file against a topic-mapping before attribution
//! is ever run, so a misconfigured mapping or a sensor on a bad time base is caught here
//! rather than surfacing as silently-missing or mis-timed bindings later.
//!
//! Two checks, both decided from the file's own contents (no external truth needed):
//! - **Mapping coverage** — which mapped topics are actually present, which declared
//!   topics are absent from the file (a typo or the wrong file), and which present topics
//!   are unmapped (ignored at import). A mapping that names a topic the file lacks is the
//!   single most common onboarding mistake.
//! - **Clock sanity** — MCAP stamps every message with both a `log_time` (recorder clock)
//!   and a `publish_time` (source-node clock). A sensor on a different time base (the
//!   classic case: a lidar on GPS time tens of seconds off the host) shows up as a large,
//!   consistent gap between the two. Topics whose worst per-message gap exceeds a
//!   threshold are flagged, because attribution compares decisions and outcomes on one
//!   monotonic timeline and a skewed source clock would bind to the wrong decision.

use std::collections::BTreeMap;

use crate::{ImportError, MappingConfig, TopicRole};

/// Default clock-skew threshold: 1 second. Sub-second gaps are ordinary transport/buffering
/// jitter; a steady gap of seconds means two different time bases, which is what the check
/// exists to catch. Callers can override it.
pub const DEFAULT_CLOCK_SKEW_THRESHOLD_NS: u64 = 1_000_000_000;

/// The role the mapping assigns a topic found in the file. `Unmapped` is not a problem:
/// undeclared topics are ignored by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosedRole {
    Decision,
    Outcome,
    Unmapped,
}

/// Per-topic findings from a [`diagnose`] pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicReport {
    /// The MCAP topic name.
    pub topic: String,
    /// The role the mapping assigns this topic.
    pub role: DiagnosedRole,
    /// How many messages the topic carries in this file.
    pub message_count: u64,
    /// Earliest `log_time` seen on the topic (nanoseconds).
    pub log_time_min_ns: u64,
    /// Latest `log_time` seen on the topic (nanoseconds).
    pub log_time_max_ns: u64,
    /// The worst per-message gap between `log_time` and `publish_time` on this topic —
    /// the clock-skew signal. A steady large value means the source clock diverges from
    /// the recorder clock.
    pub max_clock_skew_ns: u64,
}

/// The full result of a doctor pass over one MCAP file and mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnosis {
    /// One report per topic present in the file, sorted by topic name.
    pub topics: Vec<TopicReport>,
    /// Topics the mapping declares (as a decision or outcome) that do NOT appear in the
    /// file — usually a typo or the wrong file. Sorted.
    pub missing_topics: Vec<String>,
    /// Topics whose `max_clock_skew_ns` exceeded the threshold. Sorted.
    pub skewed_topics: Vec<String>,
    /// The threshold used for the skew check (nanoseconds), echoed for the report.
    pub clock_skew_threshold_ns: u64,
}

impl Diagnosis {
    /// Whether the doctor found anything that should block attribution: a declared topic
    /// missing from the file, or a topic on a divergent clock. Unmapped topics are not a
    /// problem (they are ignored by design), so they do not count here.
    #[must_use]
    pub fn has_problems(&self) -> bool {
        !self.missing_topics.is_empty() || !self.skewed_topics.is_empty()
    }
}

/// Diagnose `mcap_bytes` against `mapping` with an explicit clock-skew threshold (nanoseconds).
/// Pass [`DEFAULT_CLOCK_SKEW_THRESHOLD_NS`] to get the standard 1-second gate.
pub fn diagnose(
    mcap_bytes: &[u8],
    mapping: &MappingConfig,
    clock_skew_threshold_ns: u64,
) -> Result<Diagnosis, ImportError> {
    let roles = mapping.topic_roles()?;

    /// Running per-topic accumulator while streaming the file once.
    struct Acc {
        role: DiagnosedRole,
        count: u64,
        log_min: u64,
        log_max: u64,
        max_skew: u64,
    }
    let mut acc: BTreeMap<String, Acc> = BTreeMap::new();

    for message in mcap::MessageStream::new(mcap_bytes)? {
        let message = message?;
        let topic = message.channel.topic.as_str();
        let role = match roles.get(topic) {
            Some(TopicRole::Decision) => DiagnosedRole::Decision,
            Some(TopicRole::Outcome(_)) => DiagnosedRole::Outcome,
            None => DiagnosedRole::Unmapped,
        };
        // A zero publish_time is a writer that does not know the source stamp, not a source
        // clock 50+ years adrift — skip the skew signal rather than flag every topic.
        let skew = if message.publish_time == 0 {
            0
        } else {
            message.log_time.abs_diff(message.publish_time)
        };
        let entry = acc.entry(topic.to_owned()).or_insert_with(|| Acc {
            role,
            count: 0,
            log_min: u64::MAX,
            log_max: 0,
            max_skew: 0,
        });
        entry.count += 1;
        entry.log_min = entry.log_min.min(message.log_time);
        entry.log_max = entry.log_max.max(message.log_time);
        entry.max_skew = entry.max_skew.max(skew);
    }

    let topics: Vec<TopicReport> = acc
        .iter()
        .map(|(topic, a)| TopicReport {
            topic: topic.clone(),
            role: a.role,
            message_count: a.count,
            log_time_min_ns: a.log_min,
            log_time_max_ns: a.log_max,
            max_clock_skew_ns: a.max_skew,
        })
        .collect();

    // Declared topics that never appeared in the file.
    let missing_topics: Vec<String> = roles
        .keys()
        .filter(|topic| !acc.contains_key(*topic))
        .cloned()
        .collect();

    let skewed_topics: Vec<String> = topics
        .iter()
        .filter(|t| t.max_clock_skew_ns > clock_skew_threshold_ns)
        .map(|t| t.topic.clone())
        .collect();

    Ok(Diagnosis {
        topics,
        missing_topics,
        skewed_topics,
        clock_skew_threshold_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAPPING: &str = r#"
        tenant_id = "t"
        robot_id = "r"
        boot_id = "00000000-0000-7000-8000-000000000001"
        episode_id = "00000000-0000-7000-8000-000000000002"
        embodiment = "arm"
        policy_version = "v"
        task_id = "task"
        [[decisions]]
        topic = "/policy/action"
        [[outcomes]]
        topic = "/safety/estop"
        outcome_kind = "e_stop"
    "#;

    /// Delegate to the shared test writer, forwarding explicit (log_time, publish_time) so
    /// the clock-skew check is exercised against the real MCAP container.
    fn write_mcap(messages: &[(&str, u64, u64, &[u8])]) -> Vec<u8> {
        crate::test_util::write_mcap(messages)
    }

    #[test]
    fn clean_file_has_no_problems_and_classifies_topics() {
        // log_time == publish_time everywhere: no skew. A sensor topic is unmapped.
        let bytes = write_mcap(&[
            ("/policy/action", 1_000, 1_000, b"a"),
            ("/sensor/depth", 1_500, 1_500, b"s"),
            ("/safety/estop", 2_000, 2_000, b"e"),
        ]);
        let mapping = MappingConfig::from_toml(MAPPING).unwrap();
        let dx = diagnose(&bytes, &mapping, DEFAULT_CLOCK_SKEW_THRESHOLD_NS).unwrap();

        assert!(!dx.has_problems());
        assert!(dx.missing_topics.is_empty());
        assert!(dx.skewed_topics.is_empty());
        let roles: BTreeMap<_, _> = dx
            .topics
            .iter()
            .map(|t| (t.topic.as_str(), t.role))
            .collect();
        assert_eq!(roles["/policy/action"], DiagnosedRole::Decision);
        assert_eq!(roles["/safety/estop"], DiagnosedRole::Outcome);
        assert_eq!(roles["/sensor/depth"], DiagnosedRole::Unmapped);
    }

    #[test]
    fn a_topic_on_a_divergent_clock_is_flagged() {
        // /safety/estop publishes 5 s behind its log time — a sensor on another time
        // base. /policy/action is clean and must NOT be flagged.
        let five_s = 5_000_000_000u64;
        let bytes = write_mcap(&[
            ("/policy/action", 10_000_000_000, 10_000_000_000, b"a"),
            (
                "/safety/estop",
                12_000_000_000,
                12_000_000_000 - five_s,
                b"e",
            ),
        ]);
        let mapping = MappingConfig::from_toml(MAPPING).unwrap();
        let dx = diagnose(&bytes, &mapping, DEFAULT_CLOCK_SKEW_THRESHOLD_NS).unwrap();

        assert!(dx.has_problems());
        assert_eq!(dx.skewed_topics, vec!["/safety/estop".to_string()]);
    }

    #[test]
    fn zero_publish_time_is_not_skew() {
        // An epoch-scale log_time with publish_time 0 (writer does not know the source
        // stamp) must not appear in skewed_topics — the gap is not a clock-divergence
        // signal, just a missing timestamp.
        let bytes = write_mcap(&[("/policy/action", 1_700_000_000_000_000_000, 0, b"a")]);
        let mapping = MappingConfig::from_toml(MAPPING).unwrap();
        let dx = diagnose(&bytes, &mapping, DEFAULT_CLOCK_SKEW_THRESHOLD_NS).unwrap();

        assert!(
            dx.skewed_topics.is_empty(),
            "zero publish_time must not be treated as clock skew"
        );
    }

    #[test]
    fn a_declared_topic_absent_from_the_file_is_missing() {
        // The file has the decision topic but not the declared outcome topic.
        let bytes = write_mcap(&[("/policy/action", 1_000, 1_000, b"a")]);
        let mapping = MappingConfig::from_toml(MAPPING).unwrap();
        let dx = diagnose(&bytes, &mapping, DEFAULT_CLOCK_SKEW_THRESHOLD_NS).unwrap();

        assert!(dx.has_problems());
        assert_eq!(dx.missing_topics, vec!["/safety/estop".to_string()]);
    }
}
