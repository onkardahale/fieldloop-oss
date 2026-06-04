//! Bidirectional lineage: the provenance spine that answers, in a single index
//! lookup each way:
//!   * `episodes_for_policy(policy)` — which training episodes a policy was built from,
//!   * `policies_for_episode(episode)` — which policies an episode ever fed.
//!
//! ## Why a store trait with maintained reverse indexes
//! The two questions are the same edges walked in opposite directions: episode -> the
//! dataset commit that pinned it -> the policy trained from that commit, and back. If
//! lineage were only stored one way, the reverse question would be a full scan — slow,
//! and easy to get subtly wrong as data grows. So [`LineageStore`] is a trait
//! (production backs it with a real index/DB; tests use [`InMemoryLineageStore`]) and
//! the in-memory impl maintains BOTH directions on every write, so each lookup is a
//! single-index read. Recording a commit and recording "policy P was trained from
//! commit C" are the only writes; the indexes are derived from them.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use fieldloop_types::{EpisodeId, PolicyVersion};

use crate::slice::DatasetCommit;

/// The provenance spine. Records dataset commits and policy-trained-from-commit edges,
/// and serves the two directional lookups without a scan.
///
/// A trait so the heavy/persistent implementation (a real index or DB) is a swap-in
/// behind the same surface the pure logic and tests use.
pub trait LineageStore {
    /// Record a dataset commit — its `episodes` become the lineage anchors a later
    /// trained-from edge connects a policy to.
    fn record_commit(&mut self, commit: &DatasetCommit);

    /// Record that `policy` was trained from the commit identified by `commit_id`.
    /// Idempotent: recording the same edge twice is a no-op, so a retried training run
    /// does not duplicate lineage.
    fn record_trained_from(&mut self, policy: &PolicyVersion, commit_id: &str);

    /// The episodes a policy was trained from: walk policy -> its commits -> each
    /// commit's episodes, de-duplicated and sorted. Empty if the policy is unknown.
    fn episodes_for_policy(&self, policy: &PolicyVersion) -> Vec<EpisodeId>;

    /// The policies an episode ever fed: walk episode -> the commits that pinned it ->
    /// the policies trained from those commits, de-duplicated and sorted. Empty if the
    /// episode is unknown. This is the reverse of [`LineageStore::episodes_for_policy`]
    /// and must agree with it.
    fn policies_for_episode(&self, episode: &EpisodeId) -> Vec<PolicyVersion>;
}

/// An in-memory [`LineageStore`] that maintains both directions of every edge on
/// write, so each lookup is a single-index read rather than a scan.
///
/// Intended for tests and single-process use; a persistent backend implements the same
/// trait for production. Determinism comes from `BTree*` collections, so lookups return
/// stably-sorted results.
#[derive(Debug, Default, Clone)]
pub struct InMemoryLineageStore {
    /// commit_id -> the episodes that commit pinned.
    commit_episodes: BTreeMap<String, BTreeSet<EpisodeId>>,
    /// policy -> the commit ids it was trained from. A `HashMap` because
    /// [`PolicyVersion`] is a content-hash string keyed by `Hash`, not `Ord`; lookups
    /// return sorted vecs so callers still get deterministic output.
    policy_commits: HashMap<PolicyVersion, HashSet<String>>,
    /// Reverse index, maintained on every `record_trained_from`: episode -> the
    /// policies that an edge connects to it through a commit. Kept alongside the
    /// forward maps so `policies_for_episode` is a direct read.
    episode_policies: BTreeMap<EpisodeId, HashSet<PolicyVersion>>,
}

impl InMemoryLineageStore {
    /// A fresh, empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Recompute the episode -> policies reverse edges for one policy/commit pair.
    ///
    /// Connecting a policy to a commit transitively connects it to that commit's
    /// episodes; this folds those reverse edges in. Called whenever either a commit's
    /// episodes or a policy's commit set changes, so the reverse index can never drift
    /// out of sync with the forward maps.
    fn link_reverse(&mut self, policy: &PolicyVersion, commit_id: &str) {
        if let Some(eps) = self.commit_episodes.get(commit_id) {
            for ep in eps {
                self.episode_policies
                    .entry(*ep)
                    .or_default()
                    .insert(policy.clone());
            }
        }
    }
}

impl LineageStore for InMemoryLineageStore {
    fn record_commit(&mut self, commit: &DatasetCommit) {
        let eps: BTreeSet<EpisodeId> = commit.resolved_manifest.episodes.iter().copied().collect();
        self.commit_episodes.insert(commit.commit_id.clone(), eps);

        // A commit can be recorded AFTER a policy already claimed it was trained from
        // this id (recording order is not guaranteed). Re-link any policies already
        // pointing at this commit so the reverse index picks up the now-known episodes.
        let linked: Vec<PolicyVersion> = self
            .policy_commits
            .iter()
            .filter(|(_, commits)| commits.contains(&commit.commit_id))
            .map(|(p, _)| p.clone())
            .collect();
        for policy in linked {
            self.link_reverse(&policy, &commit.commit_id);
        }
    }

    fn record_trained_from(&mut self, policy: &PolicyVersion, commit_id: &str) {
        self.policy_commits
            .entry(policy.clone())
            .or_default()
            .insert(commit_id.to_string());
        // Maintain the reverse direction immediately, so the back-lookup never scans.
        self.link_reverse(policy, commit_id);
    }

    fn episodes_for_policy(&self, policy: &PolicyVersion) -> Vec<EpisodeId> {
        let mut out: BTreeSet<EpisodeId> = BTreeSet::new();
        if let Some(commits) = self.policy_commits.get(policy) {
            for commit_id in commits {
                if let Some(eps) = self.commit_episodes.get(commit_id) {
                    out.extend(eps.iter().copied());
                }
            }
        }
        out.into_iter().collect()
    }

    fn policies_for_episode(&self, episode: &EpisodeId) -> Vec<PolicyVersion> {
        let mut out: Vec<PolicyVersion> = self
            .episode_policies
            .get(episode)
            .map(|ps| ps.iter().cloned().collect())
            .unwrap_or_default();
        // `PolicyVersion` is `Hash` not `Ord`; sort on its inner content-hash string
        // so the returned set is deterministic.
        out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        out
    }
}
