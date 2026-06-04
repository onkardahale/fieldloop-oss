//! `OutcomeEvent` — the immutable raw outcome log.
//!
//! The raw observed or synthesized signal *before* it has been attributed to any
//! rollout. This is the immutable analytics record; it deliberately does **not**
//! carry attribution lifecycle state (whether/how it has been resolved) — that
//! mutable bookkeeping lives in a separate work queue, keeping this log
//! append-only. The JOIN consumes these plus [`crate::Rollout`]s and emits
//! [`crate::Feedback`].

use serde::{Deserialize, Serialize};

use crate::clock::{MonoClock, ServerAnchor};
use crate::ids::{OutcomeId, RolloutId};
use crate::payload::BoundedBlob;
use crate::pose::Se3Pose;
use crate::rollout::Trust;
use crate::tenant::RobotIdentity;

/// A causal hint naming an UPSTREAM station whose work plausibly produced this
/// outcome — the input the causal attribution tier follows.
///
/// Many real failures are observed *downstream* of the rollout that caused them: a
/// conveyor jam two stations later, a mis-pick that only shows up at an inspection
/// cell. Temporal and spatial attribution both fail here — the outcome is neither
/// close in time to the rollout (the part travelled) nor co-located with it (it
/// happened somewhere else) — so the only remaining signal is the *line topology*:
/// which upstream station feeds this one. An edge names that upstream station and an
/// optional bound on how long credit may reach back, so the causal tier can find the
/// rollouts at that station inside a lag window rather than guessing.
///
/// This is a HINT, not a proof: a causal binding produced from it is surfaced for a
/// curator to confirm, never treated as certain. The station label is a free string
/// the open SDK sets, so it is untrusted until validated like any open field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownstreamEdge {
    /// The upstream station / cell id that feeds the place this outcome was observed.
    /// Matched against a rollout's own `station_id` to find the candidate rollouts.
    pub upstream_station_id: String,
    /// Optional bound, in milliseconds, on how far back in (server-anchored) time
    /// credit may reach to that upstream station — the conveyor/handoff lag. `None`
    /// means "use the causal tier's default lag window"; a `Some` value pins a
    /// station-specific lag so a slow conveyor and a fast handoff are not forced to
    /// share one bound.
    pub max_lag_ms: Option<u64>,
}

impl DownstreamEdge {
    /// Construct a causal edge naming the upstream station, with no lag override
    /// (the causal tier's default lag applies).
    #[must_use]
    pub fn new(upstream_station_id: impl Into<String>) -> Self {
        Self {
            upstream_station_id: upstream_station_id.into(),
            max_lag_ms: None,
        }
    }

    /// Construct a causal edge with an explicit station-specific lag bound in
    /// milliseconds.
    #[must_use]
    pub fn with_lag(upstream_station_id: impl Into<String>, max_lag_ms: u64) -> Self {
        Self {
            upstream_station_id: upstream_station_id.into(),
            max_lag_ms: Some(max_lag_ms),
        }
    }
}

/// The kind of raw outcome observed.
///
/// A closed enum rather than a free string, so detectors and the attribution
/// cascade `match` exhaustively and a new outcome kind forces every consumer to
/// handle it. `Heartbeat` is special: it rides a separate cheap coverage path and
/// is a **no-drop class** on the robot ring, because its *absence* is what lets the
/// system synthesize "nothing went wrong here" — a dropped heartbeat would be
/// misread as a gap in coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    /// A human took over from the policy.
    TeleopTakeover,
    /// Emergency stop triggered.
    EStop,
    /// A collision was detected. A tight-window kind: it can only be attributed
    /// when the rollout and outcome share a boot and can be compared on the
    /// monotonic clock; otherwise it downgrades to ambiguous rather than binding on
    /// a skewed wall estimate.
    Collision,
    /// A failure observed downstream of the rollout (e.g. conveyor jam).
    DownstreamFailure,
    /// Liveness signal on the no-drop coverage path; its presence proves a window
    /// was covered, and its absence is what drives synthesized "no outcome here".
    Heartbeat,
}

/// `OutcomeEvent` — one immutable raw outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeEvent {
    /// UUIDv7; its embedded timestamp is advisory and must not drive attribution.
    pub id: OutcomeId,
    /// `(tenant_id, robot_id)` — never robot alone. A join where this tenant differs
    /// from the rollout's is a hard error, since binding across tenants would breach
    /// data isolation.
    #[serde(flatten)]
    pub robot: RobotIdentity,
    /// `(boot_id, mono_ns, ts_wall_ns)` — attribution scans by the skew-free
    /// monotonic clock within a boot, not by the wall estimate.
    #[serde(flatten)]
    pub clock: MonoClock,
    /// Server-set ingest time + boot anchor (trusted; set by the gateway). `None` on
    /// the pre-ingest record.
    pub server_anchor: Option<ServerAnchor>,
    /// What was observed.
    pub outcome_kind: OutcomeKind,
    /// Set iff the source threaded an explicit rollout id. `None` is the common
    /// implicit case — most outcomes arrive without a known target, so the
    /// attribution cascade must infer it; this is the core problem Fieldloop solves.
    /// Modeled as an [`Option`], not a nil-UUID sentinel, so "no explicit target"
    /// can never be mistaken for a real id.
    pub explicit_rollout_id: Option<RolloutId>,
    /// SERVER-derived trust. `None` until set at ingest.
    pub trust: Option<Trust>,
    /// Small inline payload, byte-bounded and re-validated at the gateway because it
    /// is an open-SDK field.
    pub payload: BoundedBlob,

    // ---- spatial attribution (cross-boot, server-anchored co-location) ------
    /// Where this outcome happened, as a rigid-body pose, when a detector can report
    /// one (e.g. a localized mobile base, a fixtured arm). `None` is the common case
    /// (no localization available) and disables the spatial tier for this outcome —
    /// the cascade falls through rather than inventing a location. `#[serde(default)]`
    /// so a row predating this field reads back as `None`, keeping the append
    /// backward-compatible with the frozen schema.
    #[serde(default)]
    pub pose: Option<Se3Pose>,
    /// The reference frame `pose` is expressed in (e.g. `"map"`, `"station_a_base"`).
    /// A pose is only comparable to another pose in the SAME frame, so the spatial
    /// tier compares an outcome to a rollout only when their `frame_id`s match —
    /// refusing to compare coordinates across different origins. Empty string means
    /// "no frame declared", which (like a `None` pose) keeps the outcome out of the
    /// spatial tier. `#[serde(default)]` empty for backward compatibility.
    #[serde(default)]
    pub frame_id: String,

    // ---- causal attribution (downstream-effect line topology) ---------------
    /// Upstream stations whose work plausibly produced this (downstream-observed)
    /// outcome — the input to the causal tier. Empty (the default) means no causal
    /// hint and disables the causal tier for this outcome. Several edges express a
    /// fan-in (this cell is fed by two upstream stations); the causal tier may then
    /// distribute credit across the rollouts it finds at each. `#[serde(default)]`
    /// empty for backward compatibility with rows written before this field.
    #[serde(default)]
    pub causal_parents: Vec<DownstreamEdge>,
}

impl OutcomeEvent {
    /// Minimal constructor for the pre-ingest record: mints a fresh
    /// [`OutcomeId`], leaves server-set fields `None`. Builder stub only — real
    /// emission lives in the SDK / detectors.
    #[must_use]
    pub fn new(
        robot: RobotIdentity,
        clock: MonoClock,
        outcome_kind: OutcomeKind,
        payload: BoundedBlob,
    ) -> Self {
        Self {
            id: OutcomeId::new(),
            robot,
            clock,
            server_anchor: None,
            outcome_kind,
            explicit_rollout_id: None,
            trust: None,
            payload,
            // No localization and no causal hint on the minimal pre-ingest record:
            // the spatial and causal tiers stay disabled until a detector/SDK fills
            // these in, so the common outcome rides explicit/temporal exactly as before.
            pose: None,
            frame_id: String::new(),
            causal_parents: Vec::new(),
        }
    }

    /// Attach a spatial pose in a named frame, returning the modified outcome.
    ///
    /// A builder so the common, location-less construction path's signature never
    /// changes and a pose is always a deliberate, visible addition — exactly the
    /// signal the spatial tier keys on.
    #[must_use]
    pub fn with_pose(mut self, pose: Se3Pose, frame_id: impl Into<String>) -> Self {
        self.pose = Some(pose);
        self.frame_id = frame_id.into();
        self
    }

    /// Attach causal-parent edges naming upstream stations, returning the modified
    /// outcome. A builder for the same reason as [`OutcomeEvent::with_pose`]: a causal
    /// hint is an explicit opt-in the causal tier keys on, never a silent default.
    #[must_use]
    pub fn with_causal_parents(mut self, parents: Vec<DownstreamEdge>) -> Self {
        self.causal_parents = parents;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::BootId;
    use crate::tenant::{RobotId, RobotIdentity, TenantId};

    fn sample() -> OutcomeEvent {
        OutcomeEvent::new(
            RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1")),
            MonoClock::new(BootId::new(), 1_000_000, 1_700_000_000_000_000_000),
            OutcomeKind::Collision,
            BoundedBlob::empty(),
        )
    }

    /// The minimal pre-ingest record disables the spatial and causal tiers: no pose,
    /// no frame, no causal parents — so the cascade rides explicit/temporal exactly as
    /// it did before these fields existed.
    #[test]
    fn minimal_outcome_has_no_spatial_or_causal_signal() {
        let o = sample();
        assert!(o.pose.is_none());
        assert!(o.frame_id.is_empty());
        assert!(o.causal_parents.is_empty());
    }

    /// An outcome JSON with NONE of the new fields — a row stored before this append —
    /// deserializes with `pose = None`, an empty `frame_id`, and no `causal_parents`.
    /// This is what makes the three fields a safe append to the frozen schema.
    #[test]
    fn old_json_without_new_fields_defaults_empty() {
        let mut json = serde_json::to_value(sample()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("pose");
        obj.remove("frame_id");
        obj.remove("causal_parents");
        let back: OutcomeEvent = serde_json::from_value(json).unwrap();
        assert!(back.pose.is_none());
        assert!(back.frame_id.is_empty());
        assert!(back.causal_parents.is_empty());
    }

    /// The pose, frame, and causal edges round-trip through JSON, including a
    /// station-specific lag override, so the attribution inputs survive storage.
    #[test]
    fn spatial_and_causal_fields_round_trip() {
        let o = sample()
            .with_pose(Se3Pose::at(1.0, 2.0, 0.0), "map")
            .with_causal_parents(vec![
                DownstreamEdge::new("inspection"),
                DownstreamEdge::with_lag("pick_cell", 4_000),
            ]);
        let back: OutcomeEvent = serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
        assert_eq!(back.pose, Some(Se3Pose::at(1.0, 2.0, 0.0)));
        assert_eq!(back.frame_id, "map");
        assert_eq!(back.causal_parents.len(), 2);
        assert_eq!(back.causal_parents[0].upstream_station_id, "inspection");
        assert_eq!(back.causal_parents[0].max_lag_ms, None);
        assert_eq!(back.causal_parents[1].max_lag_ms, Some(4_000));
    }
}
