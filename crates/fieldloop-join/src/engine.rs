//! The attribution cascade: the core of the JOIN engine.
//!
//! Given a batch of [`Rollout`]s and [`OutcomeEvent`]s that have already landed for
//! one tenant, decide *which* deployed-policy decision each delayed, often-implicit
//! outcome belongs to, and *how confident* that binding is, emitting [`Feedback`]
//! rows. This module is pure, in-memory, deterministic logic: no database, no I/O,
//! no wall-clock reads except the fresh-id mint on the rows it produces.
//!
//! The engine ATTRIBUTES; it does not invent failure classes. A failure's *kind*
//! always comes from the originating [`OutcomeEvent`] (its [`OutcomeKind`]) or a
//! human label — never fabricated by the join. The join's only job is the binding
//! and the confidence.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use fieldloop_config::Config;
use fieldloop_types::{
    Feedback, FeedbackId, FeedbackTarget, FeedbackValue, JoinMethod, LabelKind, MonoClock,
    OutcomeEvent, OutcomeKind, RobotIdentity, Rollout, ServerAnchor, TenantId,
};
use uuid::Uuid;

use crate::calibrator::{Calibrator, IdentityCalibrator};
use crate::confidence::{synthetic_absence_raw_score, temporal_raw_score};

/// A liveness sample on the no-drop coverage path.
///
/// Carried as its own small struct (rather than a raw [`OutcomeEvent`]) so the
/// coverage path is explicit at the call site and a heartbeat can never be
/// accidentally fed into the failure-attribution cascade. Construct one directly, or
/// from a heartbeat-kind [`OutcomeEvent`] via [`Heartbeat::from_outcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heartbeat {
    /// `(tenant_id, robot_id)` — coverage is proven per robot, never robot alone.
    pub robot: RobotIdentity,
    /// The monotonic stamp of this sample. Coverage gaps are measured as `mono_ns`
    /// deltas within one boot, the skew-free authority — a heartbeat from a
    /// different boot cannot prove coverage of this boot's window.
    pub clock: MonoClock,
}

impl Heartbeat {
    /// Build a heartbeat from an [`OutcomeEvent`] of kind [`OutcomeKind::Heartbeat`],
    /// or `None` if the event is any other kind — so a real failure event can never
    /// be mistaken for a coverage sample.
    #[must_use]
    pub fn from_outcome(outcome: &OutcomeEvent) -> Option<Self> {
        if outcome.outcome_kind == OutcomeKind::Heartbeat {
            Some(Self {
                robot: outcome.robot.clone(),
                clock: outcome.clock,
            })
        } else {
            None
        }
    }
}

/// Knobs the caller controls per attribution run.
///
/// Defaults are provided so the common call is `AttributeOptions::default()`; the
/// fields exist so a caller can pin the attribution-logic version that lands on each
/// row (`join_version`) and declare the per-robot heartbeat period the coverage check
/// measures gaps against.
#[derive(Debug, Clone)]
pub struct AttributeOptions {
    /// Version label of the attribution logic, written onto every row and folded into
    /// its `dedup_key`. A change here is a genuinely-new attribution, so a re-run with
    /// a bumped version produces fresh, winning rows rather than being deduped away.
    pub join_version: String,
    /// The expected heartbeat inter-arrival period, in nanoseconds, used by the
    /// synthetic-absence coverage check. A window counts as covered only if no gap
    /// between consecutive heartbeats (and the window edges) exceeds
    /// `heartbeat_period_ns * calibration.heartbeat_coverage_k`. `None` disables
    /// synthetic-absence entirely (coverage cannot be proven without knowing the
    /// expected period).
    pub heartbeat_period_ns: Option<u64>,
    /// The metric name written on synthesized success rows. Real outcomes carry their
    /// metric from elsewhere; an absence has none of its own, so the caller names the
    /// success metric a covered-but-quiet window should score.
    pub absence_metric_name: String,
}

impl Default for AttributeOptions {
    fn default() -> Self {
        Self {
            // Bumped from `join-v1` to `join-v2` when temporal attribution moved from
            // single-nearest to multi-step DISTRIBUTED CREDIT: the binding decision
            // genuinely changed (co-candidates are now credited, not dropped), so a
            // re-run must mint fresh winning rows rather than dedup against the old
            // single-nearest rows whose `dedup_key` carried `join-v1`.
            join_version: "join-v2".to_string(),
            heartbeat_period_ns: None,
            absence_metric_name: "task_success".to_string(),
        }
    }
}

/// Why an outcome could not be bound — surfaced rather than silently dropped, so a
/// non-binding is auditable instead of invisible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The outcome's tenant differs from every candidate rollout's tenant. A
    /// cross-tenant binding is a data-isolation breach, so this is recorded as a hard
    /// skip, never emitted as a low-confidence row.
    CrossTenant,
    /// No rollout for the same `(tenant, robot, boot)` fell inside the attribution
    /// window before the outcome.
    NoCandidateInWindow,
    /// The embodiment of the only nearby rollout declared no window for this kind and
    /// no default applied — so there is no defined window to bind within.
    NoWindowConfigured,
    /// A tight-timing kind (one whose window `requires_monotonic_colocation`) had a
    /// candidate from a *different boot*, so the two cannot be compared on the
    /// skew-free monotonic clock. We refuse to guess on a drifting wall-clock and mark
    /// it ambiguous instead of binding.
    AmbiguousMonotonicColocation,
    /// A heartbeat sample — it rides the coverage path, never the failure cascade, so
    /// it produces no per-outcome binding of its own.
    HeartbeatSample,
    /// Temporal found no same-boot candidate AND neither inferred tier (spatial,
    /// causal) could recover a binding: the outcome carried no pose co-located with
    /// any rollout and no causal edge that resolved to an upstream rollout. The honest
    /// terminal "nothing bound it", recorded rather than silently dropped.
    NoSpatialOrCausalCandidate,
}

/// One outcome that did not produce a binding, paired with the reason — the
/// auditable record of every non-binding.
#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    /// The outcome that was not bound.
    pub outcome: OutcomeEvent,
    /// Why it was not bound.
    pub reason: SkipReason,
}

/// The full result of an attribution run: the bindings plus every non-binding and
/// its reason. The thin [`attribute`] returns only `feedbacks`; this richer report is
/// for callers (and tests) that must inspect ambiguity and cross-tenant skips.
#[derive(Debug, Clone, Default)]
pub struct AttributionReport {
    /// The emitted bindings, one per successfully-attributed outcome plus any
    /// synthesized absence-of-event successes.
    pub feedbacks: Vec<Feedback>,
    /// Outcomes that produced no binding, each with its reason.
    pub skipped: Vec<Skipped>,
}

/// Attribute a batch of outcomes to rollouts, returning only the emitted bindings.
///
/// The cascade, tried most-certain first, sets each row's `join_method` and
/// `join_confidence` accordingly:
///   1. **Explicit** — the outcome threaded a rollout id that exists in the batch:
///      bind to it, `Explicit`, confidence `1.0`.
///   2. **Temporal** — no explicit id: among same-`(tenant, robot, boot)` rollouts,
///      the latest whose `mono_ns <= outcome.mono_ns` and within the embodiment's
///      window; confidence is a recency ramp. A tight-timing kind whose candidate is
///      cross-boot is refused (ambiguous), never bound on a wall-clock guess.
///   3. **Synthetic-absence** — a rollout whose window is provably heartbeat-covered
///      and attracted no failure outcome: synthesize a success, `SyntheticAbsence`,
///      confidence scaling with coverage.
///
/// `Manual` is not produced here: a curator binding is an input the read layer ranks
/// above every automated method, and the ground truth this engine's confidence is
/// calibrated against. The engine never fabricates a manual row.
///
/// Uses the default [`IdentityCalibrator`]. See [`attribute_with`] to supply a fitted
/// calibrator, or [`attribute_report`] to also receive the non-bindings.
#[must_use]
pub fn attribute(
    config: &Config,
    rollouts: &[Rollout],
    outcomes: &[OutcomeEvent],
    heartbeats: &[Heartbeat],
    opts: &AttributeOptions,
) -> Vec<Feedback> {
    attribute_with(
        config,
        &IdentityCalibrator,
        rollouts,
        outcomes,
        heartbeats,
        opts,
    )
    .feedbacks
}

/// Like [`attribute`] but also returns the auditable non-bindings (cross-tenant,
/// out-of-window, ambiguous), using the default calibrator.
#[must_use]
pub fn attribute_report(
    config: &Config,
    rollouts: &[Rollout],
    outcomes: &[OutcomeEvent],
    heartbeats: &[Heartbeat],
    opts: &AttributeOptions,
) -> AttributionReport {
    attribute_with(
        config,
        &IdentityCalibrator,
        rollouts,
        outcomes,
        heartbeats,
        opts,
    )
}

/// The full cascade with an explicit [`Calibrator`].
///
/// Confidence is computed in two layers: the cascade emits a raw, mechanical score
/// (recency for temporal, coverage for absence), then the calibrator maps that score
/// onto the `[0, 1]` confidence per `(join_method, embodiment)`. The engine clamps the
/// calibrator's output defensively, so a buggy calibrator can never put an
/// out-of-range confidence on a row.
#[must_use]
pub fn attribute_with(
    config: &Config,
    calibrator: &dyn Calibrator,
    rollouts: &[Rollout],
    outcomes: &[OutcomeEvent],
    heartbeats: &[Heartbeat],
    opts: &AttributeOptions,
) -> AttributionReport {
    let calibration_version = calibrator
        .version_tag()
        .unwrap_or(&config.calibration.version)
        .to_string();

    let mut report = AttributionReport::default();

    // Track which rollouts attracted a failure-class outcome, so synthetic-absence
    // only fires for a rollout that was genuinely quiet (no real failure bound to it).
    let mut rollouts_with_failure: std::collections::HashSet<usize> =
        std::collections::HashSet::new();

    for outcome in outcomes {
        // A heartbeat is a coverage sample, never a failure to attribute.
        if outcome.outcome_kind == OutcomeKind::Heartbeat {
            report.skipped.push(Skipped {
                outcome: outcome.clone(),
                reason: SkipReason::HeartbeatSample,
            });
            continue;
        }

        match attribute_one(
            config,
            calibrator,
            &calibration_version,
            rollouts,
            outcome,
            opts,
        ) {
            AttributeOne::Bound { bindings } => {
                for (feedback, rollout_idx) in bindings {
                    // Every contributing rollout that absorbed a share of a failure
                    // outcome is marked, so synthetic-absence cannot later claim
                    // "nothing went wrong" for any rollout that took distributed
                    // credit for this failure — not just the nearest one.
                    if is_failure_kind(outcome.outcome_kind) {
                        rollouts_with_failure.insert(rollout_idx);
                    }
                    report.feedbacks.push(feedback);
                }
            }
            AttributeOne::Skip(reason) => report.skipped.push(Skipped {
                outcome: outcome.clone(),
                reason,
            }),
        }
    }

    // Synthetic-absence pass: for each rollout whose window is provably covered by
    // heartbeats and which attracted no failure, synthesize a success.
    if let Some(period_ns) = opts.heartbeat_period_ns {
        for (idx, rollout) in rollouts.iter().enumerate() {
            if rollouts_with_failure.contains(&idx) {
                continue;
            }
            if let Some(fb) = synthesize_absence(
                config,
                calibrator,
                &calibration_version,
                rollout,
                heartbeats,
                period_ns,
                opts,
            ) {
                report.feedbacks.push(fb);
            }
        }
    }

    report
}

/// Outcome of attributing a single non-heartbeat outcome.
enum AttributeOne {
    /// Bound to one or more rollouts. A single-candidate window yields one binding
    /// (back-compatible with single-nearest attribution); a multi-candidate window
    /// yields one binding per in-window rollout, each carrying its share of the
    /// distributed credit. Each entry pairs the produced row with the index of the
    /// rollout it credits, so the caller can mark every contributor as having
    /// absorbed the outcome.
    Bound { bindings: Vec<(Feedback, usize)> },
    /// Not bound, for this reason.
    Skip(SkipReason),
}

/// Attribute one non-heartbeat outcome through the explicit -> temporal cascade.
fn attribute_one(
    config: &Config,
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    rollouts: &[Rollout],
    outcome: &OutcomeEvent,
    opts: &AttributeOptions,
) -> AttributeOne {
    // ---- Tier 1: Explicit ---------------------------------------------------
    // The source threaded a rollout id. Bind to it iff it exists in this batch AND
    // shares the tenant — a cross-tenant explicit pointer is still a hard breach.
    if let Some(explicit_id) = outcome.explicit_rollout_id
        && let Some((idx, rollout)) = rollouts
            .iter()
            .enumerate()
            .find(|(_, r)| r.id == explicit_id)
    {
        if !rollout.robot.same_tenant(&outcome.robot) {
            return AttributeOne::Skip(SkipReason::CrossTenant);
        }
        let confidence = calibrate(calibrator, JoinMethod::Explicit, &rollout.embodiment, 1.0);
        let delay = mono_delay_ns(&rollout.clock, &outcome.clock);
        let feedback = build_feedback(
            outcome,
            rollout,
            JoinMethod::Explicit,
            confidence,
            Some(outcome.id),
            delay,
            outcome_value(outcome),
            outcome_label_kind(outcome),
            calibration_version,
            opts,
            // An explicit binding has a single known target, so it is always
            // full, undistributed credit — never part of a contributing set.
            1.0,
            None,
        );
        return AttributeOne::Bound {
            bindings: vec![(feedback, idx)],
        };
    }
    // Explicit id named a rollout not in this batch: fall through to temporal,
    // which may still find a same-boot neighbour. (A missing-target explicit
    // pointer is not itself a breach; it just is not bindable explicitly.)

    // ---- Tier 2: Temporal (the workhorse) -----------------------------------
    let temporal = temporal_attribute(
        config,
        calibrator,
        calibration_version,
        rollouts,
        outcome,
        opts,
    );
    match temporal {
        // Temporal bound it: done. Spatial/causal NEVER steal a temporal binding —
        // monotonic same-boot time is the strongest inferred evidence, so the later
        // tiers only run when temporal produced nothing.
        AttributeOne::Bound { bindings } => AttributeOne::Bound { bindings },
        // A cross-tenant collision is a hard data-isolation breach, not a "no
        // candidate" — it is preserved verbatim and never re-attempted by a weaker
        // tier, since binding across tenants is forbidden regardless of pose or topology.
        AttributeOne::Skip(SkipReason::CrossTenant) => AttributeOne::Skip(SkipReason::CrossTenant),
        // Temporal produced no binding (no same-boot in-window candidate, or a
        // tight-timing kind whose only candidate was cross-boot and so ambiguous on the
        // monotonic clock). THIS is exactly where the spatial and then causal tiers
        // fire. The ambiguous-colocation case is the legitimate cross-boot situation
        // spatial is designed for: it does not guess on a drifting wall-clock, it binds
        // on a pose match within a loose server-ANCHORED time bound. If both inferred
        // tiers also fail, the terminal skip is recorded.
        AttributeOne::Skip(temporal_reason) => {
            // ---- Tier 3: Spatial (server-anchored cross-boot co-location) -------
            if let Some(bound) = spatial_attribute(
                config,
                calibrator,
                calibration_version,
                rollouts,
                outcome,
                opts,
            ) {
                return bound;
            }
            // ---- Tier 4: Causal (downstream line topology) ----------------------
            if let Some(bound) = causal_attribute(
                config,
                calibrator,
                calibration_version,
                rollouts,
                outcome,
                opts,
            ) {
                return bound;
            }
            // Both inferred tiers declined. The terminal reason depends on whether the
            // outcome even CARRIED a spatial/causal signal: if it did and nothing
            // qualified, that is the new `NoSpatialOrCausalCandidate` diagnosis. If it
            // carried neither pose nor causal hint, the inferred tiers were never
            // applicable, so the original temporal reason is preserved unchanged —
            // keeping the cascade byte-compatible for the common, location-less outcome.
            let had_inferred_signal = outcome.pose.is_some() || !outcome.causal_parents.is_empty();
            if had_inferred_signal {
                AttributeOne::Skip(SkipReason::NoSpatialOrCausalCandidate)
            } else {
                AttributeOne::Skip(temporal_reason)
            }
        }
    }
}

/// The embodiment-scale co-location epsilon, in METERS: a rollout and an outcome whose
/// poses are within this translation distance are treated as the same place. `0.5 m`
/// is a deliberately loose default — wide enough to absorb localization noise and the
/// gap between where an arm's base is logged and where its end-effector contacted the
/// world, tight enough that two genuinely-different workcells (meters apart) never
/// collide. It is a join-side constant, not a config knob, kept named so the spatial
/// bound is one documented number rather than a buried literal; a fitted deployment
/// would tighten it per embodiment exactly as the temporal window is per embodiment.
const SPATIAL_EPSILON_M: f64 = 0.5;

/// The loose cross-boot time bound, in MILLISECONDS, on the SERVER-ANCHORED clock that
/// gates a spatial bind. Spatial is the legitimate cross-boot path temporal refuses, so
/// it still must not bind a rollout to an outcome that happened a different shift later
/// at the same spot. `5 min` is wide (spatial recovers bindings temporal could not, and
/// the localization, not the clock, is the primary evidence) but finite, on the trusted
/// server-ingest-anchored timeline rather than a drifting robot wall-clock. Named so the
/// bound is auditable.
const SPATIAL_TIME_BOUND_MS: i64 = 5 * 60 * 1_000;

/// The confidence CEILING for a spatial bind: a pose match is strictly weaker evidence
/// than monotonic same-boot time, so even a perfect (zero-distance) co-location is
/// capped here, well below what a coincident temporal bind would score. `0.6` keeps a
/// spatial row clearly sub-temporal (a 1 ms temporal bind scores ~1.0) while still
/// surfacing a strong co-location as a usable, curator-reviewable signal.
const SPATIAL_CONFIDENCE_CEILING: f64 = 0.6;

/// The default causal-lag window, in MILLISECONDS, on the SERVER-ANCHORED clock: how far
/// back in time the causal tier reaches from a downstream outcome to the upstream
/// station's rollouts when a [`fieldloop_types::DownstreamEdge`] does not pin its own
/// `max_lag_ms`. `30 s` covers a typical conveyor/handoff transit; a per-edge override
/// replaces it for a slow line. Named so the lag is one documented knob.
const CAUSAL_DEFAULT_LAG_MS: u64 = 30 * 1_000;

/// The confidence CEILING for a causal bind — the LOWEST of the inferred tiers. A causal
/// binding is a HYPOTHESIS about line topology surfaced for a curator to confirm, not a
/// measurement: it rests only on "an upstream station fed this cell" plus a lag window,
/// with neither time-coincidence (temporal) nor same-place evidence (spatial). `0.3`
/// keeps it clearly below the spatial ceiling so the cascade's confidence ordering
/// (explicit > temporal > spatial > causal) is monotone on the row.
const CAUSAL_CONFIDENCE_CEILING: f64 = 0.3;

/// One in-window temporal candidate: the rollout, its batch index, and its forward
/// monotonic delay from the rollout to the outcome (always `0 <= delta <= window`).
struct TemporalCandidate<'a> {
    idx: usize,
    rollout: &'a Rollout,
    delta: i128,
    window_ns: i128,
}

/// The temporal cascade with MULTI-STEP DISTRIBUTED CREDIT.
///
/// Instead of keeping only the single nearest same-`(tenant, robot, boot)` rollout, we
/// collect EVERY rollout whose forward monotonic delta to the outcome lies inside the
/// embodiment's window `[t_o - W, t_o]`, then split one calibrated confidence across
/// them by a recency-decay kernel so co-candidates are credited rather than dropped.
/// The nearer a rollout is, the larger its share. When exactly one rollout is in the
/// window the result is identical to single-nearest attribution: one full-credit row,
/// no contributing-set id, the same confidence — so this is a strict, back-compatible
/// generalization.
fn temporal_attribute(
    config: &Config,
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    rollouts: &[Rollout],
    outcome: &OutcomeEvent,
    opts: &AttributeOptions,
) -> AttributeOne {
    // Does ANY rollout share this outcome's robot but a different tenant? That would
    // be a cross-tenant collision of robot ids; refuse it as a hard skip rather than
    // ever binding across the isolation boundary.
    let mut saw_cross_tenant = false;

    // Every same-boot candidate at or before the outcome and within its window.
    let mut candidates: Vec<TemporalCandidate> = Vec::new();
    // Did a tight-timing kind have a candidate that was only cross-boot? Then the
    // honest answer is "ambiguous", not a wall-clock guess.
    let mut saw_cross_boot_for_colocated = false;

    // Resolve the window for this outcome's kind. We need an embodiment to look it up;
    // candidates carry it. We resolve per-candidate below, since a robot could in
    // principle run rollouts tagged with different embodiments in one batch.
    for (idx, rollout) in rollouts.iter().enumerate() {
        // Same robot id but different tenant -> cross-tenant hazard.
        if rollout.robot.robot_id == outcome.robot.robot_id
            && rollout.robot.tenant_id != outcome.robot.tenant_id
        {
            saw_cross_tenant = true;
            continue;
        }
        // Must be the same robot in the same tenant to even be a candidate.
        if rollout.robot != outcome.robot {
            continue;
        }

        let window = match resolve_window(config, &rollout.embodiment, outcome.outcome_kind) {
            Some(w) => w,
            None => continue, // No window for this (embodiment, kind): not a candidate.
        };
        let window_ns = i128::from(window.window_ms) * 1_000_000;

        // The monotonic delta is defined only within one boot. A `None` here means the
        // rollout and outcome are from different boots. Restricting the candidate set
        // to the SAME boot is exactly the cross-boot refusal: a colocated (tight-timing)
        // kind is never bound across boots on a drifting wall-clock, and a loose kind's
        // cross-boot binding belongs to a server-anchored path out of this batch's
        // scope — so a cross-boot rollout is never a distributed-credit contributor.
        match rollout.clock.mono_delta_ns(&outcome.clock) {
            Some(delta) => {
                // Forward in time (outcome at or after rollout) and within the window.
                if delta >= 0 && delta <= window_ns {
                    candidates.push(TemporalCandidate {
                        idx,
                        rollout,
                        delta,
                        window_ns,
                    });
                }
            }
            None => {
                // Cross-boot. For a tight-timing kind we must NOT fall back to a
                // wall-clock guess; record that the only-cross-boot situation arose so
                // we can report ambiguity if no same-boot candidate is found.
                if window.requires_monotonic_colocation {
                    saw_cross_boot_for_colocated = true;
                }
            }
        }
    }

    if !candidates.is_empty() {
        return AttributeOne::Bound {
            bindings: distribute_credit(
                calibrator,
                calibration_version,
                outcome,
                &candidates,
                opts,
            ),
        };
    }

    if saw_cross_tenant {
        return AttributeOne::Skip(SkipReason::CrossTenant);
    }
    if saw_cross_boot_for_colocated {
        return AttributeOne::Skip(SkipReason::AmbiguousMonotonicColocation);
    }
    AttributeOne::Skip(SkipReason::NoCandidateInWindow)
}

/// Split one outcome's calibrated in-window confidence across its in-window rollouts.
///
/// THE CREDIT MATH. For each candidate `i` with forward delay `Δ_i` (rollout → outcome)
/// inside window `W`:
///   * a recency-decay kernel `raw_i = exp(-Δ_i / τ)` weights nearer rollouts more —
///     `Δ_i = 0` gives `raw = 1`, a far rollout decays toward `0`;
///   * `τ = W / TAU_WINDOW_FRACTION` is embodiment-aware because `W` is the resolved
///     per-(embodiment, kind) window: a fast arm's short collision window yields a short
///     τ (credit falls off quickly), a long takeover window yields a long τ. With
///     `TAU_WINDOW_FRACTION = 3`, a candidate at the far edge `Δ = W` carries
///     `exp(-3) ≈ 0.05` of a coincident one's weight — small but non-zero, matching the
///     intent that an edge-of-window rollout is barely-credible rather than discarded;
///   * `credit_weight_i = raw_i / Σ raw` normalizes the shares to sum to `1.0`;
///   * `C_total = calibrate(Temporal, embodiment, temporal_raw_score(Δ_min, W))` is the
///     calibrated confidence that the cause is in-window AT ALL, anchored on the NEAREST
///     candidate's recency score — so the single-candidate case reduces EXACTLY to the
///     prior single-nearest confidence.
///
/// Each emitted row carries `join_confidence = C_total` and its own `credit_weight`.
/// The substantive conservation law is that the DISTRIBUTED confidence
/// `Σ (credit_weight_i · C_total) = C_total`: one outcome's confidence is partitioned,
/// never duplicated. A single candidate gets `credit_weight = 1.0`, no contributing-set
/// id, and `join_confidence = C_total` — byte-for-byte the old behavior. Two or more
/// share one freshly-minted `contributing_set_id`.
fn distribute_credit(
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    outcome: &OutcomeEvent,
    candidates: &[TemporalCandidate],
    opts: &AttributeOptions,
) -> Vec<(Feedback, usize)> {
    // The nearest candidate (smallest forward delay) anchors C_total, exactly the
    // single-nearest rollout the prior logic would have picked. Its window resolves
    // C_total's recency score; all same-(embodiment, kind) candidates share one window.
    let nearest = candidates
        .iter()
        .min_by_key(|c| c.delta)
        .expect("candidates is non-empty");
    let c_total = calibrate(
        calibrator,
        JoinMethod::Temporal,
        &nearest.rollout.embodiment,
        temporal_raw_score(nearest.delta, nearest.window_ns),
    );

    // Decay kernel raw_i = exp(-Δ_i / τ), τ = W / TAU_WINDOW_FRACTION. τ is derived
    // per-candidate from that candidate's own resolved window so a mixed-embodiment
    // batch still decays each contributor against its own window length.
    let raws: Vec<f64> = candidates
        .iter()
        .map(|c| {
            let tau = c.window_ns as f64 / TAU_WINDOW_FRACTION;
            if tau <= 0.0 {
                // Degenerate (zero-width) window: only a coincident rollout has weight.
                if c.delta == 0 { 1.0 } else { 0.0 }
            } else {
                (-(c.delta as f64) / tau).exp()
            }
        })
        .collect();

    let sum_raw: f64 = raws.iter().sum();

    // The contributing-set id groups the rows of a MULTI-contributor split. A lone
    // candidate is the single unambiguous binding and carries no group id (so it stays
    // safety-eligible and is byte-compatible with the old single-row output).
    let set_id = if candidates.len() > 1 {
        Some(Uuid::now_v7())
    } else {
        None
    };

    candidates
        .iter()
        .zip(raws.iter())
        .map(|(c, &raw)| {
            // Normalize to a share in (0, 1]. If every raw underflowed to 0 (only
            // possible with a degenerate window and no coincident rollout), fall back
            // to an equal split so credit is conserved rather than silently dropped.
            let credit_weight = if sum_raw > 0.0 {
                (raw / sum_raw) as f32
            } else {
                (1.0 / candidates.len() as f64) as f32
            };
            let delay_ms = Some((c.delta / 1_000_000) as i64);
            let feedback = build_feedback(
                outcome,
                c.rollout,
                JoinMethod::Temporal,
                c_total,
                Some(outcome.id),
                delay_ms,
                outcome_value(outcome),
                outcome_label_kind(outcome),
                calibration_version,
                opts,
                credit_weight,
                set_id,
            );
            (feedback, c.idx)
        })
        .collect()
}

/// Fraction of the window used to derive the decay time-constant `τ = W / this`.
///
/// `3` places the far window edge at `exp(-3) ≈ 0.05` of a coincident rollout's weight:
/// a candidate at the very edge is barely-credible but not discarded, which is the
/// whole point of distributing rather than dropping co-candidates. Kept as a named
/// constant so the credit curve is one documented knob rather than a buried literal.
const TAU_WINDOW_FRACTION: f64 = 3.0;

/// An event's time on the COMMON SERVER-ANCHORED timeline, in nanoseconds, or `None`
/// if the event carries no [`fieldloop_types::ServerAnchor`].
///
/// `(boot_id, mono_ns)` is comparable only within one boot, so two events from
/// different boots cannot be ordered on the raw monotonic clock — that is exactly the
/// cross-boot refusal the temporal tier enforces. The server anchor reconciles each
/// boot's monotonic origin to trusted server-ingest time: adding
/// `server_anchor_offset_ns` to `mono_ns` projects this boot's monotonic reading onto
/// the shared server timeline, so two events from DIFFERENT boots become comparable.
/// This is the trusted clock (set by the gateway, never robot-supplied), which is why
/// the spatial and causal tiers may legitimately compare cross-boot times on it where
/// temporal may not. Returns `None` when no anchor was stamped — without it there is no
/// trusted cross-boot time and the tier declines rather than guessing on a wall-clock.
fn server_anchored_ns(clock: &MonoClock, anchor: Option<&ServerAnchor>) -> Option<i128> {
    anchor.map(|a| i128::from(a.server_anchor_offset_ns) + i128::from(clock.mono_ns))
}

/// Tier 3 — SPATIAL: bind a cross-boot outcome to a co-located rollout.
///
/// Fires only after temporal produced no binding. The outcome must carry a pose in a
/// named frame; we then find every rollout that (a) shares the outcome's
/// `(tenant, robot)`, (b) carries a pose in the SAME `frame_id` (poses in different
/// frames name different origins and are never comparable), (c) is within
/// `SPATIAL_EPSILON_M` translation distance, AND (d) is within `SPATIAL_TIME_BOUND_MS`
/// on the SERVER-ANCHORED clock — the trusted cross-boot timeline, not the robot
/// wall-clock the temporal tier refuses to guess on. Among those, the NEAREST in space
/// is chosen; its confidence is the co-location closeness scaled under
/// `SPATIAL_CONFIDENCE_CEILING`, capped well below a temporal bind because a pose match
/// is weaker evidence than monotonic time. Returns `None` (cascade falls through to
/// causal) when the outcome has no pose/frame or no rollout qualifies — never a guess.
///
/// A spatial row is single-target full-credit but is NOT safety-eligible: it is an
/// inferred cross-boot bind, and the safety gate requires a temporal/explicit certainty
/// its sub-temporal confidence ceiling can never reach.
fn spatial_attribute(
    _config: &Config,
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    rollouts: &[Rollout],
    outcome: &OutcomeEvent,
    opts: &AttributeOptions,
) -> Option<AttributeOne> {
    // The outcome must localize itself, in a declared frame, or there is nothing to
    // co-locate against. An empty frame is "no frame declared" and disables the tier.
    let out_pose = outcome.pose?;
    if outcome.frame_id.is_empty() {
        return None;
    }
    // Spatial is the server-anchored cross-boot path: without a trusted anchor on the
    // outcome there is no trusted timeline to bound the bind on, so decline.
    let out_anchored = server_anchored_ns(&outcome.clock, outcome.server_anchor.as_ref())?;
    let time_bound_ns = i128::from(SPATIAL_TIME_BOUND_MS) * 1_000_000;

    // The nearest qualifying rollout: same tenant+robot, same frame, in-epsilon, and
    // within the loose server-anchored time bound. "Nearest in space" is the binding —
    // co-location closeness is the spatial tier's only confidence signal.
    let mut best: Option<(usize, &Rollout, f64)> = None;
    for (idx, rollout) in rollouts.iter().enumerate() {
        // Same robot in the same tenant only. A cross-tenant pose match is still a
        // breach and was already refused by the temporal tier; never bind it here.
        if rollout.robot != outcome.robot {
            continue;
        }
        // The rollout must offer a pose in the SAME frame, or its coordinates are not
        // comparable to the outcome's — refuse to compare across frames.
        let Some(roll_pose) = rollout.pose else {
            continue;
        };
        if rollout.frame_id != outcome.frame_id {
            continue;
        }
        let dist = roll_pose.translation_distance(&out_pose);
        if dist > SPATIAL_EPSILON_M {
            continue;
        }
        // Loose server-anchored time bound: the rollout must also carry an anchor (else
        // no trusted cross-boot time) and fall within the window around the outcome.
        let Some(roll_anchored) =
            server_anchored_ns(&rollout.clock, rollout.server_anchor.as_ref())
        else {
            continue;
        };
        if (out_anchored - roll_anchored).abs() > time_bound_ns {
            continue;
        }
        // Keep the spatially nearest candidate.
        match best {
            Some((_, _, best_dist)) if dist >= best_dist => {}
            _ => best = Some((idx, rollout, dist)),
        }
    }

    let (idx, rollout, dist) = best?;

    // Raw co-location score: 1.0 at zero distance, ramping to 0.0 at the epsilon edge —
    // the spatial analogue of the temporal recency ramp, so a closer match scores
    // higher. Then capped under the spatial ceiling so even a perfect match stays
    // strictly below a temporal bind.
    let closeness = (1.0 - dist / SPATIAL_EPSILON_M).clamp(0.0, 1.0);
    let raw = closeness * SPATIAL_CONFIDENCE_CEILING;
    let confidence = calibrate(calibrator, JoinMethod::Spatial, &rollout.embodiment, raw)
        .min(SPATIAL_CONFIDENCE_CEILING);

    // Delay on the server-anchored timeline (signed: the outcome may precede or follow
    // the rollout across boots), in milliseconds, for the row's provenance.
    let delay_ms = Some(
        ((out_anchored
            - server_anchored_ns(&rollout.clock, rollout.server_anchor.as_ref())
                .unwrap_or(out_anchored))
            / 1_000_000) as i64,
    );

    let feedback = build_feedback(
        outcome,
        rollout,
        JoinMethod::Spatial,
        confidence,
        Some(outcome.id),
        delay_ms,
        outcome_value(outcome),
        outcome_label_kind(outcome),
        calibration_version,
        opts,
        // A spatial bind credits exactly one nearest rollout; it is full, ungrouped
        // credit. It is nonetheless NOT safety-eligible because its confidence ceiling
        // can never reach 1.0 — the safety gate rejects it on confidence, as intended.
        1.0,
        None,
    );
    Some(AttributeOne::Bound {
        bindings: vec![(feedback, idx)],
    })
}

/// Tier 4 — CAUSAL: bind a downstream outcome to its upstream station's rollouts.
///
/// Fires only after explicit, temporal, AND spatial all produced nothing. The outcome
/// must carry `causal_parents` (downstream edges naming upstream stations). For each
/// edge we collect every rollout that (a) shares the outcome's `(tenant, robot)`,
/// (b) ran at the named `station_id`, and (c) precedes the outcome within the edge's
/// lag window on the SERVER-ANCHORED clock (`max_lag_ms`, else `CAUSAL_DEFAULT_LAG_MS`)
/// — upstream work must come BEFORE the downstream effect. Credit is DISTRIBUTED across
/// all qualifying rollouts (reusing the same conservation idea as the temporal tier:
/// the single calibrated confidence is split, never duplicated), since a fan-in cell
/// genuinely cannot tell which upstream step caused the jam.
///
/// Confidence is capped under `CAUSAL_CONFIDENCE_CEILING` — the lowest inferred tier —
/// because a causal bind is a HYPOTHESIS for curator confirmation, resting only on line
/// topology and a lag bound. Multi-target causal rows are inherently distributed and so
/// NEVER safety-eligible; even a lone causal row is below the safety confidence bar.
/// Returns `None` when there are no edges or none resolve to an in-lag upstream rollout.
fn causal_attribute(
    _config: &Config,
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    rollouts: &[Rollout],
    outcome: &OutcomeEvent,
    opts: &AttributeOptions,
) -> Option<AttributeOne> {
    if outcome.causal_parents.is_empty() {
        return None;
    }
    // The downstream effect's time on the trusted cross-boot timeline; without it there
    // is no anchored lag to measure upstream rollouts against, so decline.
    let out_anchored = server_anchored_ns(&outcome.clock, outcome.server_anchor.as_ref())?;

    // Collect qualifying upstream rollouts (dedup by index so two edges naming the same
    // station do not double-credit one rollout). Each entry is the rollout, its index,
    // and its backward lag (ns) from the outcome — used as the credit-decay key.
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut candidates: Vec<(usize, &Rollout, i128)> = Vec::new();

    for edge in &outcome.causal_parents {
        if edge.upstream_station_id.is_empty() {
            continue;
        }
        let lag_ms = edge.max_lag_ms.unwrap_or(CAUSAL_DEFAULT_LAG_MS);
        let lag_ns = i128::from(lag_ms) * 1_000_000;
        for (idx, rollout) in rollouts.iter().enumerate() {
            if seen.contains(&idx) {
                continue;
            }
            // Same robot in the same tenant only — a cross-tenant causal hint is still a
            // breach and is never bound across the isolation boundary.
            if rollout.robot != outcome.robot {
                continue;
            }
            // The rollout must have run at the edge's named upstream station.
            if rollout.station_id.is_empty() || rollout.station_id != edge.upstream_station_id {
                continue;
            }
            // Upstream work must precede the downstream effect, within the lag window,
            // on the trusted server-anchored clock (a rollout without an anchor cannot
            // be placed on that timeline, so it is not a candidate).
            let Some(roll_anchored) =
                server_anchored_ns(&rollout.clock, rollout.server_anchor.as_ref())
            else {
                continue;
            };
            let back = out_anchored - roll_anchored;
            if back < 0 || back > lag_ns {
                continue;
            }
            seen.insert(idx);
            candidates.push((idx, rollout, back));
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // One calibrated confidence for the whole causal hypothesis, anchored on the closest
    // (smallest backward lag) upstream rollout, capped under the causal ceiling. The lag
    // ramps closeness exactly like the temporal/spatial ramps: a freshly-upstream rollout
    // scores near the ceiling, one at the lag edge scores near zero.
    let min_back = candidates
        .iter()
        .map(|(_, _, b)| *b)
        .min()
        .expect("candidates non-empty");
    // Anchor the ramp on the largest lag window any winning edge allowed, so closeness is
    // measured against the actual reach used.
    let max_lag_ns = outcome
        .causal_parents
        .iter()
        .map(|e| i128::from(e.max_lag_ms.unwrap_or(CAUSAL_DEFAULT_LAG_MS)) * 1_000_000)
        .max()
        .unwrap_or(i128::from(CAUSAL_DEFAULT_LAG_MS) * 1_000_000);
    let closeness = if max_lag_ns > 0 {
        (1.0 - min_back as f64 / max_lag_ns as f64).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let raw = closeness * CAUSAL_CONFIDENCE_CEILING;
    // Embodiment of the nearest rollout buckets the calibration, mirroring temporal.
    let nearest_emb = candidates
        .iter()
        .min_by_key(|(_, _, b)| *b)
        .map(|(_, r, _)| r.embodiment.as_str())
        .unwrap_or("");
    let c_total =
        calibrate(calibrator, JoinMethod::Causal, nearest_emb, raw).min(CAUSAL_CONFIDENCE_CEILING);

    // Distribute credit across the upstream candidates by a backward-lag decay kernel,
    // exactly the temporal conservation law: Σ(credit_weight · C_total) = C_total — one
    // hypothesis's confidence is partitioned across the plausible upstream causes, never
    // duplicated. A fan-in with several upstream rollouts shares one contributing-set id.
    let tau = if max_lag_ns > 0 {
        max_lag_ns as f64 / TAU_WINDOW_FRACTION
    } else {
        1.0
    };
    let raws: Vec<f64> = candidates
        .iter()
        .map(|(_, _, back)| (-(*back as f64) / tau).exp())
        .collect();
    let sum_raw: f64 = raws.iter().sum();

    let set_id = if candidates.len() > 1 {
        Some(Uuid::now_v7())
    } else {
        None
    };

    let bindings = candidates
        .iter()
        .zip(raws.iter())
        .map(|((idx, rollout, back), &raw_w)| {
            let credit_weight = if sum_raw > 0.0 {
                (raw_w / sum_raw) as f32
            } else {
                (1.0 / candidates.len() as f64) as f32
            };
            let delay_ms = Some((*back / 1_000_000) as i64);
            let feedback = build_feedback(
                outcome,
                rollout,
                JoinMethod::Causal,
                c_total,
                Some(outcome.id),
                delay_ms,
                outcome_value(outcome),
                outcome_label_kind(outcome),
                calibration_version,
                opts,
                credit_weight,
                set_id,
            );
            (feedback, *idx)
        })
        .collect();

    Some(AttributeOne::Bound { bindings })
}

/// Synthesize a "nothing went wrong" success for a quiet, heartbeat-covered rollout.
///
/// Returns `Some` only if the rollout's window `[t0, t0+window]` is provably blanketed
/// by heartbeats with no gap exceeding `period_ns * heartbeat_coverage_k`. If coverage
/// cannot be proven (a gap too large, or no heartbeats), returns `None` — we never
/// synthesize an unsupported success.
fn synthesize_absence(
    config: &Config,
    calibrator: &dyn Calibrator,
    calibration_version: &str,
    rollout: &Rollout,
    heartbeats: &[Heartbeat],
    period_ns: u64,
    opts: &AttributeOptions,
) -> Option<Feedback> {
    // The window we want covered: the default temporal window after the rollout.
    let window_ns = i128::from(config.calibration.temporal_window_default_ms) * 1_000_000;
    if window_ns <= 0 {
        return None;
    }
    let start = i128::from(rollout.clock.mono_ns);
    let end = start + window_ns;

    // Same-boot, same-robot heartbeats inside (or bracketing) the window, sorted by
    // monotonic time. Cross-boot heartbeats cannot prove this boot's coverage.
    let mut stamps: Vec<i128> = heartbeats
        .iter()
        .filter(|h| h.robot == rollout.robot && h.clock.boot_id == rollout.clock.boot_id)
        .map(|h| i128::from(h.clock.mono_ns))
        .collect();
    if stamps.is_empty() {
        return None;
    }
    stamps.sort_unstable();

    // The largest gap allowed between consecutive coverage points.
    let max_gap = (period_ns as f64 * config.calibration.heartbeat_coverage_k) as i128;
    if max_gap <= 0 {
        return None;
    }

    let covered = covered_fraction(start, end, &stamps, max_gap);
    // Require the window to be fully covered before we claim "nothing happened". A
    // partial coverage is reported as a fraction below 1.0 and is NOT synthesized,
    // because an uncovered slice could hide an unreported failure.
    if covered < 1.0 {
        return None;
    }

    let raw = synthetic_absence_raw_score(covered);
    let confidence = calibrate(
        calibrator,
        JoinMethod::SyntheticAbsence,
        &rollout.embodiment,
        raw,
    );

    Some(Feedback {
        id: FeedbackId::new(),
        tenant_id: rollout.robot.tenant_id.clone(),
        target: target_for(rollout),
        label_kind: LabelKind::TerminalOutcome,
        metric_name: opts.absence_metric_name.clone(),
        // "Nothing went wrong" is a success.
        value: FeedbackValue::Boolean { value: true },
        join_method: JoinMethod::SyntheticAbsence,
        join_confidence: confidence as f32,
        join_version: opts.join_version.clone(),
        calibration_version: calibration_version.to_string(),
        // No real outcome event backs an absence.
        source_outcome_id: None,
        delay_ms: None,
        retracted: false,
        dedup_key: absence_dedup_key(rollout, opts, max_gap),
        outcome_ts_ns: rollout.clock.ts_wall_ns,
        // A synthesized absence credits exactly one quiet rollout; there is no set of
        // co-candidates to split, so it is always full, ungrouped credit.
        credit_weight: 1.0,
        contributing_set_id: None,
    })
}

/// The fraction of `[start, end]` that the sorted coverage `stamps` blanket.
///
/// Coverage is modeled as a sequence of *boundary points* — the window start, the
/// window end, and every heartbeat clamped into `[start, end]` — walked in order. A
/// span between two consecutive boundary points counts as covered iff the gap between
/// them is within `max_gap`; a wider gap is an uncovered hole (a heartbeat could have
/// been dropped there, hiding an unreported failure). The returned fraction is the
/// covered length over the total window length, in `[0, 1]`; it is `1.0` only when no
/// gap — leading edge, interior, or trailing edge — exceeds `max_gap`.
fn covered_fraction(start: i128, end: i128, stamps: &[i128], max_gap: i128) -> f64 {
    let total = end - start;
    if total <= 0 {
        return 0.0;
    }

    // Boundary points: the window edges plus each heartbeat clamped into the window,
    // de-duplicated and sorted. A heartbeat outside the window still pulls the nearest
    // edge gap closed once clamped, which is exactly the coverage we can claim.
    let mut points: Vec<i128> = Vec::with_capacity(stamps.len() + 2);
    points.push(start);
    points.push(end);
    for &s in stamps {
        points.push(s.clamp(start, end));
    }
    points.sort_unstable();
    points.dedup();

    // Walk consecutive boundary points; a span is covered when its width is within the
    // allowed gap.
    let mut covered: i128 = 0;
    for pair in points.windows(2) {
        let gap = pair[1] - pair[0];
        if gap <= max_gap {
            covered += gap;
        }
    }

    (covered as f64 / total as f64).clamp(0.0, 1.0)
}

/// Resolve the attribution window for `(embodiment, kind)`, falling back to the
/// calibration default window (with monotonic colocation NOT required) when the
/// embodiment declared none. The fallback only carries a `window_ms`; it never
/// upgrades a kind to require colocation, since that strictness is a per-embodiment
/// declaration.
fn resolve_window(
    config: &Config,
    embodiment: &str,
    kind: OutcomeKind,
) -> Option<fieldloop_config::AttributionWindow> {
    if let Some(w) = config.attribution_window(embodiment, kind) {
        return Some(w);
    }
    // Fallback: the calibration default. A collision (tight-timing) kind is NOT given
    // a loose default — if no embodiment window declared its colocation requirement,
    // we decline to invent one and return None for it.
    if kind == OutcomeKind::Collision {
        return None;
    }
    Some(fieldloop_config::AttributionWindow {
        window_ms: config.calibration.temporal_window_default_ms,
        requires_monotonic_colocation: false,
    })
}

/// Clamp a calibrator's output into `[0, 1]` so a buggy calibrator can never put an
/// out-of-range confidence onto a row.
fn calibrate(calibrator: &dyn Calibrator, method: JoinMethod, embodiment: &str, raw: f64) -> f64 {
    calibrator
        .calibrate(method, embodiment, raw)
        .clamp(0.0, 1.0)
}

/// The monotonic delay in milliseconds from rollout to outcome, or `None` across
/// boots (where no monotonic comparison is valid).
fn mono_delay_ns(rollout_clock: &MonoClock, outcome_clock: &MonoClock) -> Option<i64> {
    rollout_clock
        .mono_delta_ns(outcome_clock)
        .map(|d| (d / 1_000_000) as i64)
}

/// True for outcome kinds that represent a failure (and so should suppress a
/// synthetic "nothing went wrong" success for the rollout they bind to). A heartbeat
/// is not a failure; the rest are.
fn is_failure_kind(kind: OutcomeKind) -> bool {
    matches!(
        kind,
        OutcomeKind::TeleopTakeover
            | OutcomeKind::EStop
            | OutcomeKind::Collision
            | OutcomeKind::DownstreamFailure
    )
}

/// The feedback target for a rollout — its rollout-grain id.
fn target_for(rollout: &Rollout) -> FeedbackTarget {
    FeedbackTarget::Rollout(rollout.id)
}

/// The typed value carried from an outcome's kind. The engine does not invent a
/// failure class — it maps the OBSERVED kind onto the typed value. A takeover /
/// e-stop / downstream failure is a boolean "something went wrong" (false = the
/// policy did not cleanly succeed); a collision is the categorical hardware/contact
/// failure class drawn from the closed taxonomy.
fn outcome_value(outcome: &OutcomeEvent) -> FeedbackValue {
    match outcome.outcome_kind {
        OutcomeKind::Collision => FeedbackValue::FailureClass {
            class: fieldloop_types::FailureClass::Hardware,
        },
        OutcomeKind::TeleopTakeover | OutcomeKind::EStop | OutcomeKind::DownstreamFailure => {
            FeedbackValue::Boolean { value: false }
        }
        // A heartbeat never reaches here (filtered before attribution), but the match
        // must be total; treat it as a success placeholder.
        OutcomeKind::Heartbeat => FeedbackValue::Boolean { value: true },
    }
}

/// The supersession slot for an outcome kind: a takeover is an [`LabelKind::Intervention`];
/// the rest are the grain's [`LabelKind::TerminalOutcome`]. Choosing the slot from the
/// kind keeps an intervention and a terminal outcome on the same target coexisting
/// rather than overwriting one another.
fn outcome_label_kind(outcome: &OutcomeEvent) -> LabelKind {
    match outcome.outcome_kind {
        OutcomeKind::TeleopTakeover | OutcomeKind::EStop => LabelKind::Intervention,
        OutcomeKind::Collision | OutcomeKind::DownstreamFailure => LabelKind::TerminalOutcome,
        OutcomeKind::Heartbeat => LabelKind::TerminalOutcome,
    }
}

/// Build a bound feedback row. Centralizes the field plumbing so explicit and
/// temporal bindings produce identically-shaped rows differing only in method,
/// confidence, and dedup digest.
#[allow(clippy::too_many_arguments)]
fn build_feedback(
    outcome: &OutcomeEvent,
    rollout: &Rollout,
    method: JoinMethod,
    confidence: f64,
    source_outcome_id: Option<fieldloop_types::OutcomeId>,
    delay_ms: Option<i64>,
    value: FeedbackValue,
    label_kind: LabelKind,
    calibration_version: &str,
    opts: &AttributeOptions,
    credit_weight: f32,
    contributing_set_id: Option<Uuid>,
) -> Feedback {
    Feedback {
        id: FeedbackId::new(),
        tenant_id: outcome.robot.tenant_id.clone(),
        target: target_for(rollout),
        label_kind,
        metric_name: metric_name_for(outcome.outcome_kind),
        value,
        join_method: method,
        join_confidence: confidence as f32,
        join_version: opts.join_version.clone(),
        calibration_version: calibration_version.to_string(),
        source_outcome_id,
        delay_ms,
        retracted: false,
        dedup_key: bound_dedup_key(outcome, rollout, method, opts),
        outcome_ts_ns: outcome.clock.ts_wall_ns,
        credit_weight,
        contributing_set_id,
    }
}

/// A metric name derived from the outcome kind, so a row is self-describing.
fn metric_name_for(kind: OutcomeKind) -> String {
    match kind {
        OutcomeKind::TeleopTakeover => "teleop_takeover",
        OutcomeKind::EStop => "e_stop",
        OutcomeKind::Collision => "collision",
        OutcomeKind::DownstreamFailure => "downstream_failure",
        OutcomeKind::Heartbeat => "heartbeat",
    }
    .to_string()
}

/// The dedup key for a bound (explicit/temporal) row.
///
/// Derived from `(source_outcome_id, target, join_method, join_version, and a digest
/// of the attribution inputs)`. An identical re-run reuses the key (idempotent); a
/// changed input — a different rollout target, a bumped join version, a changed window
/// reflected in the delta — yields a different key, so a genuinely-newer binding can
/// win on latest-write. The `Feedback.id` is always a FRESH v7 and is NOT part of this
/// key, so a replay can be deduped while a re-attribution still mints a winning id.
fn bound_dedup_key(
    outcome: &OutcomeEvent,
    rollout: &Rollout,
    method: JoinMethod,
    opts: &AttributeOptions,
) -> String {
    // The attribution-input digest: the inputs that, if changed, mean a genuinely
    // different attribution decision. Serialized deterministically (a fixed field
    // order in a tuple) and hashed, so the digest is reproducible across runs.
    let mut hasher = DefaultHasher::new();
    outcome.id.as_uuid().hash(&mut hasher);
    rollout.id.as_uuid().hash(&mut hasher);
    rollout.clock.boot_id.as_uuid().hash(&mut hasher);
    rollout.clock.mono_ns.hash(&mut hasher);
    outcome.clock.mono_ns.hash(&mut hasher);
    opts.join_version.hash(&mut hasher);
    let digest = hasher.finish();

    format!(
        "out:{}|tgt:rollout:{}|method:{}|jv:{}|digest:{:016x}",
        outcome.id,
        rollout.id,
        method_tag(method),
        opts.join_version,
        digest
    )
}

/// The dedup key for a synthesized absence row. There is no source outcome, so the
/// key keys on the target rollout, the join version, and the coverage bound used
/// (`max_gap`) — a changed coverage bound is a different absence decision and so
/// yields a different key.
fn absence_dedup_key(rollout: &Rollout, opts: &AttributeOptions, max_gap: i128) -> String {
    let mut hasher = DefaultHasher::new();
    rollout.id.as_uuid().hash(&mut hasher);
    rollout.clock.boot_id.as_uuid().hash(&mut hasher);
    rollout.clock.mono_ns.hash(&mut hasher);
    opts.join_version.hash(&mut hasher);
    max_gap.hash(&mut hasher);
    let digest = hasher.finish();

    format!(
        "absence|tgt:rollout:{}|jv:{}|digest:{:016x}",
        rollout.id, opts.join_version, digest
    )
}

/// The stable short tag for a join method used inside a dedup key.
fn method_tag(method: JoinMethod) -> &'static str {
    match method {
        JoinMethod::Explicit => "explicit",
        JoinMethod::Temporal => "temporal",
        JoinMethod::Spatial => "spatial",
        JoinMethod::Causal => "causal",
        JoinMethod::Manual => "manual",
        JoinMethod::SyntheticAbsence => "synthetic_absence",
    }
}

/// Among a set of automated bindings and any curator-supplied manual bindings, the
/// read-precedence winner per `(target, label_kind)`: a manual binding outranks every
/// automated method; otherwise the newest row (largest `id`, a coarse v7 time sort)
/// wins. This is the read-side rule the storage layer applies; surfaced here as a
/// pure helper so a caller can resolve a slot without re-implementing the precedence.
///
/// `TenantId` is taken explicitly to keep the manual ground-truth source visible: the
/// calibration that fits the confidence curves consumes exactly these manual rows as
/// truth.
#[must_use]
pub fn winning_binding<'a>(_tenant: &TenantId, candidates: &'a [Feedback]) -> Option<&'a Feedback> {
    candidates.iter().filter(|f| !f.retracted).max_by(|a, b| {
        // Manual ranks above any automated method.
        let am = u8::from(a.join_method == JoinMethod::Manual);
        let bm = u8::from(b.join_method == JoinMethod::Manual);
        am.cmp(&bm)
            // Then by recency (the fresh v7 id is a coarse wall-time sort).
            .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
    })
}
