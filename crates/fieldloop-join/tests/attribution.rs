//! Closed-loop tests for the attribution cascade.
//!
//! Every test is pure and deterministic: monotonic clocks are fixed integers (a
//! FakeClock-style fixed `mono_ns`), there is no database or I/O, and the same inputs
//! always produce the same bindings. The one intentional non-determinism is each
//! `Feedback.id`, which is a fresh wall-time UUIDv7 per row by design — and one test
//! asserts exactly that property.

use std::collections::BTreeMap;

use fieldloop_config::Config;
use fieldloop_join::{AttributeOptions, Heartbeat, SkipReason, attribute, attribute_report};
use fieldloop_types::PolicyVersion;
use fieldloop_types::payload::BoundedBlob;
use fieldloop_types::{
    BootId, DownstreamEdge, FeedbackValue, JoinMethod, MonoClock, OutcomeEvent, OutcomeKind,
    PayloadRef, RobotId, RobotIdentity, Rollout, Se3Pose, ServerAnchor, TenantId,
};

const MS: u64 = 1_000_000; // nanoseconds per millisecond.

/// The shared example config: ur5e with a clock-safe 250ms collision window, a 5s
/// loose takeover window, and a 2000ms default temporal window.
fn config() -> Config {
    Config::example()
}

fn robot(tenant: &str, id: &str) -> RobotIdentity {
    RobotIdentity::new(TenantId::new(tenant), RobotId::new(id))
}

/// A rollout at a fixed monotonic time on a given boot, tagged `ur5e`.
fn rollout_at(robot: &RobotIdentity, boot: BootId, mono_ns: u64) -> Rollout {
    let mut r = Rollout::new(
        robot.clone(),
        fieldloop_types::EpisodeId::new(),
        0,
        MonoClock::new(boot, mono_ns, mono_ns as i64),
        PolicyVersion::new("pick@v1.0.0+abc"),
        "sha256:beef".to_string(),
        "ur5e".to_string(),
        "bin_pick".to_string(),
        PayloadRef::none(),
        PayloadRef::none(),
        BoundedBlob::empty(),
        500,
    );
    r.tags = BTreeMap::new();
    r
}

/// An outcome of a given kind at a fixed monotonic time on a given boot.
fn outcome_at(
    robot: &RobotIdentity,
    boot: BootId,
    mono_ns: u64,
    kind: OutcomeKind,
) -> OutcomeEvent {
    OutcomeEvent::new(
        robot.clone(),
        MonoClock::new(boot, mono_ns, mono_ns as i64),
        kind,
        BoundedBlob::empty(),
    )
}

fn opts() -> AttributeOptions {
    AttributeOptions::default()
}

/// Explicit: an outcome carrying a rollout id that exists binds to it as `Explicit`
/// with confidence exactly 1.0 — the target is known, not inferred.
#[test]
fn explicit_binds_with_confidence_one() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let mut outcome = outcome_at(&r, boot, 1_500 * MS, OutcomeKind::DownstreamFailure);
    outcome.explicit_rollout_id = Some(rollout.id);

    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1);
    assert_eq!(fbs[0].join_method, JoinMethod::Explicit);
    assert_eq!(fbs[0].join_confidence, 1.0);
    assert_eq!(fbs[0].target.target_uuid(), rollout.id.as_uuid());
}

/// Temporal: an outcome shortly after one rollout (same boot, within the window)
/// binds to it as `Temporal` with confidence strictly between 0 and 1; an outcome
/// past the window does not bind.
#[test]
fn temporal_binds_within_window_and_not_past_it() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);

    // 1s after, inside the 5s takeover window.
    let inside = outcome_at(&r, boot, 2_000 * MS, OutcomeKind::TeleopTakeover);
    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[inside],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1);
    assert_eq!(fbs[0].join_method, JoinMethod::Temporal);
    assert!(
        fbs[0].join_confidence > 0.0 && fbs[0].join_confidence < 1.0,
        "temporal confidence must be strictly inside (0,1), got {}",
        fbs[0].join_confidence
    );

    // 6s after, past the 5s takeover window -> no binding.
    let outside = outcome_at(&r, boot, 7_000 * MS, OutcomeKind::TeleopTakeover);
    let report = attribute_report(&config(), &[rollout], &[outside], &[], &opts());
    assert!(report.feedbacks.is_empty());
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].reason, SkipReason::NoCandidateInWindow);
}

/// Two candidate rollouts before the outcome both inside the window now BOTH receive
/// distributed credit rather than the older one being dropped — this test changed with
/// the move from single-nearest to multi-step distributed credit. The nearer rollout
/// must still get strictly MORE credit (the recency-decay kernel), both rows must share
/// one contributing-set id, the weights must sum to ~1.0, and both rows carry the same
/// calibrated `join_confidence` (anchored on the nearest candidate) — which still
/// reflects the high recency of the nearer rollout. Co-candidates are no longer
/// silently discarded, so the row count is 2, not 1.
#[test]
fn distributed_credit_splits_across_both_in_window_candidates() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let older = rollout_at(&r, boot, 1_000 * MS);
    let newer = rollout_at(&r, boot, 1_800 * MS);
    let outcome = outcome_at(&r, boot, 2_000 * MS, OutcomeKind::TeleopTakeover);

    let fbs = attribute(
        &config(),
        &[older.clone(), newer.clone()],
        &[outcome],
        &[],
        &opts(),
    );
    // Both in-window candidates are credited (was 1 under single-nearest).
    assert_eq!(fbs.len(), 2, "both in-window rollouts must be credited");

    let newer_row = fbs
        .iter()
        .find(|f| f.target.target_uuid() == newer.id.as_uuid())
        .expect("the nearer rollout must have a row");
    let older_row = fbs
        .iter()
        .find(|f| f.target.target_uuid() == older.id.as_uuid())
        .expect("the farther rollout must have a row");

    // The nearer rollout (200ms delay) gets strictly more credit than the farther one
    // (1000ms delay) — the recency-decay kernel discriminates by recency.
    assert!(
        newer_row.credit_weight > older_row.credit_weight,
        "nearer rollout must get more credit: newer={} older={}",
        newer_row.credit_weight,
        older_row.credit_weight
    );

    // The weights are a normalized split: they sum to EXACTLY 1.0 (the last share is
    // assigned as the remainder, so f32 rounding cannot leak credit out of the sum).
    let sum = newer_row.credit_weight + older_row.credit_weight;
    assert!(
        sum == 1.0,
        "credit weights must sum to exactly 1.0, got {sum}"
    );

    // Both rows share ONE contributing-set id, marking them as co-contributors.
    assert!(newer_row.contributing_set_id.is_some());
    assert_eq!(
        newer_row.contributing_set_id, older_row.contributing_set_id,
        "co-contributors must share one contributing_set_id"
    );

    // Both carry the same calibrated confidence (anchored on the nearest candidate),
    // and it still reflects the high recency of the 200ms-delay nearer rollout.
    assert_eq!(newer_row.join_confidence, older_row.join_confidence);
    assert!(newer_row.join_confidence > 0.5);

    // A distributed-credit row is NEVER safety-eligible, even at full confidence:
    // the cause is ambiguous between co-contributors, so no single share may anchor
    // a safety veto.
    assert!(!newer_row.is_safety_eligible_confidence());
    assert!(!older_row.is_safety_eligible_confidence());
}

/// A cross-boot rollout is NEVER a distributed-credit contributor: only same-boot,
/// in-window rollouts are split across. Here a same-boot in-window rollout coexists
/// with a cross-boot one that is close by wall time; only the same-boot rollout is
/// credited, at full weight, and (being a lone contributor) it carries no
/// contributing-set id — the cross-boot refusal is preserved through distribution.
#[test]
fn cross_boot_rollout_is_excluded_from_credit_set() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let other_boot = BootId::new();
    let same_boot = rollout_at(&r, boot, 1_500 * MS);
    // Close to the outcome by wall time but on a different boot session.
    let cross_boot = rollout_at(&r, other_boot, 1_900 * MS);
    let outcome = outcome_at(&r, boot, 2_000 * MS, OutcomeKind::TeleopTakeover);

    let fbs = attribute(
        &config(),
        &[same_boot.clone(), cross_boot.clone()],
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1, "only the same-boot rollout may be credited");
    assert_eq!(fbs[0].target.target_uuid(), same_boot.id.as_uuid());
    assert_eq!(fbs[0].credit_weight, 1.0);
    assert!(fbs[0].contributing_set_id.is_none());
}

/// Conservation: with three in-window candidates, the DISTRIBUTED confidence
/// `Σ (credit_weight_i · join_confidence)` equals the single `C_total` carried on every
/// row — one outcome's confidence is partitioned across contributors, never duplicated.
#[test]
fn distributed_confidence_is_conserved_across_contributors() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let a = rollout_at(&r, boot, 1_000 * MS);
    let b = rollout_at(&r, boot, 1_400 * MS);
    let c = rollout_at(&r, boot, 1_900 * MS);
    let outcome = outcome_at(&r, boot, 2_000 * MS, OutcomeKind::TeleopTakeover);

    let fbs = attribute(
        &config(),
        &[a.clone(), b.clone(), c.clone()],
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 3);
    let c_total = fbs[0].join_confidence;
    // Every row carries the same C_total.
    assert!(fbs.iter().all(|f| f.join_confidence == c_total));
    // Distributed confidence sums back to exactly C_total (weights sum to 1.0).
    let distributed: f32 = fbs.iter().map(|f| f.credit_weight * c_total).sum();
    assert!(
        (distributed - c_total).abs() < 1e-5,
        "distributed confidence {distributed} must conserve C_total {c_total}"
    );
}

/// Back-compat: a SINGLE in-window rollout still produces exactly one row with full
/// credit (`credit_weight == 1.0`), no contributing-set id, and the same calibrated
/// confidence single-nearest attribution produced — so the multi-step generalization
/// reduces exactly to the prior behavior when there is no ambiguity. (Being a temporal,
/// i.e. inferred, binding it is not safety-eligible — that is gated on the method now,
/// not on the credit weight.)
#[test]
fn single_in_window_rollout_is_full_credit_and_back_compatible() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let outcome = outcome_at(&r, boot, 1_200 * MS, OutcomeKind::TeleopTakeover);

    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1, "a single candidate yields exactly one row");
    assert_eq!(fbs[0].target.target_uuid(), rollout.id.as_uuid());
    assert_eq!(
        fbs[0].credit_weight, 1.0,
        "a lone in-window rollout gets full, undistributed credit"
    );
    assert!(
        fbs[0].contributing_set_id.is_none(),
        "a single binding is not part of any contributing set"
    );
    assert_eq!(fbs[0].join_method, JoinMethod::Temporal);
}

/// A collision (its window `requires_monotonic_colocation`) whose only candidate is
/// from a DIFFERENT boot is marked ambiguous — never bound on a wall-clock guess.
#[test]
fn collision_cross_boot_is_ambiguous_not_bound() {
    let r = robot("acme", "r1");
    let rollout_boot = BootId::new();
    let outcome_boot = BootId::new(); // a different boot session.
    let rollout = rollout_at(&r, rollout_boot, 1_000 * MS);
    // Within the 250ms collision window by wall time, but a different boot.
    let outcome = outcome_at(&r, outcome_boot, 1_100 * MS, OutcomeKind::Collision);

    let report = attribute_report(&config(), &[rollout], &[outcome], &[], &opts());
    assert!(
        report.feedbacks.is_empty(),
        "a cross-boot collision must not bind on wall-clock"
    );
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(
        report.skipped[0].reason,
        SkipReason::AmbiguousMonotonicColocation
    );
}

/// A collision from the SAME boot within the window does bind — colocation is
/// satisfied, so the tight-timing kind is attributable.
#[test]
fn collision_same_boot_within_window_binds() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let outcome = outcome_at(&r, boot, 1_100 * MS, OutcomeKind::Collision); // 100ms, within 250ms.

    let fbs = attribute(&config(), &[rollout], &[outcome], &[], &opts());
    assert_eq!(fbs.len(), 1);
    assert_eq!(fbs[0].join_method, JoinMethod::Temporal);
    assert!(matches!(fbs[0].value, FeedbackValue::FailureClass { .. }));
}

/// Cross-tenant: an outcome whose tenant differs from the only matching-robot-id
/// rollout is never bound — it is a hard skip, not a low-confidence row.
#[test]
fn cross_tenant_is_never_bound() {
    let acme_robot = robot("acme", "shared-id");
    let evil_robot = robot("evil", "shared-id"); // same robot id, different tenant.
    let boot = BootId::new();
    let rollout = rollout_at(&acme_robot, boot, 1_000 * MS);
    // The outcome belongs to a different tenant though the robot id collides.
    let outcome = outcome_at(&evil_robot, boot, 1_200 * MS, OutcomeKind::TeleopTakeover);

    let report = attribute_report(&config(), &[rollout], &[outcome], &[], &opts());
    assert!(
        report.feedbacks.is_empty(),
        "no binding may cross a tenant boundary"
    );
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].reason, SkipReason::CrossTenant);
}

/// Synthetic-absence: a rollout whose window is fully heartbeat-covered and which
/// attracted no failure synthesizes a `SyntheticAbsence` success with no source
/// outcome and a confidence that reflects the (full) coverage. A coverage gap
/// suppresses synthesis.
#[test]
fn synthetic_absence_only_when_covered() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);

    // Default window is 2000ms (so [1000ms, 3000ms]). Heartbeat period 500ms, k=0.9
    // -> max gap 450ms. Place heartbeats every 400ms across the window with no gap
    // larger than 450ms.
    let period_ns = 500 * MS;
    let mut hbs = Vec::new();
    let mut t = 1_000 * MS;
    while t <= 3_000 * MS {
        hbs.push(Heartbeat {
            robot: r.clone(),
            clock: MonoClock::new(boot, t, t as i64),
        });
        t += 400 * MS;
    }

    let mut o = opts();
    o.heartbeat_period_ns = Some(period_ns);

    let fbs = attribute(&config(), std::slice::from_ref(&rollout), &[], &hbs, &o);
    assert_eq!(
        fbs.len(),
        1,
        "a covered, quiet rollout must synthesize one success"
    );
    let fb = &fbs[0];
    assert_eq!(fb.join_method, JoinMethod::SyntheticAbsence);
    assert!(
        fb.source_outcome_id.is_none(),
        "absence has no source outcome"
    );
    assert!(fb.delay_ms.is_none());
    assert_eq!(fb.value, FeedbackValue::Boolean { value: true });
    assert!(fb.join_confidence > 0.0 && fb.join_confidence <= 1.0);

    // Now punch a hole: drop the heartbeats in the middle so a >450ms gap appears.
    let holed: Vec<Heartbeat> = hbs
        .into_iter()
        .filter(|h| h.clock.mono_ns < 1_400 * MS || h.clock.mono_ns > 2_600 * MS)
        .collect();
    let fbs2 = attribute(&config(), &[rollout], &[], &holed, &o);
    assert!(
        fbs2.is_empty(),
        "an uncovered window must NOT synthesize an absence success"
    );
}

/// A rollout that DID attract a failure does not also get a synthesized "nothing went
/// wrong" success, even with full coverage — the two would contradict.
#[test]
fn synthetic_absence_suppressed_by_a_real_failure() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let failure = outcome_at(&r, boot, 1_200 * MS, OutcomeKind::TeleopTakeover);

    let period_ns = 500 * MS;
    let mut hbs = Vec::new();
    let mut t = 1_000 * MS;
    while t <= 3_000 * MS {
        hbs.push(Heartbeat {
            robot: r.clone(),
            clock: MonoClock::new(boot, t, t as i64),
        });
        t += 400 * MS;
    }

    let mut o = opts();
    o.heartbeat_period_ns = Some(period_ns);

    let fbs = attribute(&config(), &[rollout], &[failure], &hbs, &o);
    // Exactly one row: the temporal failure binding, NOT a contradicting absence.
    assert_eq!(fbs.len(), 1);
    assert_eq!(fbs[0].join_method, JoinMethod::Temporal);
}

/// Dedup: identical inputs produce an identical `dedup_key` (idempotent), a changed
/// window (different join version) produces a different key, and the `Feedback.id`
/// differs across runs (a fresh v7 each time).
#[test]
fn dedup_key_is_idempotent_but_id_is_fresh() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let outcome = outcome_at(&r, boot, 1_500 * MS, OutcomeKind::TeleopTakeover);

    let rollouts = [rollout];
    let outcomes = [outcome];
    let run1 = attribute(&config(), &rollouts, &outcomes, &[], &opts());
    let run2 = attribute(&config(), &rollouts, &outcomes, &[], &opts());
    assert_eq!(run1.len(), 1);
    assert_eq!(run2.len(), 1);

    // Idempotent: same inputs -> same dedup_key.
    assert_eq!(
        run1[0].dedup_key, run2[0].dedup_key,
        "identical inputs must produce the identical dedup_key"
    );
    // But the id is a fresh v7 each run, so latest-write can win on recency.
    assert_ne!(
        run1[0].id, run2[0].id,
        "Feedback.id must be freshly minted each run"
    );

    // A changed attribution version is a genuinely-different attribution -> new key.
    // (Uses `join-v3` because `join-v2` is now the default for distributed credit, so
    // a different version is needed to actually exercise a version change here.)
    let mut o2 = opts();
    o2.join_version = "join-v3".to_string();
    let run3 = attribute(&config(), &rollouts, &outcomes, &[], &o2);
    assert_ne!(
        run1[0].dedup_key, run3[0].dedup_key,
        "a changed join version must change the dedup_key"
    );
}

// ---------------------------------------------------------------------------
// Spatial tier — server-anchored cross-boot co-location.
// ---------------------------------------------------------------------------

/// A rollout at a fixed monotonic time on a given boot, with a server anchor and a
/// pose in the `"map"` frame — the rollout side of the spatial co-location test. The
/// `anchor_offset` reconciles this boot's monotonic origin to the shared server
/// timeline, so two different boots become comparable.
fn rollout_anchored(
    robot: &RobotIdentity,
    boot: BootId,
    mono_ns: u64,
    anchor_offset: i64,
    pose: Se3Pose,
) -> Rollout {
    let mut r = rollout_at(robot, boot, mono_ns);
    r.server_anchor = Some(ServerAnchor::new(1_700_000_000_000_000_000, anchor_offset));
    r = r.with_pose(pose, "map");
    r
}

/// An outcome with a server anchor and a pose, on a (typically different) boot.
fn outcome_anchored(
    robot: &RobotIdentity,
    boot: BootId,
    mono_ns: u64,
    anchor_offset: i64,
    kind: OutcomeKind,
    pose: Se3Pose,
) -> OutcomeEvent {
    let mut o = outcome_at(robot, boot, mono_ns, kind);
    o.server_anchor = Some(ServerAnchor::new(1_700_000_000_000_000_000, anchor_offset));
    o.with_pose(pose, "map")
}

/// SPATIAL: an outcome with NO same-boot temporal candidate but a co-located rollout
/// from a DIFFERENT boot binds as `Spatial`, with confidence strictly below what a
/// temporal bind would score (capped under the spatial ceiling), and is NOT
/// safety-eligible. This is the legitimate server-anchored cross-boot path the temporal
/// tier deliberately refuses.
#[test]
fn spatial_binds_colocated_cross_boot_rollout_below_temporal() {
    let r = robot("acme", "r1");
    let roll_boot = BootId::new();
    let out_boot = BootId::new(); // a DIFFERENT boot — temporal cannot compare.

    // Same place (0.1m apart, inside the 0.5m epsilon), aligned on the server timeline:
    // the rollout's anchored time equals the outcome's (offsets chosen so
    // offset + mono_ns lands at the same server instant), well inside the loose bound.
    let rollout = rollout_anchored(
        &r,
        roll_boot,
        1_000 * MS,
        9_000 * MS as i64,
        Se3Pose::at(1.0, 1.0, 0.0),
    );
    let outcome = outcome_anchored(
        &r,
        out_boot,
        2_000 * MS,
        8_000 * MS as i64,
        OutcomeKind::DownstreamFailure,
        Se3Pose::at(1.1, 1.0, 0.0),
    );

    let report = attribute_report(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(report.feedbacks.len(), 1, "spatial must recover a binding");
    let fb = &report.feedbacks[0];
    assert_eq!(fb.join_method, JoinMethod::Spatial);
    assert_eq!(fb.target.target_uuid(), rollout.id.as_uuid());
    // Sub-temporal: a coincident temporal bind scores ~1.0; spatial is capped at 0.6.
    assert!(
        fb.join_confidence > 0.0 && fb.join_confidence <= 0.6,
        "spatial confidence must be sub-temporal (<= ceiling 0.6), got {}",
        fb.join_confidence
    );
    // Inferred cross-boot bind: never safety-eligible.
    assert!(
        !fb.is_safety_eligible_confidence(),
        "a spatial bind must never be safety-eligible"
    );
}

/// SPATIAL refuses when the co-located rollout is too far in TIME: even at the same
/// place, a rollout outside the loose server-anchored time bound is not bound (it could
/// be a different shift entirely). Falls through to the terminal skip instead.
#[test]
fn spatial_refuses_when_outside_server_anchored_time_bound() {
    let r = robot("acme", "r1");
    let roll_boot = BootId::new();
    let out_boot = BootId::new();

    // Same place, but the anchored times are ~10 minutes apart (offsets differ by
    // 600s), beyond the 5-minute spatial bound.
    let rollout = rollout_anchored(&r, roll_boot, 1_000 * MS, 0, Se3Pose::at(1.0, 1.0, 0.0));
    let outcome = outcome_anchored(
        &r,
        out_boot,
        1_000 * MS,
        600_000 * MS as i64,
        OutcomeKind::DownstreamFailure,
        Se3Pose::at(1.0, 1.0, 0.0),
    );

    let report = attribute_report(&config(), &[rollout], &[outcome], &[], &opts());
    assert!(
        report.feedbacks.is_empty(),
        "a far-in-time co-location must not bind"
    );
    assert_eq!(
        report.skipped[0].reason,
        SkipReason::NoSpatialOrCausalCandidate
    );
}

/// A temporal-bindable outcome is still bound by TEMPORAL even when a co-located
/// cross-boot rollout exists: spatial/causal never steal a temporal binding.
#[test]
fn temporal_bindable_outcome_is_not_stolen_by_spatial() {
    let r = robot("acme", "r1");
    let boot = BootId::new();

    // Same-boot rollout 1s before the outcome (in the 5s takeover window) — temporal
    // owns this. Also give it a pose so spatial WOULD match if it were allowed to.
    let same_boot = rollout_anchored(&r, boot, 1_000 * MS, 0, Se3Pose::at(1.0, 1.0, 0.0));
    let outcome = outcome_anchored(
        &r,
        boot,
        2_000 * MS,
        0,
        OutcomeKind::TeleopTakeover,
        Se3Pose::at(1.0, 1.0, 0.0),
    );

    let fbs = attribute(&config(), &[same_boot], &[outcome], &[], &opts());
    assert_eq!(fbs.len(), 1);
    assert_eq!(
        fbs[0].join_method,
        JoinMethod::Temporal,
        "temporal must win over spatial when it can bind"
    );
}

// ---------------------------------------------------------------------------
// Causal tier — downstream line-topology hypothesis.
// ---------------------------------------------------------------------------

/// CAUSAL: an outcome carrying a `causal_parent` naming an upstream station, with no
/// temporal or spatial candidate, binds as `Causal` to that station's rollout — within
/// the lag window on the server-anchored clock — at the lowest inferred ceiling, and is
/// NOT safety-eligible (a causal bind is a hypothesis for curator confirmation).
#[test]
fn causal_binds_to_named_upstream_station() {
    let r = robot("acme", "r1");
    let roll_boot = BootId::new();
    let out_boot = BootId::new();

    // The upstream rollout ran at "pick_cell". No pose (spatial inapplicable), but a
    // server anchor so it sits on the shared timeline ~5s before the downstream effect.
    let mut upstream = rollout_at(&r, roll_boot, 1_000 * MS);
    upstream.server_anchor = Some(ServerAnchor::new(
        1_700_000_000_000_000_000,
        4_000 * MS as i64,
    ));
    upstream = upstream.at_station("pick_cell");

    // The downstream outcome 5s later (server-anchored), naming pick_cell as upstream.
    let mut outcome = outcome_at(&r, out_boot, 0, OutcomeKind::DownstreamFailure);
    outcome.server_anchor = Some(ServerAnchor::new(
        1_700_000_000_000_000_000,
        10_000 * MS as i64,
    ));
    let outcome = outcome.with_causal_parents(vec![DownstreamEdge::new("pick_cell")]);

    let report = attribute_report(
        &config(),
        std::slice::from_ref(&upstream),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(report.feedbacks.len(), 1, "causal must recover a binding");
    let fb = &report.feedbacks[0];
    assert_eq!(fb.join_method, JoinMethod::Causal);
    assert_eq!(fb.target.target_uuid(), upstream.id.as_uuid());
    // Lowest inferred ceiling: capped at 0.3, below the spatial ceiling.
    assert!(
        fb.join_confidence > 0.0 && fb.join_confidence <= 0.3,
        "causal confidence must be at the lowest inferred ceiling (<= 0.3), got {}",
        fb.join_confidence
    );
    assert!(
        !fb.is_safety_eligible_confidence(),
        "a causal bind must never be safety-eligible"
    );
}

/// CAUSAL distributes credit across a FAN-IN: an outcome whose cell is fed by two
/// upstream rollouts at the named station splits one hypothesis's confidence across
/// both (a shared contributing-set id, weights summing to ~1.0), and neither share is
/// safety-eligible — the cause is genuinely ambiguous between the upstream steps.
#[test]
fn causal_distributes_credit_across_fan_in() {
    let r = robot("acme", "r1");
    let out_boot = BootId::new();
    let base = 1_700_000_000_000_000_000;

    // Two upstream rollouts at the same station. Anchored time = offset + mono_ns, and
    // the effect is anchored at 10s; so up_a (offset 6s) lands 4s before the effect — the
    // NEARER upstream — and up_b (offset 4s) lands 6s before it — the farther.
    let mut up_a = rollout_at(&r, BootId::new(), 0);
    up_a.server_anchor = Some(ServerAnchor::new(base, 6_000 * MS as i64));
    let up_a = up_a.at_station("weld");
    let mut up_b = rollout_at(&r, BootId::new(), 0);
    up_b.server_anchor = Some(ServerAnchor::new(base, 4_000 * MS as i64));
    let up_b = up_b.at_station("weld");

    let mut outcome = outcome_at(&r, out_boot, 0, OutcomeKind::DownstreamFailure);
    outcome.server_anchor = Some(ServerAnchor::new(base, 10_000 * MS as i64));
    let outcome = outcome.with_causal_parents(vec![DownstreamEdge::new("weld")]);

    let fbs = attribute(
        &config(),
        &[up_a.clone(), up_b.clone()],
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 2, "a fan-in credits both upstream rollouts");
    assert!(fbs.iter().all(|f| f.join_method == JoinMethod::Causal));
    // Shared contributing-set id, weights conserved to ~1.0.
    let set = fbs[0].contributing_set_id;
    assert!(
        set.is_some(),
        "fan-in rows must share a contributing-set id"
    );
    assert_eq!(fbs[1].contributing_set_id, set);
    let total: f32 = fbs.iter().map(|f| f.credit_weight).sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "distributed causal credit must sum to ~1.0, got {total}"
    );
    // The nearer upstream (up_a, 4s back) gets strictly more credit than the farther
    // (up_b, 6s back) — the backward-lag decay kernel discriminates by recency.
    let w_a = fbs
        .iter()
        .find(|f| f.target.target_uuid() == up_a.id.as_uuid())
        .unwrap()
        .credit_weight;
    let w_b = fbs
        .iter()
        .find(|f| f.target.target_uuid() == up_b.id.as_uuid())
        .unwrap()
        .credit_weight;
    assert!(
        w_a > w_b,
        "the nearer upstream rollout must get more credit"
    );
    // Distributed -> never safety-eligible.
    assert!(fbs.iter().all(|f| !f.is_safety_eligible_confidence()));
}

/// CAUSAL refuses an upstream rollout outside the lag window: an edge's `max_lag_ms`
/// bounds how far back credit reaches, so a too-old upstream rollout is not bound.
#[test]
fn causal_refuses_upstream_outside_lag_window() {
    let r = robot("acme", "r1");
    let base = 1_700_000_000_000_000_000;

    // Upstream ran 20s before the effect, but the edge allows only a 5s lag.
    let mut upstream = rollout_at(&r, BootId::new(), 0);
    upstream.server_anchor = Some(ServerAnchor::new(base, 0));
    let upstream = upstream.at_station("pick_cell");

    let mut outcome = outcome_at(&r, BootId::new(), 0, OutcomeKind::DownstreamFailure);
    outcome.server_anchor = Some(ServerAnchor::new(base, 20_000 * MS as i64));
    let outcome = outcome.with_causal_parents(vec![DownstreamEdge::with_lag("pick_cell", 5_000)]);

    let report = attribute_report(&config(), &[upstream], &[outcome], &[], &opts());
    assert!(
        report.feedbacks.is_empty(),
        "an upstream rollout beyond the lag window must not bind"
    );
    assert_eq!(
        report.skipped[0].reason,
        SkipReason::NoSpatialOrCausalCandidate
    );
}

// ----------------------------------------------------------------------------
// Determinism, window-edge, and safety-gate semantics (work order: JOIN
// determinism + stable dedup hash + safety gate). Each test below pins a
// boundary the cascade's correctness lives on.
// ----------------------------------------------------------------------------

/// SPATIAL determinism: two co-located rollouts at the EXACTLY equal distance from the
/// outcome must bind the same one regardless of input slice order — the lower rollout
/// id wins. Without the tie-break the winner (and thus the dedup key) depended on
/// argument order, making attribution non-reproducible.
#[test]
fn spatial_tie_breaks_deterministically_by_rollout_id() {
    let r = robot("acme", "r1");
    let roll_boot = BootId::new();
    let out_boot = BootId::new();

    // Outcome at the origin; each rollout 0.1m away on a different axis, so the two
    // distances are computed from the identical f64 literal and are bit-for-bit equal.
    let a = rollout_anchored(
        &r,
        roll_boot,
        1_000 * MS,
        9_000 * MS as i64,
        Se3Pose::at(0.1, 0.0, 0.0),
    );
    let b = rollout_anchored(
        &r,
        roll_boot,
        1_000 * MS,
        9_000 * MS as i64,
        Se3Pose::at(0.0, 0.1, 0.0),
    );
    let outcome = outcome_anchored(
        &r,
        out_boot,
        2_000 * MS,
        8_000 * MS as i64,
        OutcomeKind::DownstreamFailure,
        Se3Pose::at(0.0, 0.0, 0.0),
    );
    let expected = a.id.min(b.id);

    for order in [vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]] {
        let report = attribute_report(
            &config(),
            &order,
            std::slice::from_ref(&outcome),
            &[],
            &opts(),
        );
        assert_eq!(
            report.feedbacks.len(),
            1,
            "the tie must still produce one binding"
        );
        assert_eq!(report.feedbacks[0].join_method, JoinMethod::Spatial);
        assert_eq!(
            report.feedbacks[0].target.target_uuid(),
            expected.as_uuid(),
            "an exact spatial tie must bind the lower rollout id regardless of input order"
        );
    }
}

/// ANTI-CIRCULARITY: a coincident (delta-0) temporal binding scores a raw 1.0 under the
/// identity calibrator, but it is still INFERRED, so it must never be safety-eligible.
/// Gating on the method rather than the float is exactly what enforces this.
#[test]
fn coincident_temporal_binding_is_not_safety_eligible() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let outcome = outcome_at(&r, boot, 1_000 * MS, OutcomeKind::TeleopTakeover); // delta 0

    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1);
    assert_eq!(fbs[0].join_method, JoinMethod::Temporal);
    assert_eq!(
        fbs[0].join_confidence, 1.0,
        "a coincident temporal binding scores raw 1.0 under identity calibration"
    );
    assert!(
        !fbs[0].is_safety_eligible_confidence(),
        "an inferred temporal binding is never safety-eligible, even at confidence 1.0"
    );
}

/// WINDOW EDGE is exclusive: an outcome exactly `window` after the rollout sits ON the
/// edge, where the recency score is 0.0. Admitting it would emit a confidence-0.0
/// binding nobody decided on; instead it is an auditable `NoCandidateInWindow`.
#[test]
fn outcome_exactly_at_window_edge_does_not_bind() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    // The takeover window is 5s; an outcome exactly 5s later is on the exclusive edge.
    let outcome = outcome_at(&r, boot, 6_000 * MS, OutcomeKind::TeleopTakeover);

    let report = attribute_report(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert!(
        report.feedbacks.is_empty(),
        "an at-edge outcome must not bind"
    );
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].reason, SkipReason::NoCandidateInWindow);
}

/// An outcome BEFORE every rollout has a negative forward delay and so is no temporal
/// candidate at all — it must skip, not bind to a later rollout.
#[test]
fn outcome_before_all_rollouts_does_not_bind() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 2_000 * MS);
    let outcome = outcome_at(&r, boot, 1_000 * MS, OutcomeKind::TeleopTakeover); // earlier

    let report = attribute_report(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert!(report.feedbacks.is_empty());
    assert_eq!(report.skipped[0].reason, SkipReason::NoCandidateInWindow);
}

/// An explicit rollout id that is NOT in the batch falls through to the temporal tier
/// rather than failing — the explicit hint is an optimization, not a requirement.
#[test]
fn explicit_id_absent_from_batch_falls_through_to_temporal() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let mut outcome = outcome_at(&r, boot, 1_500 * MS, OutcomeKind::TeleopTakeover);
    outcome.explicit_rollout_id = Some(fieldloop_types::RolloutId::new()); // not present

    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1);
    assert_eq!(
        fbs[0].join_method,
        JoinMethod::Temporal,
        "an explicit id absent from the batch must fall through to temporal"
    );
    assert_eq!(fbs[0].target.target_uuid(), rollout.id.as_uuid());
}

/// Distributed credit weights sum to EXACTLY 1.0 (not merely within a tolerance): the
/// last share is the remainder, so f32 rounding cannot leak credit out of the sum.
#[test]
fn distributed_credit_weights_sum_to_exactly_one() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let a = rollout_at(&r, boot, 1_000 * MS);
    let b = rollout_at(&r, boot, 1_400 * MS);
    let c = rollout_at(&r, boot, 1_900 * MS);
    let outcome = outcome_at(&r, boot, 2_000 * MS, OutcomeKind::TeleopTakeover);

    let fbs = attribute(&config(), &[a, b, c], &[outcome], &[], &opts());
    assert_eq!(fbs.len(), 3);
    let sum: f32 = fbs.iter().map(|f| f.credit_weight).sum();
    assert_eq!(
        sum, 1.0_f32,
        "distributed credit weights must sum to exactly 1.0, got {sum}"
    );
}

/// The dedup key carries the pinned `dk2:` scheme tag, so a key written by the stable
/// FNV-1a digest can never collide with a legacy (std-hash) key — the cutover is
/// explicit in the key itself.
#[test]
fn dedup_key_carries_dk2_scheme_tag() {
    let r = robot("acme", "r1");
    let boot = BootId::new();
    let rollout = rollout_at(&r, boot, 1_000 * MS);
    let mut outcome = outcome_at(&r, boot, 1_500 * MS, OutcomeKind::DownstreamFailure);
    outcome.explicit_rollout_id = Some(rollout.id);

    let fbs = attribute(
        &config(),
        std::slice::from_ref(&rollout),
        &[outcome],
        &[],
        &opts(),
    );
    assert_eq!(fbs.len(), 1);
    assert!(
        fbs[0].dedup_key.starts_with("dk2:"),
        "dedup key must carry the dk2 scheme tag, got {}",
        fbs[0].dedup_key
    );
}
