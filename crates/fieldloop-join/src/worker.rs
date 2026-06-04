//! The join WORKER: the process that reacts to the gateway's "rollouts landed" events
//! and drives real downstream JOIN work for the tenant they name.
//!
//! The pure engine in this crate decides bindings from in-memory inputs. This module is
//! the live edge around it: it subscribes to the gateway's NATS landed events and, for
//! each one, runs the canonical reward join against a real ClickHouse and reports how many
//! rollouts were attributed. That proves the gateway's event genuinely *triggers* the join
//! — the join runs because an event arrived, not because a timer fired.
//!
//! The whole module is behind the optional `worker` feature, so the default build (and the
//! offline closed-loop gate) never compiles it and never pulls `async-nats` or the live
//! ClickHouse client.
//!
//! It is split so a test can drive the core without a broker:
//!   * [`process_one_event`] is a plain async function: given a parsed event and a live
//!     ClickHouse client, it runs the join and returns a [`JoinSummary`]. A test calls it
//!     directly after a real NATS round-trip, so the assertion is on real ClickHouse state
//!     rather than a sleep-based poll.
//!   * [`run`] is the subscribe loop: it pulls landed events off a NATS subscription and
//!     calls `process_one_event` for each, forever.

use fieldloop_config::Config;
use fieldloop_store::clickhouse::queries::{
    RecentFeedbackParams, RecentWindowParams, RewardJoinParams, recent_feedback, recent_outcomes,
    recent_rollouts, reward_join,
};
use fieldloop_store::clickhouse::rows::{
    feedback_row, parse_feedback, parse_outcome, parse_rollout,
};
use fieldloop_store::live::ClickHouseClient;
use fieldloop_store::tenant::{ParamValue, TenantQuery};
use fieldloop_types::{Feedback, OutcomeEvent, Rollout, TenantId};
use serde::Deserialize;

use crate::calibrator::FittedCalibrator;
use crate::engine::{AttributeOptions, Heartbeat, attribute_with};

use futures::StreamExt;

/// The landed event as it arrives off NATS. This mirrors the gateway's published shape
/// (`{tenant, robot, rollout_ids}`) but is defined here, not imported from the gateway
/// crate, so the worker does not take a (live-feature-only) dependency on the gateway —
/// the JSON wire contract is the only coupling, which is exactly what a real event bus
/// gives you between two independently deployed services.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LandedEvent {
    /// The tenant whose rollouts landed — the scope the join runs for.
    pub tenant: String,
    /// The robot the batch came from (carried for routing/observability).
    pub robot: String,
    /// The ids of the rollouts that landed, as canonical UUID strings.
    pub rollout_ids: Vec<String>,
}

/// What one processed landed event produced: the tenant it ran for and how many rollout
/// rows the canonical reward join returned. The count is the headline proof that the event
/// drove real downstream work against ClickHouse — a test asserts the landed rollouts show
/// up here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinSummary {
    /// The tenant the join ran for (echoed from the event).
    pub tenant: String,
    /// The number of rollout rows the reward join returned for this tenant in the window.
    /// Each row is one rollout, with or without an attached outcome (the join is a LEFT
    /// JOIN), so this is the count of rollouts the join considered — at least the ones that
    /// just landed.
    pub rollouts_considered: usize,
}

/// Render a built [`TenantQuery`]'s bound values into its ClickHouse `{pN:Type}` markers.
///
/// The canonical join is built fail-closed with every value as a bound parameter (so a
/// value with SQL metacharacters can never reach the SQL text). The live ClickHouse client
/// here POSTs a single statement body, so the bound values are substituted into their
/// markers just before the POST. The tenant comes from a NATS message, so it is escaped by
/// doubling single quotes — the only metacharacter that matters for a single-quoted string
/// literal — keeping a crafted tenant from breaking out of the literal.
fn bind_params(q: &TenantQuery) -> String {
    let mut sql = q.sql.clone();
    for (i, p) in q.params.iter().enumerate() {
        let value = match p {
            ParamValue::Str(s) => format!("'{}'", s.replace('\'', "''")),
            ParamValue::I64(v) => v.to_string(),
            ParamValue::F64(v) => v.to_string(),
            ParamValue::Bool(b) => i64::from(*b).to_string(),
        };
        let marker = format!("{{p{i}:{}}}", p.clickhouse_type());
        sql = sql.replace(&marker, &value);
    }
    sql
}

/// Run the canonical reward join for one landed event against a live ClickHouse, returning
/// a [`JoinSummary`] with the count of rollouts the join returned for that tenant.
///
/// This is the testable core: a test can publish a landed event over real NATS, receive
/// it, parse it, and call this directly — asserting on real ClickHouse rows rather than
/// sleeping and polling. The reward join is tenant-scoped (the event's tenant is the only
/// scope), windowed wide so the just-landed rollouts are inside it, and read through the
/// `FeedbackByTargetId` view; the worker does not need the outcome to exist yet — a freshly
/// landed rollout with no outcome still comes back as a `has_outcome = 0` row, which is the
/// honest "attributed nothing yet" state.
///
/// # Errors
/// Returns an error string if building the join SQL fails (only on builder misuse) or the
/// ClickHouse read fails (a transport/SQL error, surfaced with the server's own message).
pub async fn process_one_event(
    event: &LandedEvent,
    clickhouse: &ClickHouseClient,
) -> Result<JoinSummary, String> {
    // Window the join very wide (the full plausible epoch range in fractional seconds) so a
    // rollout that landed at any wall-clock time is inside it. The worker's job is to prove
    // the event triggered a real read for the tenant, not to apply a business time filter;
    // a narrow window is a later API concern, so here the window is "everything".
    let params = RewardJoinParams {
        tenant: TenantId::new(event.tenant.clone()),
        metric_name: "reward".to_string(),
        label_kind: "terminal_outcome".to_string(),
        // Accept any calibrated confidence: the worker is counting coverage, not gating on
        // a confidence floor, so 0.0 lets every binding through while awaiting-outcome
        // rollouts still survive the LEFT JOIN regardless.
        min_confidence: 0.0,
        outcome_ts_from: 0.0,
        outcome_ts_to: 4_102_444_800.0, // 2100-01-01: an upper bound past any real outcome.
    };
    let query = reward_join(&params).map_err(|e| format!("build reward join: {e}"))?;
    let sql = bind_params(&query);
    let rows = clickhouse
        .query(&sql)
        .await
        .map_err(|e| format!("run reward join for tenant {}: {e}", event.tenant))?;
    Ok(JoinSummary {
        tenant: event.tenant.clone(),
        rollouts_considered: rows.len(),
    })
}

/// How many curator labels a `(join_method, embodiment)` bucket needs before the worker
/// fits a real calibration curve for it; below this it stays identity, so a thin bucket
/// never fabricates precision. Named here as the one knob the live write-side path uses,
/// kept conservative because a curve fit from a handful of labels is noise, not signal.
const MIN_CALIBRATION_LABELS: usize = 8;

/// What one write-side attribution pass produced for a tenant: the counts that prove the
/// REAL cascade (not a SQL read-join) ran on live data and wrote bindings back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeWriteSummary {
    /// The tenant the pass ran for.
    pub tenant: String,
    /// Rollouts read from the live window and fed to the cascade (the LEFT side).
    pub rollouts_read: usize,
    /// Outcomes read and attributed (heartbeat coverage samples included).
    pub outcomes_read: usize,
    /// Existing feedback rows read to FIT the calibrator (manual ground truth + prior
    /// automated samples).
    pub feedback_read: usize,
    /// `true` when the fit cleared `MIN_CALIBRATION_LABELS` for at least one bucket and a
    /// real curve was used; `false` means identity fallback (too few labels). Surfaced so
    /// an operator sees whether the fitted curve or the honest default ran.
    pub used_fitted_curve: bool,
    /// Feedback rows the cascade PRODUCED and inserted back — the rows the dashboard's
    /// `reward_join` read later surfaces. This is the headline proof the engine's tiers +
    /// the fitted calibrator ran on live ingested data and wrote real attribution.
    pub feedback_written: usize,
}

/// The PURE core of the write-side attribution path: given in-memory rollouts, outcomes,
/// and existing feedback already read for one tenant, fit a calibrator from the existing
/// feedback and run the REAL cascade to PRODUCE new feedback bindings.
///
/// This is exactly the engine work the live SQL read-join never did: it fits a
/// [`FittedCalibrator`] from the curator's manual ground truth + prior automated samples
/// (falling back to identity for any bucket with too few labels), then runs
/// [`attribute_with`] so the cascade's explicit/temporal/synthetic-absence tiers AND the
/// fitted confidence curve run on the live data. Heartbeat coverage samples are lifted
/// from the outcomes (a `Heartbeat`-kind outcome rides the coverage path, never the
/// failure cascade). The returned `(feedbacks, used_fitted_curve)` pairs the produced
/// bindings with whether a real curve (vs identity) backed them.
///
/// Pure and deterministic — no DB, no I/O — so it is fully testable from in-memory inputs.
/// The live wrapper [`attribute_landed_tenant`] is the only part that needs a ClickHouse:
/// it reads the inputs, calls this, and inserts the output.
#[must_use]
pub fn attribute_landed_batch(
    config: &Config,
    rollouts: &[Rollout],
    outcomes: &[OutcomeEvent],
    existing_feedback: &[Feedback],
    opts: &AttributeOptions,
    min_labels: usize,
) -> (Vec<Feedback>, bool) {
    // Heartbeats are not a separate stream on the wire; they arrive as Heartbeat-kind
    // outcomes. Lift them onto the coverage path so synthetic-absence can fire for a quiet
    // covered window, while the failure cascade never sees a heartbeat as a failure.
    let heartbeats: Vec<Heartbeat> = outcomes
        .iter()
        .filter_map(Heartbeat::from_outcome)
        .collect();

    // Fit the calibrator from the EXISTING feedback: the curator's manual rows are truth,
    // a prior automated binding is a labeled sample. A bucket below `min_labels` stays
    // identity (no invented precision), so on a cold tenant this is exactly the honest
    // identity default — which is why the cascade still runs, just uncalibrated.
    let calibrator = FittedCalibrator::fit(existing_feedback, rollouts, min_labels);
    // The fit earned a real curve iff its version tag is present AND at least one bucket
    // cleared the threshold; `FittedCalibrator` always tags itself, so probe a bucket that
    // the existing feedback would have populated to decide whether identity fallback ran.
    let used_fitted_curve = fitted_curve_is_active(&calibrator, existing_feedback, rollouts);

    let report = attribute_with(config, &calibrator, rollouts, outcomes, &heartbeats, opts);
    (report.feedbacks, used_fitted_curve)
}

/// Decide whether the fitted calibrator actually learned a curve (vs. falling back to
/// identity everywhere). A bucket is "active" when calibrating a mid-range raw score for
/// some `(method, embodiment)` present in the existing feedback bends the score away from
/// the identity pass-through — i.e. a learned curve moved it. This is a best-effort probe
/// for the summary's `used_fitted_curve` flag, not a correctness gate: the cascade runs
/// regardless of the answer.
fn fitted_curve_is_active(
    calibrator: &FittedCalibrator,
    existing_feedback: &[Feedback],
    rollouts: &[Rollout],
) -> bool {
    use crate::calibrator::Calibrator;
    use fieldloop_types::{FeedbackTarget, JoinMethod};
    // Resolve each automated feedback row's embodiment via its target rollout, then probe
    // that exact (method, embodiment) bucket — the only buckets the fit could have learned.
    let mut emb_by_rollout = std::collections::HashMap::new();
    for r in rollouts {
        emb_by_rollout.insert(r.id, r.embodiment.as_str());
    }
    for f in existing_feedback {
        if matches!(f.join_method, JoinMethod::Explicit | JoinMethod::Manual) {
            continue;
        }
        let emb = match f.target {
            FeedbackTarget::Rollout(id) => emb_by_rollout.get(&id).copied(),
            FeedbackTarget::Episode(_) => None,
        };
        if let Some(emb) = emb {
            // 0.5 is a neutral mid-range raw score; if a learned curve maps it to anything
            // other than itself, the bucket has a real (non-identity) curve.
            let probe = calibrator.calibrate(f.join_method, emb, 0.5);
            if (probe - 0.5).abs() > f64::EPSILON {
                return true;
            }
        }
    }
    false
}

/// Run the REAL write-side attribution cascade for one landed tenant against a live
/// ClickHouse: read the recent rollouts, outcomes, and existing feedback, fit a
/// calibrator, run the cascade, and INSERT the produced feedback rows back.
///
/// This closes the gap the read-side [`process_one_event`] leaves open. `process_one_event`
/// runs the cheap `reward_join` SELECT that only SURFACES feedback that already exists; it
/// never runs the engine, so the cascade tiers and the fitted calibrator never touch live
/// ingested data. This function is the WRITE side: it produces the feedback in the first
/// place, by running [`attribute_landed_batch`] (the pure cascade + fitted calibrator) over
/// rows read from the live store, then appending the result. The two are complementary —
/// this writes the bindings, `reward_join` later reads them for the dashboard.
///
/// The pure attribution (fit + cascade) is [`attribute_landed_batch`], fully unit-tested
/// without a DB. Everything this function adds is the live round-trip: the three SELECTs,
/// row parsing, and the Feedback INSERT — which genuinely needs a running ClickHouse to
/// EXECUTE and is therefore operator-run (exercised by the env-gated, `#[ignore]`d live
/// integration test, never in the offline gate).
///
/// # Errors
/// Returns an error string if building any read SQL fails (builder misuse), a read or the
/// insert fails against ClickHouse (transport/SQL error, with the server's message), or a
/// stored row cannot be parsed back into a typed value (a malformed row is named, never
/// silently dropped into the cascade).
pub async fn attribute_landed_tenant(
    event: &LandedEvent,
    config: &Config,
    clickhouse: &ClickHouseClient,
    window: &AttributeWindow,
    opts: &AttributeOptions,
) -> Result<AttributeWriteSummary, String> {
    let tenant = TenantId::new(event.tenant.clone());

    // ---- Read the three live streams for this tenant's recent window ----------
    let rollout_q = recent_rollouts(&RecentWindowParams {
        tenant: tenant.clone(),
        ts_wall_from_ns: window.ts_wall_from_ns,
        ts_wall_to_ns: window.ts_wall_to_ns,
    })
    .map_err(|e| format!("build recent_rollouts: {e}"))?;
    let rollout_rows = clickhouse
        .query(&bind_params(&rollout_q))
        .await
        .map_err(|e| format!("read rollouts for tenant {}: {e}", event.tenant))?;
    let rollouts: Vec<Rollout> = rollout_rows
        .iter()
        .map(parse_rollout)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("parse rollout row: {e}"))?;

    let outcome_q = recent_outcomes(&RecentWindowParams {
        tenant: tenant.clone(),
        ts_wall_from_ns: window.ts_wall_from_ns,
        ts_wall_to_ns: window.ts_wall_to_ns,
    })
    .map_err(|e| format!("build recent_outcomes: {e}"))?;
    let outcome_rows = clickhouse
        .query(&bind_params(&outcome_q))
        .await
        .map_err(|e| format!("read outcomes for tenant {}: {e}", event.tenant))?;
    let outcomes: Vec<OutcomeEvent> = outcome_rows
        .iter()
        .map(parse_outcome)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("parse outcome row: {e}"))?;

    let feedback_q = recent_feedback(&RecentFeedbackParams {
        tenant: tenant.clone(),
        outcome_ts_from: window.outcome_ts_from,
        outcome_ts_to: window.outcome_ts_to,
    })
    .map_err(|e| format!("build recent_feedback: {e}"))?;
    let feedback_rows = clickhouse
        .query(&bind_params(&feedback_q))
        .await
        .map_err(|e| format!("read feedback for tenant {}: {e}", event.tenant))?;
    let existing_feedback: Vec<Feedback> = feedback_rows
        .iter()
        .map(parse_feedback)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("parse feedback row: {e}"))?;

    // ---- Run the REAL cascade + fitted calibrator on the live data ------------
    let (produced, used_fitted_curve) = attribute_landed_batch(
        config,
        &rollouts,
        &outcomes,
        &existing_feedback,
        opts,
        MIN_CALIBRATION_LABELS,
    );

    // ---- Write the produced bindings back -------------------------------------
    // The produced feedback is appended to the same `Feedback` table `reward_join` reads,
    // via the existing serializer + insert path, so this write is the source of the rows
    // the dashboard later surfaces. An empty batch is a no-op (nothing to attribute yet),
    // not an error — a freshly landed window may have no bindable outcomes.
    if !produced.is_empty() {
        let rows: Vec<serde_json::Value> = produced.iter().map(feedback_row).collect();
        clickhouse
            .insert_json_each_row("Feedback", &rows)
            .await
            .map_err(|e| format!("insert produced feedback for tenant {}: {e}", event.tenant))?;
    }

    Ok(AttributeWriteSummary {
        tenant: event.tenant.clone(),
        rollouts_read: rollouts.len(),
        outcomes_read: outcomes.len(),
        feedback_read: existing_feedback.len(),
        used_fitted_curve,
        feedback_written: produced.len(),
    })
}

/// The bounded window one write-side attribution pass reads over.
///
/// Carries both the `ts_wall_ns` bounds the append-only Rollout/OutcomeEvent reads window
/// on and the `outcome_ts` (fractional-seconds) bounds the partitioned Feedback read
/// windows on — the two columns differ per stream (see the store's query params), so both
/// pairs ride here so one pass reads a consistent recent slice across all three.
#[derive(Debug, Clone)]
pub struct AttributeWindow {
    /// Inclusive lower `ts_wall_ns` bound for the Rollout/OutcomeEvent reads.
    pub ts_wall_from_ns: i64,
    /// Exclusive upper `ts_wall_ns` bound for the Rollout/OutcomeEvent reads.
    pub ts_wall_to_ns: i64,
    /// Inclusive lower `outcome_ts` (fractional seconds) bound for the Feedback read.
    pub outcome_ts_from: f64,
    /// Exclusive upper `outcome_ts` (fractional seconds) bound for the Feedback read.
    pub outcome_ts_to: f64,
}

/// The worker's subscribe loop: pull landed events off a NATS subscription and run the
/// join for each, forever.
///
/// Each message body is the gateway's JSON `LandedEvent`. A message that fails to parse is
/// logged and skipped (a malformed event must not kill the worker), and a join that fails
/// is logged and skipped (a transient ClickHouse error must not kill the worker either) —
/// the next landed event still gets processed. The loop ends only when the subscription
/// stream ends (the connection closed), at which point it returns so the caller can decide
/// whether to reconnect.
///
/// `subscription` is an `async_nats::Subscriber`, which is a stream of messages; this pulls
/// it with `StreamExt::next`. The caller subscribes to the landed wildcard subject (every
/// tenant) and hands the subscriber here.
pub async fn run(mut subscription: async_nats::Subscriber, clickhouse: ClickHouseClient) {
    while let Some(message) = subscription.next().await {
        let event: LandedEvent = match serde_json::from_slice(&message.payload) {
            Ok(ev) => ev,
            Err(e) => {
                eprintln!("join-worker: skipping unparseable landed event: {e}");
                continue;
            }
        };
        match process_one_event(&event, &clickhouse).await {
            Ok(summary) => {
                // Surface the attributed-rollout count so an operator sees the event drove a
                // real read (and how much it covered) rather than a silent no-op.
                println!(
                    "join-worker: tenant={} ran reward join, {} rollouts considered",
                    summary.tenant, summary.rollouts_considered
                );
            }
            Err(e) => eprintln!("join-worker: join failed for tenant {}: {e}", event.tenant),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_config::Config;
    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, FeedbackId, FeedbackTarget, FeedbackValue, JoinMethod,
        LabelKind, MonoClock, OutcomeEvent, OutcomeId, OutcomeKind, PayloadRef, PolicyVersion,
        RobotId, RobotIdentity, Rollout, RolloutId,
    };

    const MS: u64 = 1_000_000;

    fn robot(tenant: &str) -> RobotIdentity {
        RobotIdentity::new(TenantId::new(tenant), RobotId::new("r1"))
    }

    fn rollout_at(robot: &RobotIdentity, boot: BootId, mono_ns: u64, emb: &str) -> Rollout {
        Rollout::new(
            robot.clone(),
            EpisodeId::new(),
            0,
            MonoClock::new(boot, mono_ns, mono_ns as i64),
            PolicyVersion::new("pick@v1.0.0+abc"),
            "sha256:beef".to_string(),
            emb.to_string(),
            "bin_pick".to_string(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            500,
        )
    }

    fn manual_label(target: RolloutId, fail: bool) -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "task_success".into(),
            value: FeedbackValue::Boolean { value: !fail },
            join_method: JoinMethod::Manual,
            join_confidence: 1.0,
            join_version: "join-v2".into(),
            calibration_version: "manual".into(),
            source_outcome_id: None,
            delay_ms: None,
            retracted: false,
            dedup_key: format!("m:{target}"),
            outcome_ts_ns: 1,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    fn auto_temporal(target: RolloutId, conf: f32, fail: bool) -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "teleop_takeover".into(),
            value: FeedbackValue::Boolean { value: !fail },
            join_method: JoinMethod::Temporal,
            join_confidence: conf,
            join_version: "join-v2".into(),
            calibration_version: "fitted-v1".into(),
            source_outcome_id: Some(OutcomeId::new()),
            delay_ms: Some(1),
            retracted: false,
            dedup_key: format!("a:{target}:{conf}"),
            outcome_ts_ns: 1,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    /// The PURE core of the write-side path produces real bindings from in-memory inputs,
    /// with NO database: given a rollout and a takeover outcome shortly after it (same
    /// boot, inside the window), the cascade emits a `Temporal` binding to that rollout.
    /// This is exactly the engine work the live SQL read-join never did — the cascade
    /// runs on the landed data and writes a new attribution.
    #[test]
    fn pure_core_runs_the_cascade_and_produces_a_binding() {
        let r = robot("acme");
        let boot = BootId::new();
        let rollout = rollout_at(&r, boot, 1_000 * MS, "ur5e");
        let outcome = OutcomeEvent::new(
            r.clone(),
            MonoClock::new(boot, 2_000 * MS, (2_000 * MS) as i64),
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        );

        let (produced, used_fitted) = attribute_landed_batch(
            &Config::example(),
            std::slice::from_ref(&rollout),
            &[outcome],
            &[], // no existing feedback => identity fallback, but the cascade still runs.
            &AttributeOptions::default(),
            MIN_CALIBRATION_LABELS,
        );

        assert_eq!(produced.len(), 1, "the cascade must produce one binding");
        assert_eq!(produced[0].join_method, JoinMethod::Temporal);
        assert_eq!(produced[0].target.target_uuid(), rollout.id.as_uuid());
        // No labels were available, so the fit fell back to identity — honestly reported.
        assert!(
            !used_fitted,
            "no existing labels => identity fallback, not a fabricated curve"
        );
    }

    /// With enough curator labels, the PURE core fits a REAL calibration curve and runs
    /// the cascade THROUGH it: the produced temporal binding's confidence is the fitted
    /// value (curators contradicted low-raw bindings, so the curve bends a low raw score
    /// down), and `used_fitted_curve` is honestly true. This proves the FittedCalibrator —
    /// not just the identity default — runs on the landed data.
    #[test]
    fn pure_core_fits_a_curve_and_calibrates_the_binding() {
        let r = robot("acme");
        let boot = BootId::new();
        // The rollout the new outcome will bind to. Tag it `ur5e` so it shares the
        // calibration bucket with the labeled history below.
        let rollout = rollout_at(&r, boot, 1_000 * MS, "ur5e");
        let outcome = OutcomeEvent::new(
            r.clone(),
            // Far into the 5s takeover window so the raw recency score is LOW — the region
            // the curve was contradicted in, so a fitted curve should bend it down.
            MonoClock::new(boot, 4_900 * MS, (4_900 * MS) as i64),
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        );

        // Build labeled history: 8 ur5e temporal bindings curators CONTRADICTED at low raw
        // confidence (the automated row says "fail", the manual truth says "success"), so
        // the fitted curve learns low-raw -> low-confidence for this bucket.
        let mut history_rollouts = vec![rollout.clone()];
        let mut existing = Vec::new();
        for i in 0..8 {
            let hist = rollout_at(&r, boot, (10 + i) * MS, "ur5e");
            let id = hist.id;
            history_rollouts.push(hist);
            let conf = 0.10 + 0.01 * (i as f32);
            existing.push(auto_temporal(id, conf, true)); // automated says failure
            existing.push(manual_label(id, false)); // curator says success => contradicted
        }

        let (produced, used_fitted) = attribute_landed_batch(
            &Config::example(),
            &history_rollouts,
            &[outcome],
            &existing,
            &AttributeOptions::default(),
            MIN_CALIBRATION_LABELS,
        );

        // The cascade bound the new outcome to the target rollout (the labeled history
        // rollouts are far earlier, outside this outcome's window, so they do not compete).
        let bound: Vec<_> = produced
            .iter()
            .filter(|f| f.target.target_uuid() == rollout.id.as_uuid())
            .collect();
        assert_eq!(
            bound.len(),
            1,
            "exactly one binding to the target: {produced:?}"
        );
        let fb = bound[0];
        assert_eq!(fb.join_method, JoinMethod::Temporal);
        // A real curve was fit and used (not identity), and it stamped its fit tag on the
        // produced row — so the confidence is traceable to the fit.
        assert!(used_fitted, "enough labels => a real fitted curve ran");
        assert_eq!(fb.calibration_version, "fitted-v1");
        // The fitted curve bent the low raw recency score down toward the contradicted
        // accuracy: the produced confidence is low, not the bare recency score.
        assert!(
            fb.join_confidence < 0.5,
            "contradicted low-raw bucket calibrates DOWN, got {}",
            fb.join_confidence
        );
    }

    /// A `Heartbeat`-kind outcome is lifted onto the coverage path by the pure core, never
    /// the failure cascade — so it produces no per-outcome failure binding of its own. This
    /// guards the heartbeat lift the live wrapper relies on.
    #[test]
    fn pure_core_routes_heartbeats_to_coverage_not_failure() {
        let r = robot("acme");
        let boot = BootId::new();
        let rollout = rollout_at(&r, boot, 1_000 * MS, "ur5e");
        let hb = OutcomeEvent::new(
            r.clone(),
            MonoClock::new(boot, 1_500 * MS, (1_500 * MS) as i64),
            OutcomeKind::Heartbeat,
            BoundedBlob::empty(),
        );
        // No heartbeat_period_ns in default opts => synthetic-absence is disabled, so a
        // heartbeat alone produces nothing: it is a coverage sample, not a failure.
        let (produced, _) = attribute_landed_batch(
            &Config::example(),
            std::slice::from_ref(&rollout),
            &[hb],
            &[],
            &AttributeOptions::default(),
            MIN_CALIBRATION_LABELS,
        );
        assert!(
            produced.is_empty(),
            "a heartbeat must not produce a failure binding: {produced:?}"
        );
    }

    /// OPERATOR-RUN live round-trip: this is the one part of the write-side path that
    /// genuinely needs a running ClickHouse to EXECUTE — the three SELECTs, the row
    /// parsing, and the produced-feedback INSERT. It is `#[ignore]`d and a no-op unless
    /// `CLICKHOUSE_URL` is set, so neither `cargo test` nor `cargo test --features worker`
    /// requires a database; an operator runs it with `-- --ignored` against a real server.
    ///
    /// It proves the gap is closed end-to-end: insert a real Rollout + a takeover
    /// OutcomeEvent (no pre-existing Feedback — the read-join would surface NOTHING here),
    /// run [`attribute_landed_tenant`], and confirm the cascade PRODUCED and WROTE a
    /// `Temporal` binding that `reward_join` then reads back. The write is the thing the
    /// SQL read-join never did.
    #[tokio::test]
    #[ignore = "requires a live ClickHouse; set CLICKHOUSE_URL — operator-run"]
    async fn live_write_side_attribution_produces_and_persists_feedback() {
        let Ok(url) = std::env::var("CLICKHOUSE_URL") else {
            return;
        };
        let ch = ClickHouseClient::from_env(url);
        ch.apply_migrations().await.expect("migrate");

        // Unique tenant per run so repeated runs do not see each other's rows.
        let tenant = format!("w-{}", FeedbackId::new().as_uuid().simple());
        let r = RobotIdentity::new(TenantId::new(&tenant), RobotId::new("r1"));
        let boot = BootId::new();
        // A wall stamp inside a fixed window the read will span.
        let base_wall: i64 = 1_700_000_000_000_000_000;
        let mut rollout = rollout_at(&r, boot, 1_000 * MS, "ur5e");
        rollout.clock.ts_wall_ns = base_wall;
        let mut outcome = OutcomeEvent::new(
            r.clone(),
            MonoClock::new(boot, 2_000 * MS, base_wall + (1_000 * MS) as i64),
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        );
        outcome.clock.ts_wall_ns = base_wall + (1_000 * MS) as i64;

        ch.insert_json_each_row(
            "Rollout",
            &[fieldloop_store::clickhouse::rows::rollout_row(&rollout)],
        )
        .await
        .expect("insert rollout");
        ch.insert_json_each_row(
            "OutcomeEvent",
            &[fieldloop_store::clickhouse::rows::outcome_row(&outcome)],
        )
        .await
        .expect("insert outcome");

        // Run the REAL write-side cascade over the live data.
        let event = LandedEvent {
            tenant: tenant.clone(),
            robot: "r1".into(),
            rollout_ids: vec![rollout.id.to_string()],
        };
        let window = AttributeWindow {
            ts_wall_from_ns: base_wall - 1,
            ts_wall_to_ns: base_wall + 10 * (1_000 * MS) as i64,
            // The produced feedback's outcome_ts comes from the outcome's wall stamp
            // (~1.7e9 s); window the feedback read widely around it.
            outcome_ts_from: 1_700_000_000.0,
            outcome_ts_to: 1_700_000_100.0,
        };
        let summary = attribute_landed_tenant(
            &event,
            &Config::example(),
            &ch,
            &window,
            &AttributeOptions::default(),
        )
        .await
        .expect("write-side attribution");

        // The cascade read the landed rollout + outcome and PRODUCED at least one binding.
        assert_eq!(summary.rollouts_read, 1, "{summary:?}");
        assert_eq!(summary.outcomes_read, 1, "{summary:?}");
        assert!(
            summary.feedback_written >= 1,
            "the cascade must have produced and written feedback: {summary:?}"
        );

        // Force the view merge so the read sees the freshly written binding.
        ch.execute("OPTIMIZE TABLE FeedbackByTargetId FINAL")
            .await
            .ok();

        // reward_join now surfaces the binding the WRITE side produced — proving the
        // produced rows are the ones the dashboard read later sees. The metric written for
        // a takeover is `teleop_takeover` on the Intervention slot.
        let q = reward_join(&RewardJoinParams {
            tenant: TenantId::new(&tenant),
            metric_name: "teleop_takeover".into(),
            label_kind: "intervention".into(),
            min_confidence: 0.0,
            outcome_ts_from: 1_700_000_000.0,
            outcome_ts_to: 1_700_000_100.0,
        })
        .expect("build reward join");
        let rows = ch.query(&bind_params(&q)).await.expect("read back");
        assert!(
            rows.iter().any(|row| {
                row["rollout_id"] == serde_json::json!(rollout.id.to_string())
                    && row["has_outcome"] == serde_json::json!(1)
            }),
            "the produced binding must be readable via reward_join: {rows:?}"
        );
    }

    /// The worker parses the gateway's exact published JSON shape. This is the wire
    /// contract between two independently deployed services, so a pure round-trip check
    /// (no broker, no DB) guards it: the gateway's `{tenant, robot, rollout_ids}` must
    /// deserialize into the worker's `LandedEvent`.
    #[test]
    fn landed_event_parses_gateway_json() {
        let json = r#"{"tenant":"acme","robot":"robot-42","rollout_ids":["a","b"]}"#;
        let event: LandedEvent = serde_json::from_str(json).expect("parse landed event");
        assert_eq!(event.tenant, "acme");
        assert_eq!(event.robot, "robot-42");
        assert_eq!(event.rollout_ids, vec!["a".to_string(), "b".to_string()]);
    }

    /// The bound tenant is escaped into the SQL by doubling single quotes, so a tenant
    /// carrying a quote cannot break out of the string literal. Pure string check — the
    /// values come from the fail-closed builder, this only confirms the substitution step
    /// the live client needs is injection-safe for the one field that comes off the wire.
    #[test]
    fn bound_tenant_is_quote_escaped() {
        let q = reward_join(&RewardJoinParams {
            tenant: TenantId::new("ev'il"),
            metric_name: "reward".into(),
            label_kind: "terminal_outcome".into(),
            min_confidence: 0.0,
            outcome_ts_from: 0.0,
            outcome_ts_to: 1.0,
        })
        .expect("build join");
        let sql = bind_params(&q);
        // The single quote is doubled, so the literal stays one string the tenant cannot
        // escape, and the raw `ev'il` never appears unescaped.
        assert!(sql.contains("'ev''il'"), "{sql}");
    }
}
