//! The calibration seam: map a raw attribution score onto a calibrated confidence.
//!
//! Confidence must be *calibratable, not asserted*. The cascade computes a raw,
//! mechanical score (recency for temporal, coverage for synthetic-absence); a
//! [`Calibrator`] then maps that raw score onto the `[0, 1]` confidence that lands on
//! the [`fieldloop_types::Feedback`] row, bucketed by `(join_method, embodiment)`.
//!
//! The seam exists because the *right* mapping is empirical: in production it is fit
//! from the manual-label ground truth (a curator's [`fieldloop_types::JoinMethod::Manual`]
//! bindings are the truth set), so that, say, a raw temporal score of `0.8` is
//! reported as whatever fraction of `0.8`-scored bindings actually turned out
//! correct for that robot type. We do not ship a fitted curve here (there is no
//! field data yet); instead we ship a documented, simple default and keep the trait
//! open so a fitted calibrator can be swapped in without touching the cascade.

use std::collections::HashMap;

use fieldloop_types::{Feedback, FeedbackTarget, FeedbackValue, JoinMethod, Rollout};

/// Maps a raw attribution score to a calibrated confidence, per
/// `(join_method, embodiment)`.
///
/// A trait (not a hard-coded function) so the empirical, fit-from-ground-truth curve
/// can replace the default without the cascade changing. Implementors MUST return a
/// value in `[0, 1]`; the engine additionally clamps the result so a buggy calibrator
/// can never emit an out-of-range confidence onto a row.
pub trait Calibrator: std::fmt::Debug {
    /// Calibrate `raw_score` (already in `[0, 1]`) for the given method and
    /// embodiment. The method and embodiment are passed so a real implementation can
    /// look up the fitted curve for that exact bucket — failure timing on a fast arm
    /// calibrates differently from a slow mobile base.
    fn calibrate(&self, method: JoinMethod, embodiment: &str, raw_score: f64) -> f64;

    /// A short label identifying this calibrator's curve set, written onto each row's
    /// `calibration_version` so a confidence number is traceable to how it was
    /// calibrated. Defaults to the config's calibration version via the engine; a
    /// fitted calibrator overrides this with its own fit identifier.
    fn version_tag(&self) -> Option<&str> {
        None
    }
}

/// The default calibrator: the identity map, surfacing the raw score unchanged as
/// the confidence.
///
/// Identity is chosen deliberately as the honest default before any field data
/// exists: it neither inflates nor deflates the raw recency/coverage score, so the
/// confidence on a row means exactly "this is how close in time / how covered the
/// evidence was", with no pretend precision from a curve we have not earned the data
/// to fit. The one exception is [`JoinMethod::Explicit`] / [`JoinMethod::Manual`],
/// which are certain by construction (an explicit id was threaded, or a curator
/// asserted the binding) and so map to `1.0` regardless of any raw score — there is
/// nothing to calibrate when the target is known rather than inferred.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityCalibrator;

impl Calibrator for IdentityCalibrator {
    fn calibrate(&self, method: JoinMethod, _embodiment: &str, raw_score: f64) -> f64 {
        match method {
            // Known, not inferred: certain by construction.
            JoinMethod::Explicit | JoinMethod::Manual => 1.0,
            // Inferred methods surface their raw score unchanged under the identity
            // default; a fitted calibrator would bend this toward observed accuracy.
            JoinMethod::Temporal
            | JoinMethod::Spatial
            | JoinMethod::Causal
            | JoinMethod::SyntheticAbsence => raw_score.clamp(0.0, 1.0),
        }
    }
}

/// A confidence calibrator **fit from the curator's ground-truth labels** — the empirical
/// curve the module doc describes, finally realized.
///
/// `IdentityCalibrator` reports the raw recency/coverage score *as* the confidence, which is
/// only honest before any field data exists. Once curators have confirmed/retracted automated
/// bindings, the truth is measurable: of all temporal bindings that scored ~0.8 on a given
/// embodiment, what fraction actually turned out correct? `FittedCalibrator` answers that with
/// an **isotonic regression** (a monotone, non-decreasing step function — the natural fit since
/// a higher raw score should never calibrate to a lower confidence) per `(join_method,
/// embodiment)` bucket. A bucket with too few labels falls back to identity, so the calibrator
/// never invents precision it has not earned, and `Explicit`/`Manual` stay certain by
/// construction.
#[derive(Debug, Clone)]
pub struct FittedCalibrator {
    /// Per method, then per embodiment: the fitted curve as sorted `(raw_threshold,
    /// confidence)` points with non-decreasing confidence. Empty/absent for unlearned
    /// buckets. Nested (`method -> embodiment -> curve`) rather than keyed by a
    /// `(method, String)` tuple so the per-row `calibrate` lookup borrows the embodiment
    /// `&str` instead of allocating a `String` on every binding it scores.
    curves: HashMap<JoinMethod, HashMap<String, Vec<(f64, f64)>>>,
    /// Identifier written onto each row's `calibration_version`, so a confidence is traceable
    /// to the fit that produced it (distinct from the identity default).
    tag: String,
}

impl FittedCalibrator {
    /// Fit calibration curves from feedback history. The curator's non-retracted
    /// [`JoinMethod::Manual`] rows are the ground truth for a target's outcome polarity; an
    /// automated (inferred) binding is a **correct** sample when it agrees with that truth and
    /// an **incorrect** sample when it disagrees or was retracted. Samples are bucketed by
    /// `(method, embodiment)` — embodiment resolved from the target's rollout, since a fast arm
    /// and a slow mobile base calibrate differently — and each bucket with at least `min_labels`
    /// samples is fit with isotonic regression. Buckets below the threshold are left unlearned
    /// (identity at calibrate time), so a thin bucket never fabricates a curve.
    #[must_use]
    pub fn fit(feedback: &[Feedback], rollouts: &[Rollout], min_labels: usize) -> Self {
        // Resolve a target to its embodiment via the source rollouts (by rollout id, and by
        // episode id for episode-grain feedback).
        let mut emb_by_rollout: HashMap<String, &str> = HashMap::new();
        let mut emb_by_episode: HashMap<String, &str> = HashMap::new();
        for r in rollouts {
            emb_by_rollout.insert(format!("r:{}", r.id), r.embodiment.as_str());
            emb_by_episode
                .entry(format!("e:{}", r.episode_id))
                .or_insert(r.embodiment.as_str());
        }
        let embodiment_of = |t: &FeedbackTarget| -> Option<&str> {
            match t {
                FeedbackTarget::Rollout(id) => emb_by_rollout.get(&format!("r:{id}")).copied(),
                FeedbackTarget::Episode(id) => emb_by_episode.get(&format!("e:{id}")).copied(),
            }
        };

        // Ground-truth failure polarity per target from the curator's manual, non-retracted
        // labels (latest write wins; manual is the asserted truth).
        let mut truth: HashMap<String, bool> = HashMap::new();
        for f in feedback {
            if f.join_method == JoinMethod::Manual && !f.retracted {
                truth.insert(target_key(&f.target), value_is_failure(&f.value));
            }
        }

        // Collect (raw_score, correct) samples per (method, embodiment).
        let mut samples: HashMap<(JoinMethod, String), Vec<(f64, f64)>> = HashMap::new();
        for f in feedback {
            // Only inferred methods are calibrated; Explicit/Manual are certain by construction.
            if matches!(f.join_method, JoinMethod::Explicit | JoinMethod::Manual) {
                continue;
            }
            let Some(emb) = embodiment_of(&f.target) else {
                continue;
            };
            // A retracted automated binding is ground-truth incorrect on its own; otherwise it
            // needs a manual truth to judge against (agreement = correct).
            let correct = if f.retracted {
                Some(false)
            } else {
                truth
                    .get(&target_key(&f.target))
                    .map(|&truth_fail| value_is_failure(&f.value) == truth_fail)
            };
            if let Some(c) = correct {
                samples
                    .entry((f.join_method, emb.to_string()))
                    .or_default()
                    .push((f64::from(f.join_confidence), if c { 1.0 } else { 0.0 }));
            }
        }

        let mut curves: HashMap<JoinMethod, HashMap<String, Vec<(f64, f64)>>> = HashMap::new();
        for ((method, embodiment), pts) in samples {
            if pts.len() >= min_labels {
                curves
                    .entry(method)
                    .or_default()
                    .insert(embodiment, isotonic(pts));
            }
        }
        Self {
            curves,
            tag: "fitted-v1".to_string(),
        }
    }
}

impl Calibrator for FittedCalibrator {
    fn calibrate(&self, method: JoinMethod, embodiment: &str, raw_score: f64) -> f64 {
        match method {
            // Known, not inferred: certain by construction, exactly as the identity default.
            JoinMethod::Explicit | JoinMethod::Manual => 1.0,
            // Borrowed two-level lookup: no `String` allocation on the per-row hot path.
            _ => match self
                .curves
                .get(&method)
                .and_then(|by_embodiment| by_embodiment.get(embodiment))
            {
                Some(curve) => eval_isotonic(curve, raw_score.clamp(0.0, 1.0)),
                // Unlearned bucket: fall back to identity rather than invent a confidence.
                None => raw_score.clamp(0.0, 1.0),
            },
        }
    }

    fn version_tag(&self) -> Option<&str> {
        Some(&self.tag)
    }
}

/// A stable string key for a feedback target, so manual ground truth and automated bindings on
/// the same rollout/episode collate regardless of the id type.
fn target_key(t: &FeedbackTarget) -> String {
    match t {
        FeedbackTarget::Rollout(id) => format!("r:{id}"),
        FeedbackTarget::Episode(id) => format!("e:{id}"),
    }
}

/// Whether a feedback value denotes a failure outcome — the polarity calibration agreement is
/// measured on. A boolean `false`, any named failure class, and a correction-demonstration ref
/// (which only exists because a human had to take over) are failures; a boolean `true` or a
/// reward at/above the midpoint is a success.
fn value_is_failure(v: &FeedbackValue) -> bool {
    match v {
        FeedbackValue::Boolean { value } => !*value,
        FeedbackValue::Float { value } => *value < 0.5,
        FeedbackValue::FailureClass { .. } | FeedbackValue::DemonstrationRef { .. } => true,
    }
}

/// Isotonic regression by pool-adjacent-violators: collapse the points into blocks whose mean
/// confidence is non-decreasing in the raw score. Returns sorted `(x_left, mean)` blocks. This
/// is the monotone calibration curve — a higher raw score can never map to a lower confidence,
/// which is the one property a recency/coverage score must preserve.
fn isotonic(mut pts: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Each block is (sum_y, count, x_left).
    let mut blocks: Vec<(f64, f64, f64)> = Vec::new();
    for (x, y) in pts {
        blocks.push((y, 1.0, x));
        // Merge backwards while the previous block's mean exceeds this one's (a violation).
        while blocks.len() >= 2 {
            let (s2, c2, _) = blocks[blocks.len() - 1];
            let (s1, c1, x1) = blocks[blocks.len() - 2];
            if s1 / c1 > s2 / c2 {
                blocks.truncate(blocks.len() - 2);
                blocks.push((s1 + s2, c1 + c2, x1));
            } else {
                break;
            }
        }
    }
    blocks
        .into_iter()
        .map(|(sum, count, x_left)| (x_left, sum / count))
        .collect()
}

/// Evaluate the isotonic curve at `x`: the confidence of the last block whose `x_left <= x`
/// (the first block's value when `x` is below all thresholds). The curve is monotone, so this
/// is a well-defined non-decreasing calibration.
fn eval_isotonic(curve: &[(f64, f64)], x: f64) -> f64 {
    let mut out = curve.first().map_or(x, |&(_, m)| m);
    for &(x_left, mean) in curve {
        if x_left <= x {
            out = mean;
        } else {
            break;
        }
    }
    out.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity passes an inferred score straight through, so the confidence on a
    /// temporal row is exactly the recency score with no hidden curve.
    #[test]
    fn identity_passes_inferred_score_through() {
        let c = IdentityCalibrator;
        assert_eq!(c.calibrate(JoinMethod::Temporal, "ur5e", 0.42), 0.42);
        assert_eq!(c.calibrate(JoinMethod::SyntheticAbsence, "ur5e", 0.9), 0.9);
    }

    /// Explicit and manual are certain by construction and report 1.0 regardless of
    /// the raw score handed in.
    #[test]
    fn identity_forces_certain_methods_to_one() {
        let c = IdentityCalibrator;
        assert_eq!(c.calibrate(JoinMethod::Explicit, "ur5e", 0.0), 1.0);
        assert_eq!(c.calibrate(JoinMethod::Manual, "ur5e", 0.3), 1.0);
    }

    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, FeedbackId, LabelKind, MonoClock, PayloadRef,
        PolicyVersion, RobotId, RobotIdentity, RolloutId, TenantId,
    };

    fn roll(embodiment: &str) -> Rollout {
        Rollout::new(
            RobotIdentity::new(TenantId::new("t"), RobotId::new("r")),
            EpisodeId::new(),
            0,
            MonoClock::new(BootId::new(), 1000, 1_700_000_000_000_000_000),
            PolicyVersion::new("pick@v1.0.0+aaaaaaaaaaaa"),
            "sha256:x".to_string(),
            embodiment.to_string(),
            "task".to_string(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            100,
        )
    }

    fn fb(
        target: RolloutId,
        method: JoinMethod,
        conf: f32,
        fail: bool,
        retracted: bool,
    ) -> Feedback {
        Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("t"),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "task_success".to_string(),
            value: FeedbackValue::Boolean { value: !fail },
            join_method: method,
            join_confidence: conf,
            join_version: "join-v1".to_string(),
            calibration_version: "id".to_string(),
            source_outcome_id: None,
            delay_ms: Some(1),
            retracted,
            dedup_key: format!("dk:{target}:{method:?}:{conf}"),
            outcome_ts_ns: 1,
            credit_weight: 1.0,
            contributing_set_id: None,
        }
    }

    /// A fitted calibrator bends the raw recency score toward observed accuracy: high-raw
    /// temporal bindings that curators CONFIRMED calibrate to high confidence, and low-raw
    /// bindings curators CONTRADICTED calibrate to low confidence — monotonically, per
    /// embodiment, with an identity fallback for unlearned buckets and certain methods at 1.0.
    /// This is the "raw score == confidence" lie, killed.
    #[test]
    fn fitted_calibrator_bends_raw_toward_observed_accuracy() {
        let mut feedback = Vec::new();
        let mut rollouts = Vec::new();
        // High-raw temporal bindings that a manual label CONFIRMED (same failure polarity).
        for i in 0..6 {
            let r = roll("arm");
            let id = r.id;
            rollouts.push(r);
            let conf = 0.85 + 0.02 * (i as f32);
            feedback.push(fb(id, JoinMethod::Temporal, conf, true, false));
            feedback.push(fb(id, JoinMethod::Manual, 1.0, true, false));
        }
        // Low-raw temporal bindings the manual label CONTRADICTED (opposite polarity).
        for i in 0..6 {
            let r = roll("arm");
            let id = r.id;
            rollouts.push(r);
            let conf = 0.10 + 0.02 * (i as f32);
            feedback.push(fb(id, JoinMethod::Temporal, conf, true, false));
            feedback.push(fb(id, JoinMethod::Manual, 1.0, false, false));
        }

        let cal = FittedCalibrator::fit(&feedback, &rollouts, 4);
        let hi = cal.calibrate(JoinMethod::Temporal, "arm", 0.9);
        let lo = cal.calibrate(JoinMethod::Temporal, "arm", 0.12);
        assert!(
            hi > 0.7,
            "confirmed high-raw bindings -> high confidence, got {hi}"
        );
        assert!(
            lo < 0.3,
            "contradicted low-raw bindings -> low confidence, got {lo}"
        );
        assert!(hi >= lo, "calibration is monotone in the raw score");

        // An unlearned bucket falls back to identity (no invented precision).
        assert_eq!(cal.calibrate(JoinMethod::Temporal, "other-arm", 0.5), 0.5);
        // Certain methods stay 1.0; the fit is traceable via its version tag.
        assert_eq!(cal.calibrate(JoinMethod::Manual, "arm", 0.0), 1.0);
        assert_eq!(cal.calibrate(JoinMethod::Explicit, "arm", 0.0), 1.0);
        assert_eq!(cal.version_tag(), Some("fitted-v1"));

        // Too few labels to earn a curve -> identity, not a fabricated one.
        let thin = FittedCalibrator::fit(&feedback, &rollouts, 1000);
        assert_eq!(thin.calibrate(JoinMethod::Temporal, "arm", 0.9), 0.9);
    }
}
