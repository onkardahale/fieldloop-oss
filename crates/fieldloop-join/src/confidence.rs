//! The raw scoring functions: how a recency delay (temporal) and a coverage
//! fraction (synthetic-absence) turn into a raw score in `[0, 1]`.
//!
//! These are kept separate from the [`crate::calibrator`] seam on purpose: this
//! module emits a *raw, mechanical* score that is a documented, reproducible
//! function of the inputs and nothing else. The calibrator then maps that raw score
//! onto a calibrated confidence per `(join_method, embodiment)`. Splitting the two
//! means the raw score never silently bakes in a fitted curve, and a re-run on
//! identical inputs always yields the identical raw score (so the binding is
//! reproducible).

/// The raw temporal score: closer in time to the rollout means a higher score.
///
/// We use a linear ramp from `1.0` at zero delay down to `0.0` at the window edge:
/// `score = 1 - (delay / window)`, clamped to `[0, 1]`. A linear ramp is chosen
/// over a flat constant because a flat constant would assert the same confidence for
/// an outcome that landed `1ms` after a rollout as for one that landed at the very
/// edge of the window — which is exactly the ambiguity the score is meant to
/// surface. The ramp is monotone (nearer is never scored lower than farther) and
/// reproducible (a pure function of the two integers), so a re-attribution on the
/// same inputs lands on the same number.
///
/// A `delay_ns` of `0` scores `1.0`; a delay equal to (or beyond) `window_ns` scores
/// `0.0`. A zero-width window is degenerate and scores `0.0` for any positive delay
/// (and `1.0` only for an exactly-coincident event), since there is no room inside it
/// to be "close".
///
/// `delay_ns` must be non-negative (the outcome is at or after the rollout); a
/// negative delay would mean the outcome preceded the rollout and is not a temporal
/// candidate at all, so it is clamped to score `0.0`.
#[must_use]
pub fn temporal_raw_score(delay_ns: i128, window_ns: i128) -> f64 {
    if delay_ns < 0 {
        // The outcome preceded the rollout: not a forward-in-time candidate.
        return 0.0;
    }
    if window_ns <= 0 {
        // Degenerate window: only an exactly-coincident event is "close".
        return if delay_ns == 0 { 1.0 } else { 0.0 };
    }
    if delay_ns >= window_ns {
        return 0.0;
    }
    // Linear ramp 1.0 -> 0.0 across the window. Done in f64; the delay and window
    // are bounded by the configured window (milliseconds), so the magnitudes are far
    // inside f64's exact-integer range and the division is well-conditioned.
    let frac = delay_ns as f64 / window_ns as f64;
    (1.0 - frac).clamp(0.0, 1.0)
}

/// The raw synthetic-absence score: more fully a window is proven covered by
/// heartbeats, the higher the score.
///
/// The coverage fraction is already in `[0, 1]` (it is the share of the rollout's
/// window that heartbeats demonstrably blanket without a gap larger than the
/// allowed inter-arrival bound). We pass it through as the raw score because a
/// fuller coverage is exactly a stronger claim that "nothing happened here", and the
/// claim should never be reported at a flat constant that ignores how complete the
/// coverage actually was. Clamped to `[0, 1]` defensively in case a caller passes a
/// fraction slightly outside the range from floating-point accumulation.
#[must_use]
pub fn synthetic_absence_raw_score(coverage_fraction: f64) -> f64 {
    coverage_fraction.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero delay scores the maximum, the window edge scores the minimum, and a
    /// mid-window delay lands strictly between — so the ramp actually discriminates
    /// by recency rather than asserting one flat number.
    #[test]
    fn temporal_ramp_is_monotone_and_bounded() {
        assert_eq!(temporal_raw_score(0, 1000), 1.0);
        assert_eq!(temporal_raw_score(1000, 1000), 0.0);
        assert_eq!(temporal_raw_score(1500, 1000), 0.0);
        let mid = temporal_raw_score(250, 1000);
        assert!(
            mid > 0.0 && mid < 1.0,
            "mid-window must be strictly between"
        );
        // Nearer is never scored lower than farther.
        assert!(temporal_raw_score(100, 1000) > temporal_raw_score(900, 1000));
    }

    /// A negative delay (outcome before rollout) is not a forward candidate and
    /// scores zero rather than wrapping into a spuriously-high value.
    #[test]
    fn temporal_negative_delay_scores_zero() {
        assert_eq!(temporal_raw_score(-5, 1000), 0.0);
    }

    /// Coverage passes through but is clamped, so an out-of-range fraction can never
    /// leak a confidence above 1.0 or below 0.0.
    #[test]
    fn coverage_passes_through_clamped() {
        assert_eq!(synthetic_absence_raw_score(0.75), 0.75);
        assert_eq!(synthetic_absence_raw_score(1.5), 1.0);
        assert_eq!(synthetic_absence_raw_score(-0.2), 0.0);
    }
}
