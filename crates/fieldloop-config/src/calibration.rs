//! Calibration knobs: the tunable numbers that shape attribution, kept versioned in
//! config rather than as code literals.
//!
//! These are the dials operators turn as they learn how a fleet behaves — a default
//! temporal window, the coverage factor that decides when enough heartbeats prove a
//! window was covered. Carrying a `version` string means every confidence number the
//! runtime emits can be traced back to the exact knob set that produced it, so a
//! later change to the dials does not silently rewrite the meaning of past scores.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// The raw, unvalidated calibration section as written in TOML.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedCalibrationConfig {
    /// An opaque label identifying this set of knobs; required and non-empty so the
    /// confidence numbers derived from these dials remain traceable to them.
    pub version: String,
    /// The default temporal attribution window, in milliseconds, used when an
    /// embodiment does not override it for a given outcome kind.
    pub temporal_window_default_ms: u64,
    /// The coverage factor that decides how many heartbeats are enough to treat a
    /// window as covered (and therefore eligible for synthesized "nothing happened
    /// here").
    pub heartbeat_coverage_k: f64,
}

/// A validated calibration configuration.
///
/// Structurally identical to its raw form; the separate type marks that its
/// invariant (a non-empty version) has been checked, so holding one is proof the
/// check ran.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationConfig {
    /// The non-empty version label for this knob set.
    pub version: String,
    /// The default temporal attribution window in milliseconds.
    pub temporal_window_default_ms: u64,
    /// The heartbeat coverage factor.
    pub heartbeat_coverage_k: f64,
}

impl UninitializedCalibrationConfig {
    /// Validate that the version is non-empty and produce the [`CalibrationConfig`].
    pub(crate) fn load(self) -> Result<CalibrationConfig, ConfigError> {
        if self.version.trim().is_empty() {
            return Err(ConfigError::EmptyCalibrationVersion);
        }
        Ok(CalibrationConfig {
            version: self.version,
            temporal_window_default_ms: self.temporal_window_default_ms,
            heartbeat_coverage_k: self.heartbeat_coverage_k,
        })
    }
}
