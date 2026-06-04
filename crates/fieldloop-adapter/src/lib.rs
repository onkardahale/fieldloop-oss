//! # `fieldloop-adapter` — embodiment-adapter SDK + conformance harness
//!
//! Onboarding a robot type ("embodiment") means specifying how it threads through capture, join, and
//! curation: outcome→rollout binding windows, which outcomes need clock-safe timing, failure-class
//! and success heuristics, action canonicalization, and the training-export feature schema. This
//! crate makes that a bounded, self-checking task:
//! - [`EmbodimentAdapter`] — the one trait a robot type implements: the whole contract on a single
//!   reviewable surface.
//! - [`check_conformance`] — a pure harness that asserts the contract on *any* adapter and returns a
//!   worst-first [`ConformanceReport`], so an OEM knows their adapter is correct before it touches
//!   the live pipeline.
//! - [`registry`] — name → adapter with an exhaustive match, so a new embodiment is a compile error
//!   until wired in.
//! - [`reference`] — a small correct 6-DOF arm adapter that passes conformance: template + fixture.
//!
//! Pure and deterministic (no I/O). The adapter supplies *behavior*; [`fieldloop_config`] supplies
//! the declared attribution windows — so a fielded fleet's tunable windows live in config, the
//! per-robot logic in code.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod adapter;
pub mod conformance;
pub mod cosmos;
pub mod reference;
pub mod registry;

pub use adapter::{
    EmbodimentAdapter, FeatureSpec, LeRobotFeatures, NormalizedAction, RawAction, SuccessVerdict,
};
pub use conformance::{ConformanceFailure, ConformanceReport, Severity, check_conformance};
pub use cosmos::{
    COSMOS_GENERATOR, CosmosEpisode, CosmosStep, ingest_cosmos_export, ingest_synthetic_export,
};
pub use reference::SixDofArmAdapter;
pub use registry::{Embodiment, select_adapter};
