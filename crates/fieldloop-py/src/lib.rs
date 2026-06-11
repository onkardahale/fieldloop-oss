//! Python binding (`fieldloop`) over the Rust capture core.
//!
//! This crate is the entry point a Python robotics engineer actually calls. It is a
//! deliberately THIN PyO3 wrapper: it owns no logic of its own beyond converting
//! Python values to/from the `fieldloop-capture` API. Every behavioral guarantee —
//! non-blocking minting, drop-and-count on a full queue, off-loop draining — comes
//! from the core, not from here.
//!
//! The headline property the wrapper preserves: the in-loop call NEVER blocks the
//! control loop. A robot's control loop runs at ~50Hz, and a stall in capture would
//! stall the robot. So `log_step` only mints an id, reads a monotonic clock, and
//! hands a small record to a bounded in-memory queue; if that queue is full the
//! record is dropped and counted rather than waited on, and the minted id is still
//! returned. Heavy work (resolving strings, assembling full rollouts) happens off
//! the hot path in `drain`.

use std::cell::RefCell;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fieldloop_capture::{Capture as CoreCapture, ContextHandle, Drain as CoreDrain};
use fieldloop_types::{EpisodeId, PolicyVersion, RobotId, RobotIdentity, TenantId};

mod attribute;
mod curate;
mod doctor;
mod marshal;
mod trigger;

/// On-robot capture handle for a single robot, callable from Python.
///
/// Holds both halves of the Rust core inside one object: the hot-path `Capture`
/// (mints ids, enqueues compact records) and the off-loop `Drain` (expands those
/// records into full rollouts). Bundling them means the Python caller manages one
/// object, and `drain()` can read whatever the loop has enqueued so far.
///
/// The control loop is never blocked: `log_step` is non-allocating and never waits
/// on a full queue — it drops-and-counts instead, while still returning the id the
/// robot needs to thread onto its outbound action message.
///
/// Marked `unsendable` because the core's `Drain` holds a `std::sync::mpsc::Receiver`
/// (not `Sync`), and a `Capture` is meant to live on the one thread that owns the
/// robot's control loop anyway. PyO3 then keeps the object pinned to its creating
/// thread and raises if another thread touches it, which matches that single-thread
/// ownership exactly.
#[pyclass(unsendable)]
struct Capture {
    capture: CoreCapture,
    drain: CoreDrain,
    /// Registered context handles, kept in registration order.
    ///
    /// The core's `ContextHandle` is intentionally opaque and has no public
    /// constructor, so the binding cannot rebuild one from a bare integer. Instead it
    /// stores each handle here and hands Python the slot index as the public handle.
    /// `log_step` indexes back into this list to recover the real `ContextHandle`.
    /// `RefCell` because `register_context` mutates this list while PyO3 hands us a
    /// shared `&self` (the binding is single-threaded per `Capture` object).
    handles: RefCell<Vec<ContextHandle>>,
}

#[pymethods]
impl Capture {
    /// Construct a capture instance for `(tenant_id, robot_id)` with a bounded queue
    /// of `capacity` un-drained records.
    ///
    /// A robot id is unique only within its tenant, so both ids are required — the
    /// core carries them as one `RobotIdentity`. `capacity` is the queue depth: at
    /// most this many records buffer before further ones are dropped-and-counted,
    /// which is exactly what keeps a slow consumer from back-pressuring the loop.
    #[new]
    fn new(tenant_id: &str, robot_id: &str, capacity: usize) -> Self {
        let robot = RobotIdentity::new(TenantId::new(tenant_id), RobotId::new(robot_id));
        let (capture, drain) = CoreCapture::new(robot, capacity);
        Self {
            capture,
            drain,
            handles: RefCell::new(Vec::new()),
        }
    }

    /// Register the slowly-changing context once, off the hot path, and get back an
    /// integer handle to reference it in-loop.
    ///
    /// Policy version, model hash, embodiment, and task rarely change between steps
    /// and are strings, so registering them once (and copying only the small integer
    /// handle per step) keeps string work out of the control loop. Returns the
    /// handle to pass as `ctx` to `log_step`.
    fn register_context(
        &self,
        policy_version: &str,
        model_hash: &str,
        embodiment: &str,
        task_id: &str,
    ) -> u32 {
        let handle = self.capture.register_context(
            PolicyVersion::new(policy_version),
            model_hash,
            embodiment,
            task_id,
        );
        // Store the opaque handle and return its slot index as the public handle.
        let mut handles = self.handles.borrow_mut();
        let index = handles.len();
        handles.push(handle);
        u32::try_from(index).expect("too many registered contexts for a u32 handle")
    }

    /// The hot-path call: mint an id for this inference step and enqueue a compact
    /// record. Returns the rollout id as a string.
    ///
    /// Never blocks and never raises on a full queue — if the bounded queue is full
    /// the record is dropped and counted (see `dropped`), but the freshly minted id
    /// is STILL returned, because the robot may already be threading that id onto its
    /// outbound action message. It raises `ValueError` only on a malformed
    /// `episode_id` (must be a valid UUID string) or an unknown `ctx` handle, before
    /// anything is enqueued.
    fn log_step(
        &self,
        episode_id: &str,
        step_index: u32,
        ctx: u32,
        inference_us: u32,
    ) -> PyResult<String> {
        // Parse the episode id up front. A bad uuid is a caller error surfaced as a
        // Python ValueError, not a silently-dropped step.
        let episode = EpisodeId::from_str(episode_id)
            .map_err(|e| PyValueError::new_err(format!("invalid episode_id: {e}")))?;
        // Recover the opaque handle from the integer slot the caller was given. An
        // out-of-range integer is a caller error, surfaced as a ValueError.
        let handle = *self
            .handles
            .borrow()
            .get(ctx as usize)
            .ok_or_else(|| PyValueError::new_err(format!("unknown context handle: {ctx}")))?;
        let rollout_id = self
            .capture
            .log_step(episode, step_index, handle, inference_us);
        Ok(rollout_id.to_string())
    }

    /// Number of records dropped so far because the bounded queue was full (or the
    /// drain was gone). Ops telemetry: a rising count means draining is not keeping
    /// up with the loop.
    fn dropped(&self) -> u64 {
        self.capture.dropped()
    }

    /// Drain every currently-buffered record and return them as a list of flat
    /// dicts, one per rollout.
    ///
    /// This is the off-hot-path stage: it resolves each context handle back to its
    /// registered strings and assembles full rollouts. Each dict carries only plain
    /// Python values and is, by construction, exactly the rollout-dict shape the
    /// module-level `attribute` function consumes — so a drained rollout feeds
    /// straight into attribution with no reshaping. The keys: `rollout_id`,
    /// `tenant_id`, `robot_id`, `boot_id`, `mono_ns`, `wall_ns`, `episode_id`,
    /// `step_index`, `policy_version`, `model_hash`, `embodiment`, `task_id`,
    /// `inference_us`.
    fn drain<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let rollouts = self.drain.drain_available();
        let mut out = Vec::with_capacity(rollouts.len());
        for r in rollouts {
            let d = PyDict::new(py);
            // Ids and the policy version are stringified so the Python side gets
            // ordinary str values, not opaque wrappers.
            d.set_item("rollout_id", r.id.to_string())?;
            // The robot identity and boot id are emitted so the dict round-trips into
            // `attribute`: attribution binds per `(tenant, robot, boot)`, so without
            // these a captured rollout could not be attributed against.
            d.set_item("tenant_id", r.robot.tenant_id.to_string())?;
            d.set_item("robot_id", r.robot.robot_id.to_string())?;
            d.set_item("boot_id", r.clock.boot_id.to_string())?;
            d.set_item("episode_id", r.episode_id.to_string())?;
            d.set_item("policy_version", r.policy_version.to_string())?;
            d.set_item("model_hash", r.model_hash)?;
            d.set_item("embodiment", r.embodiment)?;
            d.set_item("task_id", r.task_id)?;
            d.set_item("step_index", r.step_index)?;
            // mono_ns is the monotonic attribution clock reading for this step;
            // wall_ns is the advisory wall estimate carried for coarse alignment only.
            d.set_item("mono_ns", r.clock.mono_ns)?;
            d.set_item("wall_ns", r.clock.ts_wall_ns)?;
            d.set_item("inference_us", r.inference_us)?;
            out.push(d);
        }
        Ok(out)
    }
}

/// The compiled core of the `fieldloop` package, imported as `fieldloop._native`.
///
/// Callers never import this name directly: the package `__init__` re-exports the
/// whole surface, so `import fieldloop` keeps working unchanged. The underscore name
/// marks the compiled module as an implementation detail, which lets the package
/// layer pure-Python tooling (the `fieldloop` CLI, the demo scenario) around the Rust
/// core without the Rust crate owning any of it.
///
/// Surfaces the OSS loop core: the on-robot `Capture` class (the non-blocking hot
/// path) plus the pure analytic stages as module-level functions — `attribute`
/// (decision→outcome attribution), `curate` (training-slice compilation), and
/// `select_uploads` (detection + budgeted selective upload). All behavior lives in the
/// Rust crates; this module only marshals values across the boundary.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Capture>()?;
    m.add_class::<attribute::PyCalibrator>()?;
    m.add_function(wrap_pyfunction!(attribute::attribute, m)?)?;
    m.add_function(wrap_pyfunction!(attribute::attribute_mcap, m)?)?;
    m.add_function(wrap_pyfunction!(attribute::fit_calibrator, m)?)?;
    m.add_function(wrap_pyfunction!(doctor::doctor, m)?)?;
    m.add_function(wrap_pyfunction!(curate::curate, m)?)?;
    m.add_function(wrap_pyfunction!(trigger::select_uploads, m)?)?;
    Ok(())
}
