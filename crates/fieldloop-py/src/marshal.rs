//! Shared dict↔schema marshaling for the Python bindings.
//!
//! `attribute`, `curate`, and `select_uploads` all speak the same flat dict shapes for
//! a rollout, an outcome, and a feedback row, so the conversion lives here once. The
//! rule throughout: a missing required key, or a present-but-wrong-typed value, is a
//! caller error surfaced as a Python `ValueError` BEFORE any engine runs — never a
//! silently-defaulted field. Enum strings are the canonical snake_case spelling (the
//! same the config tables and the wire form use), in both directions.

use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fieldloop_curation::SITE_TAG_KEY;
use fieldloop_trigger::POLICY_CONFIDENCE_TAG;
use fieldloop_types::{
    BootId, BoundedBlob, EpisodeId, FailureClass, Feedback, FeedbackTarget, FeedbackValue,
    JoinMethod, LabelKind, MonoClock, OutcomeEvent, OutcomeId, OutcomeKind, PayloadRef,
    PolicyVersion, Provenance, RobotId, RobotIdentity, Rollout, TenantId,
};

// ---------------------------------------------------------------------------
// Field readers.
// ---------------------------------------------------------------------------

/// The value at `key`, or `None` if the key is absent or explicitly `None`. Treating a
/// present `None` like absence lets a caller pass `None` for an optional field without
/// it being mistaken for a typed value.
pub(crate) fn field<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => Some(v),
        _ => None,
    })
}

/// Required string field, or a `ValueError` naming the missing/mistyped key.
pub(crate) fn req_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    field(d, key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field `{key}`")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a string")))
}

/// Optional string field; `None` when absent or `None`.
pub(crate) fn opt_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match field(d, key)? {
        Some(v) => Ok(Some(v.extract().map_err(|_| {
            PyValueError::new_err(format!("field `{key}` must be a string"))
        })?)),
        None => Ok(None),
    }
}

/// Required `u64` field, or a `ValueError`.
pub(crate) fn req_u64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<u64> {
    field(d, key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field `{key}`")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a non-negative integer")))
}

/// Required `u32` field, or a `ValueError`.
pub(crate) fn req_u32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<u32> {
    field(d, key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field `{key}`")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a non-negative integer")))
}

/// Optional `u32` field, defaulting to `default` when absent.
pub(crate) fn opt_u32(d: &Bound<'_, PyDict>, key: &str, default: u32) -> PyResult<u32> {
    match field(d, key)? {
        Some(v) => v.extract().map_err(|_| {
            PyValueError::new_err(format!("field `{key}` must be a non-negative integer"))
        }),
        None => Ok(default),
    }
}

/// Optional `i64` field, defaulting to `default` when absent.
pub(crate) fn opt_i64(d: &Bound<'_, PyDict>, key: &str, default: i64) -> PyResult<i64> {
    match field(d, key)? {
        Some(v) => v
            .extract()
            .map_err(|_| PyValueError::new_err(format!("field `{key}` must be an integer"))),
        None => Ok(default),
    }
}

/// Optional `i64` field as an `Option` (a `None`/absent value stays `None`).
pub(crate) fn opt_i64_opt(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    match field(d, key)? {
        Some(v) => Ok(Some(v.extract().map_err(|_| {
            PyValueError::new_err(format!("field `{key}` must be an integer"))
        })?)),
        None => Ok(None),
    }
}

/// Required `f32` field, or a `ValueError`.
pub(crate) fn req_f32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<f32> {
    field(d, key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field `{key}`")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a number")))
}

/// Optional `f32` field as an `Option`.
pub(crate) fn opt_f32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f32>> {
    match field(d, key)? {
        Some(v) => Ok(Some(v.extract().map_err(|_| {
            PyValueError::new_err(format!("field `{key}` must be a number"))
        })?)),
        None => Ok(None),
    }
}

/// Required `bool` field, or a `ValueError`.
pub(crate) fn req_bool(d: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    field(d, key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field `{key}`")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a boolean")))
}

/// Optional `bool` field, defaulting to `default` when absent.
pub(crate) fn opt_bool(d: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match field(d, key)? {
        Some(v) => v
            .extract()
            .map_err(|_| PyValueError::new_err(format!("field `{key}` must be a boolean"))),
        None => Ok(default),
    }
}

/// Parse an id newtype from a dict string, tagging the error with the field name.
pub(crate) fn parse_id<T: FromStr>(s: &str, key: &str) -> PyResult<T>
where
    T::Err: std::fmt::Display,
{
    T::from_str(s)
        .map_err(|e| PyValueError::new_err(format!("field `{key}` is not a valid UUID: {e}")))
}

// ---------------------------------------------------------------------------
// Enum parsing (string -> closed taxonomy). An unknown variant is rejected so a
// typo can never silently change the meaning of a record.
// ---------------------------------------------------------------------------

/// Map the snake_case outcome-kind string onto the closed [`OutcomeKind`] taxonomy.
pub(crate) fn parse_outcome_kind(s: &str) -> PyResult<OutcomeKind> {
    Ok(match s {
        "teleop_takeover" => OutcomeKind::TeleopTakeover,
        "e_stop" => OutcomeKind::EStop,
        "collision" => OutcomeKind::Collision,
        "downstream_failure" => OutcomeKind::DownstreamFailure,
        "heartbeat" => OutcomeKind::Heartbeat,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown outcome_kind `{other}` (expected one of: teleop_takeover, e_stop, \
                 collision, downstream_failure, heartbeat)"
            )));
        }
    })
}

/// Map the snake_case failure-class string onto the closed [`FailureClass`] taxonomy.
pub(crate) fn parse_failure_class(s: &str) -> PyResult<FailureClass> {
    Ok(match s {
        "perception" => FailureClass::Perception,
        "planning" => FailureClass::Planning,
        "manipulation" => FailureClass::Manipulation,
        "hardware" => FailureClass::Hardware,
        "network" => FailureClass::Network,
        "environment" => FailureClass::Environment,
        "operator" => FailureClass::Operator,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown failure_class `{other}`"
            )));
        }
    })
}

/// Map the snake_case label-kind string onto the closed [`LabelKind`] taxonomy.
pub(crate) fn parse_label_kind(s: &str) -> PyResult<LabelKind> {
    Ok(match s {
        "terminal_outcome" => LabelKind::TerminalOutcome,
        "intervention" => LabelKind::Intervention,
        "annotation" => LabelKind::Annotation,
        "episode_return" => LabelKind::EpisodeReturn,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown label_kind `{other}`"
            )));
        }
    })
}

/// Map the snake_case join-method string onto the closed [`JoinMethod`] taxonomy.
pub(crate) fn parse_join_method(s: &str) -> PyResult<JoinMethod> {
    Ok(match s {
        "explicit" => JoinMethod::Explicit,
        "temporal" => JoinMethod::Temporal,
        "spatial" => JoinMethod::Spatial,
        "causal" => JoinMethod::Causal,
        "manual" => JoinMethod::Manual,
        "synthetic_absence" => JoinMethod::SyntheticAbsence,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown join_method `{other}`"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Enum spelling (closed taxonomy -> snake_case string), for output dicts.
// ---------------------------------------------------------------------------

/// Snake_case spelling of an [`OutcomeKind`].
pub(crate) fn outcome_kind_str(kind: OutcomeKind) -> &'static str {
    match kind {
        OutcomeKind::TeleopTakeover => "teleop_takeover",
        OutcomeKind::EStop => "e_stop",
        OutcomeKind::Collision => "collision",
        OutcomeKind::DownstreamFailure => "downstream_failure",
        OutcomeKind::Heartbeat => "heartbeat",
    }
}

/// Snake_case spelling of a [`FailureClass`].
pub(crate) fn failure_class_str(class: FailureClass) -> &'static str {
    match class {
        FailureClass::Perception => "perception",
        FailureClass::Planning => "planning",
        FailureClass::Manipulation => "manipulation",
        FailureClass::Hardware => "hardware",
        FailureClass::Network => "network",
        FailureClass::Environment => "environment",
        FailureClass::Operator => "operator",
    }
}

/// Snake_case spelling of a [`LabelKind`].
pub(crate) fn label_kind_str(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::TerminalOutcome => "terminal_outcome",
        LabelKind::Intervention => "intervention",
        LabelKind::Annotation => "annotation",
        LabelKind::EpisodeReturn => "episode_return",
    }
}

/// Snake_case spelling of a [`JoinMethod`].
pub(crate) fn join_method_str(method: JoinMethod) -> &'static str {
    match method {
        JoinMethod::Explicit => "explicit",
        JoinMethod::Temporal => "temporal",
        JoinMethod::Spatial => "spatial",
        JoinMethod::Causal => "causal",
        JoinMethod::Manual => "manual",
        JoinMethod::SyntheticAbsence => "synthetic_absence",
    }
}

// ---------------------------------------------------------------------------
// Record marshaling.
// ---------------------------------------------------------------------------

/// Build the robot identity + monotonic clock shared by the rollout and outcome
/// readers — both records carry exactly `(tenant_id, robot_id)` and
/// `(boot_id, mono_ns, wall_ns)`, so they parse identically.
fn read_robot_and_clock(d: &Bound<'_, PyDict>) -> PyResult<(RobotIdentity, MonoClock)> {
    let robot = RobotIdentity::new(
        TenantId::new(req_str(d, "tenant_id")?),
        RobotId::new(req_str(d, "robot_id")?),
    );
    let boot_id: BootId = parse_id(&req_str(d, "boot_id")?, "boot_id")?;
    // mono_ns is the skew-free attribution authority; wall_ns is an advisory estimate
    // that defaults to 0 because it never decides a binding on its own.
    let clock = MonoClock::new(boot_id, req_u64(d, "mono_ns")?, opt_i64(d, "wall_ns", 0)?);
    Ok((robot, clock))
}

/// Marshal one rollout dict into a [`Rollout`].
///
/// The caller's `rollout_id` overrides the freshly-minted id `Rollout::new` would
/// assign, so an outcome that threads an explicit rollout id binds to *this exact*
/// rollout. Three optional fields drive downstream stages without bloating the common
/// case: `policy_confidence` (the on-robot pre-filter's signal, written to the tag the
/// low-confidence detector reads), `site` (the curation site filter's tag), and
/// `synthetic_generator` (marks the rollout synthetic, so curation can keep it opt-in).
/// Payload pointers and inline context are defaulted empty — attribution and curation
/// read none of them.
pub(crate) fn dict_to_rollout(d: &Bound<'_, PyDict>) -> PyResult<Rollout> {
    let (robot, clock) = read_robot_and_clock(d)?;
    let episode_id: EpisodeId = parse_id(&req_str(d, "episode_id")?, "episode_id")?;
    let mut rollout = Rollout::new(
        robot,
        episode_id,
        req_u32(d, "step_index")?,
        clock,
        PolicyVersion::new(req_str(d, "policy_version")?),
        opt_str(d, "model_hash")?.unwrap_or_default(),
        req_str(d, "embodiment")?,
        req_str(d, "task_id")?,
        PayloadRef::none(),
        PayloadRef::none(),
        BoundedBlob::empty(),
        opt_u32(d, "inference_us", 0)?,
    );
    rollout.id = parse_id(&req_str(d, "rollout_id")?, "rollout_id")?;
    if let Some(confidence) = opt_f32(d, "policy_confidence")? {
        rollout
            .tags
            .insert(POLICY_CONFIDENCE_TAG.to_string(), confidence.to_string());
    }
    if let Some(site) = opt_str(d, "site")? {
        rollout.tags.insert(SITE_TAG_KEY.to_string(), site);
    }
    if let Some(generator) = opt_str(d, "synthetic_generator")? {
        rollout.provenance = Provenance::synthetic(generator);
    }
    Ok(rollout)
}

/// An outcome dict resolves to either a failure/success event to attribute, or a
/// heartbeat coverage sample — never both. Routing the heartbeat here keeps it out of
/// the per-outcome failure cascade exactly as the engine requires.
pub(crate) enum Routed {
    /// A real outcome to push through the attribution cascade / detectors.
    Outcome(Box<OutcomeEvent>),
    /// A heartbeat coverage sample feeding the synthetic-absence check.
    Heartbeat(fieldloop_join::Heartbeat),
}

/// Route one outcome to the failure cascade or the heartbeat coverage set — the single
/// home of the heartbeat-vs-outcome policy, shared by the dict and file-import paths.
pub(crate) fn route_outcome(outcome: OutcomeEvent) -> Routed {
    if outcome.outcome_kind == OutcomeKind::Heartbeat {
        return Routed::Heartbeat(fieldloop_join::Heartbeat {
            robot: outcome.robot,
            clock: outcome.clock,
        });
    }
    Routed::Outcome(Box::new(outcome))
}

/// Marshal one outcome dict, routing a heartbeat kind to the coverage set.
///
/// An optional `explicit_rollout_id` is the highest-certainty channel: when present the
/// engine binds to it at confidence `1.0`, and the trigger detectors use it to pair an
/// outcome with the rollout it fired on. An optional `outcome_id` lets a caller pin the
/// id so a returned record can be matched back to its input; absent, a fresh one is minted.
pub(crate) fn dict_to_outcome(d: &Bound<'_, PyDict>) -> PyResult<Routed> {
    let (robot, clock) = read_robot_and_clock(d)?;
    let kind = parse_outcome_kind(&req_str(d, "outcome_kind")?)?;
    // Heartbeat early-return: a heartbeat dict carries no explicit_rollout_id/outcome_id
    // worth reading, so we route before parsing those optional fields.
    if kind == OutcomeKind::Heartbeat {
        return Ok(route_outcome(OutcomeEvent::new(
            robot,
            clock,
            kind,
            BoundedBlob::empty(),
        )));
    }
    let mut outcome = OutcomeEvent::new(robot, clock, kind, BoundedBlob::empty());
    if let Some(explicit) = opt_str(d, "explicit_rollout_id")? {
        outcome.explicit_rollout_id = Some(parse_id(&explicit, "explicit_rollout_id")?);
    }
    if let Some(id) = opt_str(d, "outcome_id")? {
        outcome.id = parse_id::<OutcomeId>(&id, "outcome_id")?;
    }
    Ok(route_outcome(outcome))
}

/// Convert one [`Feedback`] binding into a flat dict. The `target` and `value` sum types
/// are projected onto `(target_type, target_id)` and `(value_type, value/...)` pairs so
/// a Python caller reads ordinary keys, never an opaque tagged object. The dict is the
/// exact shape [`dict_to_feedback`] reads back, so `attribute`'s output feeds `curate`.
pub(crate) fn feedback_to_dict<'py>(
    py: Python<'py>,
    fb: &Feedback,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("feedback_id", fb.id.to_string())?;
    d.set_item("tenant_id", fb.tenant_id.to_string())?;

    let (target_type, target_id) = match fb.target {
        FeedbackTarget::Rollout(id) => ("rollout", id.to_string()),
        FeedbackTarget::Episode(id) => ("episode", id.to_string()),
    };
    d.set_item("target_type", target_type)?;
    d.set_item("target_id", target_id)?;

    d.set_item("label_kind", label_kind_str(fb.label_kind))?;
    d.set_item("metric_name", fb.metric_name.as_str())?;

    match &fb.value {
        FeedbackValue::Boolean { value } => {
            d.set_item("value_type", "boolean")?;
            d.set_item("value", *value)?;
        }
        FeedbackValue::Float { value } => {
            d.set_item("value_type", "float")?;
            d.set_item("value", *value)?;
        }
        FeedbackValue::FailureClass { class } => {
            d.set_item("value_type", "failure_class")?;
            d.set_item("failure_class", failure_class_str(*class))?;
        }
        FeedbackValue::DemonstrationRef {
            object_key,
            content_sha256,
        } => {
            d.set_item("value_type", "demonstration_ref")?;
            d.set_item("object_key", object_key.as_str())?;
            d.set_item("content_sha256", content_sha256.clone())?;
        }
    }

    d.set_item("join_method", join_method_str(fb.join_method))?;
    d.set_item("join_confidence", fb.join_confidence)?;
    d.set_item("join_version", fb.join_version.as_str())?;
    d.set_item("calibration_version", fb.calibration_version.as_str())?;
    d.set_item(
        "source_outcome_id",
        fb.source_outcome_id.map(|i| i.to_string()),
    )?;
    d.set_item("delay_ms", fb.delay_ms)?;
    d.set_item("retracted", fb.retracted)?;
    d.set_item("dedup_key", fb.dedup_key.as_str())?;
    d.set_item("outcome_ts_ns", fb.outcome_ts_ns)?;
    d.set_item("credit_weight", fb.credit_weight)?;
    d.set_item(
        "contributing_set_id",
        fb.contributing_set_id.map(|u| u.to_string()),
    )?;
    Ok(d)
}

/// Parse the `(target_type, target_id)` pair back into a [`FeedbackTarget`].
fn parse_feedback_target(d: &Bound<'_, PyDict>) -> PyResult<FeedbackTarget> {
    let target_id = req_str(d, "target_id")?;
    Ok(match req_str(d, "target_type")?.as_str() {
        "rollout" => FeedbackTarget::Rollout(parse_id(&target_id, "target_id")?),
        "episode" => FeedbackTarget::Episode(parse_id(&target_id, "target_id")?),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown target_type `{other}` (expected `rollout` or `episode`)"
            )));
        }
    })
}

/// Parse the `value_type` + payload back into a typed [`FeedbackValue`].
fn parse_feedback_value(d: &Bound<'_, PyDict>) -> PyResult<FeedbackValue> {
    Ok(match req_str(d, "value_type")?.as_str() {
        "boolean" => FeedbackValue::Boolean {
            value: req_bool(d, "value")?,
        },
        "float" => FeedbackValue::Float {
            value: req_f32(d, "value")?,
        },
        "failure_class" => FeedbackValue::FailureClass {
            class: parse_failure_class(&req_str(d, "failure_class")?)?,
        },
        "demonstration_ref" => FeedbackValue::DemonstrationRef {
            object_key: req_str(d, "object_key")?,
            content_sha256: opt_str(d, "content_sha256")?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown value_type `{other}`"
            )));
        }
    })
}

/// Reconstruct a [`Feedback`] from the flat dict [`feedback_to_dict`] produced.
///
/// This is the inverse marshaling that makes the attribute→curate pipeline a closed
/// loop: `curate` reads back exactly the feedback dicts `attribute` returned. Fields
/// absent from an older dict fall back to their schema defaults (`credit_weight` to full
/// credit, `retracted` to false), matching the Rust `#[serde(default)]`s.
pub(crate) fn dict_to_feedback(d: &Bound<'_, PyDict>) -> PyResult<Feedback> {
    let target = parse_feedback_target(d)?;
    let value = parse_feedback_value(d)?;
    let contributing_set_id = match opt_str(d, "contributing_set_id")? {
        Some(s) => Some(uuid::Uuid::from_str(&s).map_err(|e| {
            PyValueError::new_err(format!(
                "field `contributing_set_id` is not a valid UUID: {e}"
            ))
        })?),
        None => None,
    };
    let source_outcome_id = match opt_str(d, "source_outcome_id")? {
        Some(s) => Some(parse_id::<OutcomeId>(&s, "source_outcome_id")?),
        None => None,
    };
    Ok(Feedback {
        id: parse_id(&req_str(d, "feedback_id")?, "feedback_id")?,
        tenant_id: TenantId::new(req_str(d, "tenant_id")?),
        target,
        label_kind: parse_label_kind(&req_str(d, "label_kind")?)?,
        metric_name: req_str(d, "metric_name")?,
        value,
        join_method: parse_join_method(&req_str(d, "join_method")?)?,
        join_confidence: req_f32(d, "join_confidence")?,
        join_version: opt_str(d, "join_version")?.unwrap_or_default(),
        calibration_version: opt_str(d, "calibration_version")?.unwrap_or_default(),
        source_outcome_id,
        delay_ms: opt_i64_opt(d, "delay_ms")?,
        retracted: opt_bool(d, "retracted", false)?,
        dedup_key: opt_str(d, "dedup_key")?.unwrap_or_default(),
        outcome_ts_ns: opt_i64(d, "outcome_ts_ns", 0)?,
        credit_weight: opt_f32(d, "credit_weight")?.unwrap_or(1.0),
        contributing_set_id,
    })
}
