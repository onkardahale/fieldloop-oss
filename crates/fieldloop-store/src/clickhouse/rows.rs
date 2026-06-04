//! Pure row serialization: a canonical schema value -> the `JSONEachRow` line
//! ClickHouse ingests.
//!
//! ClickHouse's `INSERT ... FORMAT JSONEachRow` consumes one JSON object per line,
//! with keys matching the table columns. These functions build that object as a
//! [`serde_json::Value`] (and [`to_line`] renders it to the single-line string the
//! HTTP insert path posts). They are pure — no database, no I/O — so the shape is
//! fully unit-testable.
//!
//! Two mappings matter and are done explicitly here rather than relying on the
//! types' own `derive(Serialize)`:
//!   * **id newtype -> the right column**: a [`fieldloop_types::Feedback`] carries a
//!     typed [`fieldloop_types::FeedbackTarget`]; storage projects it onto the flat
//!     `(target_id, target_type)` columns. The typed [`fieldloop_types::FeedbackValue`]
//!     sum type projects onto the `value_*` union columns.
//!   * **`UInt128` re-key columns**: the base tables emit the id as a `UUID` string,
//!     but a row destined for a `*ById`/`*ByTargetId` lookup needs the id as a
//!     decimal `UInt128`. [`uuid_to_uint128_string`] produces exactly the value a
//!     `toUInt128(<uuid>)` would, so the re-key column is correct on write.

use fieldloop_types::{Feedback, FeedbackValue, OutcomeEvent, Rollout};
use serde_json::{Map, Value, json};

/// Render a JSON row object to the single `JSONEachRow` line ClickHouse ingests.
///
/// `JSONEachRow` is one compact JSON object per line, so this is just compact
/// serialization. Returns the line WITHOUT a trailing newline; the insert path joins
/// rows with `\n`.
#[must_use]
pub fn to_line(row: &Value) -> String {
    // `to_string` on a serde_json Value is already compact (no pretty whitespace),
    // which is exactly the one-object-per-line form ClickHouse wants.
    row.to_string()
}

/// Encode a UUID as the decimal string a ClickHouse `toUInt128(<uuid>)` produces.
///
/// ClickHouse interprets a `UUID`'s 16 bytes as a big-endian 128-bit integer. We
/// reproduce that here so a re-key column (e.g. `id_uint`) holds the same value the
/// materialized view's `toUInt128(id)` would compute — keeping a UUIDv7 sort key in
/// chronological order even though ClickHouse stores `UUID` big-endian.
#[must_use]
pub fn uuid_to_uint128_string(id: uuid::Uuid) -> String {
    // The 16 bytes are already big-endian (most-significant first); fold them into a
    // u128 and print as a decimal string (JSON cannot carry a 128-bit integer
    // natively, so the value rides as a numeric string).
    let n = u128::from_be_bytes(*id.as_bytes());
    n.to_string()
}

/// Serialize a [`Rollout`] into its `Rollout` base-table `JSONEachRow` object.
///
/// Flattens the `(tenant_id, robot_id)` identity and the eval context onto flat
/// columns, maps the optional `trust` enum to a lowercase string (empty when unset,
/// since the pre-ingest record has none), and serializes the payload pointers as
/// JSON strings (the storage columns are `String`).
#[must_use]
pub fn rollout_row(r: &Rollout) -> Value {
    let (eval_run_id, arm_label) = match &r.eval {
        // An eval rollout carries the run id + opaque arm label.
        Some(e) => (e.eval_run_id.to_string(), e.arm_label.clone()),
        // Normal production: no eval window. Use the nil UUID / empty label as the
        // "absent" encoding for the non-nullable lookup columns.
        None => (uuid::Uuid::nil().to_string(), String::new()),
    };
    // The server-trusted anchor the ingest gateway stamps (trusted ingest time + the offset
    // that reconciles this boot's monotonic origin to server time). It was being set on the
    // Rollout at ingest but dropped here, so it never survived to storage; persist it. An
    // unstamped rollout (no anchor yet) serializes 0/0 — an honest "no server anchor", since a
    // real `ts_ingest_ns` is epoch-nanoseconds and never 0.
    let (server_ts_ingest_ns, server_anchor_offset_ns) = match &r.server_anchor {
        Some(a) => (a.ts_ingest_ns, a.server_anchor_offset_ns),
        None => (0, 0),
    };
    // The spatial co-location channel on the rollout side — the bind TARGET of the
    // spatial tier. Same nil-pose-with-`has_pose`-flag encoding as the outcome so a
    // non-localized rollout (the common case) is never mistaken for one at the origin.
    let (has_pose, px, py, pz, qw, qx, qy, qz) = match &r.pose {
        Some(p) => (1u8, p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz),
        None => (0u8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    };
    json!({
        "id": r.id.to_string(),
        "tenant_id": r.robot.tenant_id.as_str(),
        "robot_id": r.robot.robot_id.as_str(),
        "episode_id": r.episode_id.to_string(),
        "step_index": r.step_index,
        // The two-clock attribution spine, now persisted on the Rollout (it previously
        // survived only on OutcomeEvent, so a stored rollout had lost its monotonic timeline
        // and skew-free attribution was impossible on the rollout side). `boot_id` scopes
        // `mono_ns` to one robot boot; `mono_ns` is the attribution authority; `ts_wall_ns` is
        // the advisory wall estimate. Mirrors the OutcomeEvent serialization exactly.
        "boot_id": r.clock.boot_id.to_string(),
        "mono_ns": r.clock.mono_ns,
        "ts_wall_ns": r.clock.ts_wall_ns,
        // Server-trusted anchor (0/0 when unstamped) — now persisted, so a stored rollout keeps
        // the trusted-time reconciliation handle the gateway computed at ingest.
        "server_ts_ingest_ns": server_ts_ingest_ns,
        "server_anchor_offset_ns": server_anchor_offset_ns,
        "policy_version": r.policy_version.as_str(),
        "model_hash": r.model_hash,
        "trust": trust_str(r.trust),
        "embodiment": r.embodiment,
        "task_id": r.task_id,
        "eval_run_id": eval_run_id,
        "arm_label": arm_label,
        "observation_ref": serde_json::to_string(&r.observation_ref).unwrap_or_default(),
        "action_ref": serde_json::to_string(&r.action_ref).unwrap_or_default(),
        "context": serde_json::to_string(&r.context).unwrap_or_default(),
        "schema_conformance": enum_snake(&r.schema_conformance),
        "inference_us": r.inference_us,
        "tags": tags_to_json(&r.tags),
        // Real-vs-synthetic provenance, projected onto two flat columns so a query can
        // filter on `provenance_kind = 'real'` (and exclude synthetic from a safety set)
        // and audit the generator. An old row missing these columns reads back as real
        // via the type's `#[serde(default)]`.
        "provenance_kind": provenance_kind(&r.provenance),
        "provenance_generator": provenance_generator(&r.provenance),
        // Spatial co-location target columns + the causal station label. `has_pose`
        // distinguishes "no pose" from "origin pose"; `frame_id` scopes the coordinates
        // to a comparable origin; `station_id` is the upstream label a downstream
        // outcome's causal edge matches against. All default empty for a non-localized,
        // unstationed rollout.
        "has_pose": has_pose,
        "pose_x": px,
        "pose_y": py,
        "pose_z": pz,
        "pose_qw": qw,
        "pose_qx": qx,
        "pose_qy": qy,
        "pose_qz": qz,
        "frame_id": r.frame_id,
        "station_id": r.station_id,
        // Named scalar signal channels as a JSON string column, so the diagnostic
        // signal-deviation summary survives to storage rather than dying at the serializer.
        "signals": serde_json::to_string(&r.signals).unwrap_or_else(|_| "{}".to_string()),
    })
}

/// The `provenance_kind` storage string: `'real'` or `'synthetic'`. Lets a query
/// separate or exclude synthetic data without parsing a nested object.
fn provenance_kind(p: &fieldloop_types::Provenance) -> &'static str {
    match p {
        fieldloop_types::Provenance::Real => "real",
        fieldloop_types::Provenance::Synthetic { .. } => "synthetic",
    }
}

/// The synthetic generator string (e.g. `"cosmos-3"`), empty for real data — kept so
/// synthetic rows are auditable by their source generator.
fn provenance_generator(p: &fieldloop_types::Provenance) -> String {
    match p {
        fieldloop_types::Provenance::Real => String::new(),
        fieldloop_types::Provenance::Synthetic { generator } => generator.clone(),
    }
}

/// Serialize an [`OutcomeEvent`] into its `OutcomeEvent` base-table `JSONEachRow`
/// object. `explicit_rollout_id` maps to a nullable UUID column (`null` is the common
/// implicit case — no known target).
#[must_use]
pub fn outcome_row(o: &OutcomeEvent) -> Value {
    // The spatial co-location channel, projected onto flat columns. A `None` pose (the
    // common, non-localized case) serializes as the nil pose `0/0/0` + the identity
    // quaternion with `has_pose = 0`, so a reader can tell "no pose" from "pose at the
    // origin" without a nullable struct. `frame_id` empty mirrors "no frame declared".
    let (has_pose, px, py, pz, qw, qx, qy, qz) = match &o.pose {
        Some(p) => (1u8, p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz),
        None => (0u8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    };
    json!({
        "id": o.id.to_string(),
        "tenant_id": o.robot.tenant_id.as_str(),
        "robot_id": o.robot.robot_id.as_str(),
        "boot_id": o.clock.boot_id.to_string(),
        "mono_ns": o.clock.mono_ns,
        "ts_wall_ns": o.clock.ts_wall_ns,
        "outcome_kind": enum_snake(&o.outcome_kind),
        "explicit_rollout_id": o.explicit_rollout_id.map(|id| id.to_string()),
        "trust": trust_str(o.trust),
        "payload": serde_json::to_string(&o.payload).unwrap_or_default(),
        // Spatial co-location: `has_pose` distinguishes a real pose from the nil
        // default, `frame_id` scopes the coordinates to a comparable origin.
        "has_pose": has_pose,
        "pose_x": px,
        "pose_y": py,
        "pose_z": pz,
        "pose_qw": qw,
        "pose_qx": qx,
        "pose_qy": qy,
        "pose_qz": qz,
        "frame_id": o.frame_id,
        // Causal line topology: the downstream edges serialize as a JSON string (a small,
        // bounded list of station+lag hints), kept whole so the causal tier reconstructs
        // the exact edges. Empty `[]` for the common no-hint case.
        "causal_parents": serde_json::to_string(&o.causal_parents).unwrap_or_else(|_| "[]".to_string()),
    })
}

/// Serialize a [`Feedback`] into its `Feedback` base-table `JSONEachRow` object.
///
/// Projects the typed [`fieldloop_types::FeedbackTarget`] onto the flat
/// `(target_id, target_type)` columns (so the id and its grain can never disagree)
/// and the typed [`FeedbackValue`] onto the `value_*` union columns (so a categorical
/// failure class is never stuffed into the float column). `outcome_ts` is emitted as
/// fractional seconds since the epoch, matching the table's `DateTime64(9)` column.
#[must_use]
pub fn feedback_row(f: &Feedback) -> Value {
    let mut obj = Map::new();
    obj.insert("id".into(), json!(f.id.to_string()));
    obj.insert("tenant_id".into(), json!(f.tenant_id.as_str()));
    // Target: flat (target_id, target_type) projected from the indivisible sum type.
    obj.insert(
        "target_id".into(),
        json!(f.target.target_uuid().to_string()),
    );
    obj.insert("target_type".into(), json!(target_type_str(&f.target)));
    obj.insert("label_kind".into(), json!(enum_snake(&f.label_kind)));
    obj.insert("metric_name".into(), json!(f.metric_name));

    // Value union: exactly one of the value_* columns is meaningful per value_type.
    let (vt, vbool, vfloat, vfclass, vkey, vsha) = decompose_value(&f.value);
    obj.insert("value_type".into(), json!(vt));
    obj.insert("value_bool".into(), json!(vbool));
    obj.insert("value_float".into(), json!(vfloat));
    obj.insert("value_failure_class".into(), json!(vfclass));
    obj.insert("value_object_key".into(), json!(vkey));
    obj.insert("value_content_sha256".into(), json!(vsha));

    obj.insert("join_method".into(), json!(enum_snake(&f.join_method)));
    obj.insert("join_confidence".into(), json!(f.join_confidence));
    obj.insert("join_version".into(), json!(f.join_version));
    obj.insert("calibration_version".into(), json!(f.calibration_version));
    obj.insert(
        "source_outcome_id".into(),
        json!(f.source_outcome_id.map(|id| id.to_string())),
    );
    obj.insert("delay_ms".into(), json!(f.delay_ms));
    obj.insert("retracted".into(), json!(f.retracted));
    obj.insert("dedup_key".into(), json!(f.dedup_key));
    // DateTime64(9): a canonical decimal-seconds STRING with all nine fractional
    // digits (`<seconds>.<nanoseconds:09>`). A JSON float cannot carry this: an f64
    // mantissa loses sub-microsecond bits once the nanosecond count passes ~2^53, so
    // dividing by 1e9 would silently round real timestamps. The number form also fails
    // to parse — ClickHouse's DateTime64 reader consumes the integer part and then
    // rejects the trailing `.0`. A fixed 9-digit decimal string is exact and parses
    // unambiguously. `div_euclid`/`rem_euclid` keep the fraction non-negative.
    let secs = f.outcome_ts_ns.div_euclid(1_000_000_000);
    let frac = f.outcome_ts_ns.rem_euclid(1_000_000_000);
    obj.insert("outcome_ts".into(), json!(format!("{secs}.{frac:09}")));

    // Distributed-credit columns. `credit_weight` is this row's share of its outcome's
    // calibrated in-window confidence (1.0 for the single unambiguous binding).
    // `contributing_set_id` groups the rows of one multi-contributor split; the nil
    // UUID is the "no group" encoding for a single-contributor row, matching the
    // table's `DEFAULT '00000000-...'` so an old row predating the column reads back as
    // ungrouped.
    obj.insert("credit_weight".into(), json!(f.credit_weight));
    obj.insert(
        "contributing_set_id".into(),
        json!(
            f.contributing_set_id
                .unwrap_or_else(uuid::Uuid::nil)
                .to_string()
        ),
    );
    Value::Object(obj)
}

/// The decimal-`UInt128` re-key value for a feedback row's own id, for the
/// `FeedbackByTargetId.id_uint` lookup column. The lookup table sorts on the id, so
/// it must be the `toUInt128` form (a UUIDv7 stored as `UUID` would not sort
/// chronologically), and the recency tiebreak in the read relies on it.
#[must_use]
pub fn feedback_id_uint(f: &Feedback) -> String {
    uuid_to_uint128_string(f.id.as_uuid())
}

/// The decimal-`UInt128` re-key value for a rollout's own id, for the
/// `RolloutById.id_uint` lookup column.
#[must_use]
pub fn rollout_id_uint(r: &Rollout) -> String {
    uuid_to_uint128_string(r.id.as_uuid())
}

/// A parse failure turning a stored `JSONEachRow` object back into a typed struct.
///
/// The write-side attribution worker reads rows the cascade needs (rollouts, outcomes,
/// existing feedback) and must rebuild the typed values the pure engine consumes. The
/// stored row is a FLAT projection (the serializers above), not the struct's serde form,
/// so reconstruction is explicit here; this error names the column that was missing or
/// malformed so a bad row is auditable rather than a silent default.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RowParseError {
    /// A required column was absent from the row object.
    #[error("row is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// A column was present but the wrong JSON shape (e.g. a non-string where a UUID
    /// string was expected, or an enum string outside its closed set).
    #[error("column `{column}` had an unexpected value: {detail}")]
    BadValue {
        /// The column whose value could not be parsed.
        column: &'static str,
        /// Why it failed (the underlying parse error or a shape mismatch).
        detail: String,
    },
}

/// A required string column, or [`RowParseError::MissingColumn`].
fn str_col<'a>(row: &'a Value, col: &'static str) -> Result<&'a str, RowParseError> {
    row.get(col)
        .and_then(Value::as_str)
        .ok_or(RowParseError::MissingColumn(col))
}

/// A required integer column read as `i64`. ClickHouse `JSONEachRow` renders 64-bit
/// integers as JSON strings (an f64 cannot hold them exactly), so accept either a JSON
/// number or a numeric string — whichever the server emitted — and fail with the column
/// name if neither parses.
fn i64_col(row: &Value, col: &'static str) -> Result<i64, RowParseError> {
    let v = row.get(col).ok_or(RowParseError::MissingColumn(col))?;
    if let Some(n) = v.as_i64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        return s.parse::<i64>().map_err(|e| RowParseError::BadValue {
            column: col,
            detail: e.to_string(),
        });
    }
    Err(RowParseError::BadValue {
        column: col,
        detail: format!("not an integer: {v}"),
    })
}

/// A required `u64` column, with the same number-or-string acceptance as [`i64_col`]
/// (ClickHouse emits `UInt64` as a JSON string).
fn u64_col(row: &Value, col: &'static str) -> Result<u64, RowParseError> {
    let v = row.get(col).ok_or(RowParseError::MissingColumn(col))?;
    if let Some(n) = v.as_u64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        return s.parse::<u64>().map_err(|e| RowParseError::BadValue {
            column: col,
            detail: e.to_string(),
        });
    }
    Err(RowParseError::BadValue {
        column: col,
        detail: format!("not an unsigned integer: {v}"),
    })
}

/// Parse a UUID column into a `Uuid`, naming the column on failure.
fn uuid_col(row: &Value, col: &'static str) -> Result<uuid::Uuid, RowParseError> {
    let s = str_col(row, col)?;
    s.parse::<uuid::Uuid>()
        .map_err(|e| RowParseError::BadValue {
            column: col,
            detail: e.to_string(),
        })
}

/// Deserialize a closed enum from its `snake_case` storage string, via the type's own
/// serde form (the inverse of [`enum_snake`]), so an unknown value is rejected rather
/// than silently coerced.
fn enum_col<T: serde::de::DeserializeOwned>(
    row: &Value,
    col: &'static str,
) -> Result<T, RowParseError> {
    let s = str_col(row, col)?;
    serde_json::from_value(Value::String(s.to_string())).map_err(|e| RowParseError::BadValue {
        column: col,
        detail: e.to_string(),
    })
}

/// Reconstruct a [`Rollout`] from a [`crate::clickhouse::queries::recent_rollouts`] row.
///
/// The inverse of [`rollout_row`] over the columns that query selects. Only the fields
/// the attribution cascade and the calibrator actually read are reconstructed faithfully
/// — the identity, the monotonic clock spine (the attribution authority), the policy
/// version, the embodiment (the calibration bucket), and provenance — while
/// payload-pointer columns are not selected (the engine never reads sensor bytes) and so
/// are rebuilt as the empty/absent forms. A stored row therefore round-trips into exactly
/// the shape the pure engine needs, with no fabricated payload data.
///
/// # Errors
/// [`RowParseError`] naming the first missing or malformed column.
pub fn parse_rollout(row: &Value) -> Result<Rollout, RowParseError> {
    use fieldloop_types::{
        BootId, BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, Provenance, RobotId,
        RobotIdentity, RolloutId, Se3Pose, ServerAnchor, TenantId,
    };

    let robot = RobotIdentity::new(
        TenantId::new(str_col(row, "tenant_id")?),
        RobotId::new(str_col(row, "robot_id")?),
    );
    let clock = MonoClock::new(
        BootId::from_uuid(uuid_col(row, "boot_id")?),
        u64_col(row, "mono_ns")?,
        i64_col(row, "ts_wall_ns")?,
    );
    let mut r = Rollout::new(
        robot,
        EpisodeId::from_uuid(uuid_col(row, "episode_id")?),
        u32::try_from(i64_col(row, "step_index")?).map_err(|e| RowParseError::BadValue {
            column: "step_index",
            detail: e.to_string(),
        })?,
        clock,
        PolicyVersion::new(str_col(row, "policy_version")?),
        str_col(row, "model_hash")?.to_string(),
        str_col(row, "embodiment")?.to_string(),
        str_col(row, "task_id")?.to_string(),
        // Payload pointers are not part of attribution and are not selected by the read,
        // so they are rebuilt as the "absent" forms rather than invented.
        PayloadRef::none(),
        PayloadRef::none(),
        BoundedBlob::empty(),
        u32::try_from(i64_col(row, "inference_us")?).map_err(|e| RowParseError::BadValue {
            column: "inference_us",
            detail: e.to_string(),
        })?,
    );
    r.id = RolloutId::from_uuid(uuid_col(row, "id")?);
    // Provenance survives so the safety path can still exclude synthetic data after a
    // round-trip; the flat `provenance_kind`/`provenance_generator` columns rebuild the
    // tagged enum.
    r.provenance = match str_col(row, "provenance_kind")? {
        "synthetic" => Provenance::synthetic(str_col(row, "provenance_generator")?),
        _ => Provenance::Real,
    };
    // The server anchor reconciles this boot's monotonic origin to trusted server time —
    // the timeline the spatial and causal tiers compare cross-boot events on. A 0/0 row
    // (unstamped, or predating M0008) reads back as `None`, an honest "no anchor", since
    // a real `ts_ingest_ns` is epoch-nanoseconds and never 0.
    if let (Some(ts), Some(off)) = (
        row.get("server_ts_ingest_ns").and_then(Value::as_i64),
        row.get("server_anchor_offset_ns").and_then(Value::as_i64),
    ) && (ts != 0 || off != 0)
    {
        r.server_anchor = Some(ServerAnchor::new(ts, off));
    }
    // The spatial co-location pose (target side) — reconstructed only when `has_pose = 1`
    // so the nil default reads back as `None`, never a fabricated origin pose. The causal
    // `station_id` rebuilds the upstream label a downstream outcome's edge matches.
    let has_pose = row.get("has_pose").map_or(0, |v| v.as_u64().unwrap_or(0));
    if has_pose == 1 {
        r.pose = Some(Se3Pose::new(
            i_or_f_col(row, "pose_x")?,
            i_or_f_col(row, "pose_y")?,
            i_or_f_col(row, "pose_z")?,
            i_or_f_col(row, "pose_qw")?,
            i_or_f_col(row, "pose_qx")?,
            i_or_f_col(row, "pose_qy")?,
            i_or_f_col(row, "pose_qz")?,
        ));
        r.frame_id = row
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
    }
    r.station_id = row
        .get("station_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Signals round-trip through the JSON string column; an absent/blank/old column reads back
    // as the empty map (no signals reported), never an error.
    r.signals = row
        .get("signals")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    Ok(r)
}

/// Reconstruct an [`OutcomeEvent`] from a
/// [`crate::clickhouse::queries::recent_outcomes`] row — the inverse of [`outcome_row`].
///
/// Rebuilds the identity, the monotonic clock (the attribution authority), the outcome
/// kind, and any explicit rollout id (the nullable column reads back as `None` in the
/// common implicit case the cascade must infer). The inline payload is not selected and
/// is rebuilt empty, since attribution binds on time and kind, never on payload bytes.
///
/// # Errors
/// [`RowParseError`] naming the first missing or malformed column.
pub fn parse_outcome(row: &Value) -> Result<OutcomeEvent, RowParseError> {
    use fieldloop_types::{
        BootId, BoundedBlob, DownstreamEdge, MonoClock, OutcomeEvent, OutcomeId, OutcomeKind,
        RobotId, RobotIdentity, RolloutId, Se3Pose, TenantId,
    };

    let robot = RobotIdentity::new(
        TenantId::new(str_col(row, "tenant_id")?),
        RobotId::new(str_col(row, "robot_id")?),
    );
    let clock = MonoClock::new(
        BootId::from_uuid(uuid_col(row, "boot_id")?),
        u64_col(row, "mono_ns")?,
        i64_col(row, "ts_wall_ns")?,
    );
    let kind: OutcomeKind = enum_col(row, "outcome_kind")?;
    let mut o = OutcomeEvent::new(robot, clock, kind, BoundedBlob::empty());
    o.id = OutcomeId::from_uuid(uuid_col(row, "id")?);
    // `explicit_rollout_id` is a Nullable(UUID): a JSON null (or absent) is the common
    // implicit case the cascade must infer, so it maps to `None`; a present string is a
    // threaded explicit target.
    o.explicit_rollout_id = match row.get("explicit_rollout_id") {
        Some(Value::String(s)) => Some(RolloutId::from_uuid(s.parse().map_err(
            |e: uuid::Error| RowParseError::BadValue {
                column: "explicit_rollout_id",
                detail: e.to_string(),
            },
        )?)),
        _ => None,
    };
    // Spatial co-location: reconstruct the pose only when `has_pose = 1`, so the nil
    // default (0/0/0 + identity quaternion) is read back as `None` — "no pose" never
    // masquerades as a real pose at the origin. A row predating these columns (absent
    // `has_pose`) reads back as `None`, matching the type's `#[serde(default)]`.
    let has_pose = row.get("has_pose").map_or(0, |v| v.as_u64().unwrap_or(0));
    if has_pose == 1 {
        o.pose = Some(Se3Pose::new(
            i_or_f_col(row, "pose_x")?,
            i_or_f_col(row, "pose_y")?,
            i_or_f_col(row, "pose_z")?,
            i_or_f_col(row, "pose_qw")?,
            i_or_f_col(row, "pose_qx")?,
            i_or_f_col(row, "pose_qy")?,
            i_or_f_col(row, "pose_qz")?,
        ));
        o.frame_id = row
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
    }
    // Causal line topology: the edges ride as a JSON-string column; an absent or empty
    // string is the no-hint default. Parse failures surface as a named bad value rather
    // than silently dropping a hint.
    if let Some(Value::String(s)) = row.get("causal_parents")
        && !s.is_empty()
        && s != "[]"
    {
        o.causal_parents = serde_json::from_str::<Vec<DownstreamEdge>>(s).map_err(|e| {
            RowParseError::BadValue {
                column: "causal_parents",
                detail: e.to_string(),
            }
        })?;
    }
    Ok(o)
}

/// Reconstruct a [`Feedback`] from a [`crate::clickhouse::queries::recent_feedback`] row
/// — the inverse of [`feedback_row`], used to fit the calibrator from existing labels.
///
/// Rebuilds the indivisible `(target_id, target_type)` back into the typed
/// [`fieldloop_types::FeedbackTarget`] and the `value_*` union columns back into the
/// typed [`FeedbackValue`], so the calibrator sees the same shapes it was written from:
/// a `Manual` row's failure polarity is the ground truth, an automated row's
/// confidence/retraction the labeled sample. `outcome_ts_ns` is read from the
/// `toUnixTimestamp64Nano(outcome_ts)` projection the read selects, so the integer
/// nanoseconds are exact rather than re-derived from a lossy float.
///
/// # Errors
/// [`RowParseError`] naming the first missing or malformed column.
pub fn parse_feedback(row: &Value) -> Result<Feedback, RowParseError> {
    use fieldloop_types::{
        EpisodeId, FeedbackId, FeedbackTarget, JoinMethod, LabelKind, RolloutId, TenantId,
    };

    let target_id = uuid_col(row, "target_id")?;
    let target = match str_col(row, "target_type")? {
        "rollout" => FeedbackTarget::Rollout(RolloutId::from_uuid(target_id)),
        "episode" => FeedbackTarget::Episode(EpisodeId::from_uuid(target_id)),
        other => {
            return Err(RowParseError::BadValue {
                column: "target_type",
                detail: format!("unknown target_type `{other}`"),
            });
        }
    };
    let method: JoinMethod = enum_col(row, "join_method")?;
    let label_kind: LabelKind = enum_col(row, "label_kind")?;
    let value = parse_feedback_value(row)?;
    let source_outcome_id = match row.get("source_outcome_id") {
        Some(Value::String(s)) => Some(fieldloop_types::OutcomeId::from_uuid(s.parse().map_err(
            |e: uuid::Error| RowParseError::BadValue {
                column: "source_outcome_id",
                detail: e.to_string(),
            },
        )?)),
        _ => None,
    };
    let delay_ms = match row.get("delay_ms") {
        Some(v) if !v.is_null() => Some(i64_col(row, "delay_ms")?),
        _ => None,
    };
    let contributing_set_id = match row.get("contributing_set_id") {
        Some(Value::String(s)) if s != "00000000-0000-0000-0000-000000000000" => Some(
            s.parse::<uuid::Uuid>()
                .map_err(|e| RowParseError::BadValue {
                    column: "contributing_set_id",
                    detail: e.to_string(),
                })?,
        ),
        _ => None,
    };

    Ok(Feedback {
        id: FeedbackId::from_uuid(uuid_col(row, "id")?),
        tenant_id: TenantId::new(str_col(row, "tenant_id")?),
        target,
        label_kind,
        metric_name: str_col(row, "metric_name")?.to_string(),
        value,
        join_method: method,
        join_confidence: i_or_f_col(row, "join_confidence")? as f32,
        join_version: str_col(row, "join_version")?.to_string(),
        calibration_version: str_col(row, "calibration_version")?.to_string(),
        source_outcome_id,
        delay_ms,
        retracted: bool_col(row, "retracted")?,
        dedup_key: str_col(row, "dedup_key")?.to_string(),
        outcome_ts_ns: i64_col(row, "outcome_ts_ns")?,
        credit_weight: i_or_f_col(row, "credit_weight")? as f32,
        contributing_set_id,
    })
}

/// Rebuild the typed [`FeedbackValue`] from the stored `value_*` union columns, the
/// inverse of [`decompose_value`]: exactly the column the `value_type` names is read, so
/// a failure class is never resurrected from the float column.
fn parse_feedback_value(row: &Value) -> Result<FeedbackValue, RowParseError> {
    match str_col(row, "value_type")? {
        "boolean" => Ok(FeedbackValue::Boolean {
            value: bool_col(row, "value_bool")?,
        }),
        "float" => Ok(FeedbackValue::Float {
            value: i_or_f_col(row, "value_float")? as f32,
        }),
        "failure_class" => Ok(FeedbackValue::FailureClass {
            class: enum_col(row, "value_failure_class")?,
        }),
        "demonstration_ref" => Ok(FeedbackValue::DemonstrationRef {
            object_key: str_col(row, "value_object_key")?.to_string(),
            content_sha256: row
                .get("value_content_sha256")
                .and_then(Value::as_str)
                .map(str::to_string),
        }),
        other => Err(RowParseError::BadValue {
            column: "value_type",
            detail: format!("unknown value_type `{other}`"),
        }),
    }
}

/// A required boolean column. ClickHouse `Bool` comes back as a JSON bool, but accept a
/// 0/1 number too (the HTTP interface can render `Bool` either way depending on settings).
fn bool_col(row: &Value, col: &'static str) -> Result<bool, RowParseError> {
    let v = row.get(col).ok_or(RowParseError::MissingColumn(col))?;
    if let Some(b) = v.as_bool() {
        return Ok(b);
    }
    if let Some(n) = v.as_i64() {
        return Ok(n != 0);
    }
    Err(RowParseError::BadValue {
        column: col,
        detail: format!("not a boolean: {v}"),
    })
}

/// A required numeric column read as `f64`, accepting a JSON number or a numeric string.
/// Used for the `Float32` columns (confidence, value_float, credit_weight) the server may
/// render as a bare number.
fn i_or_f_col(row: &Value, col: &'static str) -> Result<f64, RowParseError> {
    let v = row.get(col).ok_or(RowParseError::MissingColumn(col))?;
    if let Some(f) = v.as_f64() {
        return Ok(f);
    }
    if let Some(s) = v.as_str() {
        return s.parse::<f64>().map_err(|e| RowParseError::BadValue {
            column: col,
            detail: e.to_string(),
        });
    }
    Err(RowParseError::BadValue {
        column: col,
        detail: format!("not a number: {v}"),
    })
}

// ---- helpers ---------------------------------------------------------------

/// Map the optional `Trust` enum onto its storage string. Empty string is the
/// "unset" encoding for the pre-ingest record (the column is non-nullable
/// `LowCardinality(String)` so the gateway can fill it later).
fn trust_str(t: Option<fieldloop_types::Trust>) -> &'static str {
    match t {
        Some(fieldloop_types::Trust::Trusted) => "trusted",
        Some(fieldloop_types::Trust::Untrusted) => "untrusted",
        None => "",
    }
}

/// The `target_type` storage string ('rollout' / 'episode') from the typed target.
fn target_type_str(t: &fieldloop_types::FeedbackTarget) -> &'static str {
    match t {
        fieldloop_types::FeedbackTarget::Rollout(_) => "rollout",
        fieldloop_types::FeedbackTarget::Episode(_) => "episode",
    }
}

/// Decompose the typed [`FeedbackValue`] into the storage `value_*` union columns.
/// Exactly the column for the active variant is `Some`; the rest are `None`/empty.
#[allow(clippy::type_complexity)]
fn decompose_value(
    v: &FeedbackValue,
) -> (
    &'static str,
    Option<bool>,
    Option<f32>,
    String,
    String,
    Option<String>,
) {
    // Start with every column empty, then fill only the one this variant uses, so the
    // typed value's shape is preserved on the wire (a failure class never lands in the
    // float column, etc.).
    let vt;
    let mut vbool = None;
    let mut vfloat = None;
    let mut vfclass = String::new();
    let mut vkey = String::new();
    let mut vsha = None;
    match v {
        FeedbackValue::Boolean { value } => {
            vt = "boolean";
            vbool = Some(*value);
        }
        FeedbackValue::Float { value } => {
            vt = "float";
            vfloat = Some(*value);
        }
        FeedbackValue::FailureClass { class } => {
            vt = "failure_class";
            vfclass = format!("{class:?}").to_lowercase();
        }
        FeedbackValue::DemonstrationRef {
            object_key,
            content_sha256,
        } => {
            vt = "demonstration_ref";
            vkey = object_key.clone();
            vsha = content_sha256.clone();
        }
    }
    (vt, vbool, vfloat, vfclass, vkey, vsha)
}

/// Render a `BTreeMap<String,String>` as a JSON object for a ClickHouse
/// `Map(String,String)` column.
fn tags_to_json(tags: &std::collections::BTreeMap<String, String>) -> Value {
    let mut m = Map::new();
    for (k, val) in tags {
        m.insert(k.clone(), json!(val));
    }
    Value::Object(m)
}

/// A serializable closed enum -> its `snake_case` storage string, going through the
/// type's own serde form (these enums use `#[serde(rename_all = "snake_case")]`), so
/// the storage string always matches the canonical wire name and a `Debug`-vs-serde
/// drift is impossible.
fn enum_snake<T: serde::Serialize>(e: &T) -> String {
    // The enums serialize to a bare JSON string; pull out that string.
    serde_json::to_value(e)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldloop_types::{
        BootId, BoundedBlob, DownstreamEdge, EpisodeId, Feedback, FeedbackId, FeedbackTarget,
        FeedbackValue, JoinMethod, LabelKind, MonoClock, OutcomeEvent, OutcomeId, OutcomeKind,
        PayloadRef, PolicyVersion, RobotId, RobotIdentity, Rollout, RolloutId, Se3Pose, TenantId,
    };

    /// `toUInt128` reproduction is correct: the all-zero UUID is 0, and a known
    /// big-endian byte pattern folds to the expected integer.
    #[test]
    fn uuid_to_uint128_matches_big_endian() {
        assert_eq!(uuid_to_uint128_string(uuid::Uuid::nil()), "0");
        // 0x...01 in the least-significant byte => 1.
        let mut bytes = [0u8; 16];
        bytes[15] = 1;
        assert_eq!(uuid_to_uint128_string(uuid::Uuid::from_bytes(bytes)), "1");
    }

    fn sample_rollout() -> Rollout {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 42,
            ts_wall_ns: 1_700_000_000_000_000_000,
        };
        Rollout::new(
            robot,
            EpisodeId::new(),
            7,
            clock,
            PolicyVersion::new("pol@v1+abc123def456"),
            "sha256:weights".into(),
            "arm6dof".into(),
            "pick_place".into(),
            PayloadRef::none(),
            PayloadRef::none(),
            BoundedBlob::empty(),
            1234,
        )
    }

    /// A sample Rollout serializes with the flat identity columns, the policy
    /// version, and the open `tags` map — the expected `JSONEachRow` shape.
    #[test]
    fn rollout_row_has_expected_shape() {
        let r = sample_rollout();
        let v = rollout_row(&r);
        assert_eq!(v["tenant_id"], json!("acme"));
        assert_eq!(v["robot_id"], json!("r1"));
        assert_eq!(v["step_index"], json!(7));
        assert_eq!(v["policy_version"], json!("pol@v1+abc123def456"));
        assert_eq!(v["id"], json!(r.id.to_string()));
        assert!(v["tags"].is_object());
        // The clock spine is now persisted (was dropped before), so a stored rollout keeps its
        // attribution authority: boot_id scopes the monotonic timeline, mono_ns is the value.
        assert_eq!(v["boot_id"], json!(r.clock.boot_id.to_string()));
        assert_eq!(v["mono_ns"], json!(42));
        assert_eq!(v["ts_wall_ns"], json!(1_700_000_000_000_000_000i64));
        // No server anchor stamped yet => 0/0 (an honest "absent", never a real epoch time).
        assert_eq!(v["server_ts_ingest_ns"], json!(0));
        assert_eq!(v["server_anchor_offset_ns"], json!(0));
        // No eval window => nil eval_run_id and empty arm_label.
        assert_eq!(v["arm_label"], json!(""));

        // The UInt128 re-key column is the toUInt128 form of the id.
        assert_eq!(rollout_id_uint(&r), uuid_to_uint128_string(r.id.as_uuid()));
        // ...and round-trips back to the same id via from_be_bytes.
        let n: u128 = rollout_id_uint(&r).parse().unwrap();
        assert_eq!(uuid::Uuid::from_bytes(n.to_be_bytes()), r.id.as_uuid());
    }

    /// A rollout the gateway stamped with a server anchor serializes its trusted ingest time and
    /// reconciling offset, so the anchor survives to storage instead of being dropped.
    #[test]
    fn stamped_rollout_persists_server_anchor() {
        let mut r = sample_rollout();
        r.server_anchor = Some(fieldloop_types::ServerAnchor::new(
            1_700_000_111_222_333_444,
            -55,
        ));
        let v = rollout_row(&r);
        assert_eq!(
            v["server_ts_ingest_ns"],
            json!(1_700_000_111_222_333_444i64)
        );
        assert_eq!(v["server_anchor_offset_ns"], json!(-55));
    }

    /// A float-valued Feedback projects onto the right `value_*` union columns: the
    /// float is populated, the others are null/empty, and `target_type` matches the
    /// typed grain.
    #[test]
    fn feedback_row_float_value_shape() {
        let target = RolloutId::new();
        let f = Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "reward".into(),
            value: FeedbackValue::Float { value: 0.75 },
            join_method: JoinMethod::Temporal,
            join_confidence: 0.9,
            join_version: "j1".into(),
            calibration_version: "c1".into(),
            source_outcome_id: Some(OutcomeId::new()),
            delay_ms: Some(120),
            retracted: false,
            dedup_key: "dk-1".into(),
            outcome_ts_ns: 1_700_000_000_000_000_000,
            credit_weight: 1.0,
            contributing_set_id: None,
        };
        let v = feedback_row(&f);
        assert_eq!(v["tenant_id"], json!("acme"));
        assert_eq!(v["target_id"], json!(target.to_string()));
        assert_eq!(v["target_type"], json!("rollout"));
        assert_eq!(v["label_kind"], json!("terminal_outcome"));
        assert_eq!(v["value_type"], json!("float"));
        assert_eq!(v["value_float"], json!(0.75_f32));
        assert_eq!(v["value_bool"], Value::Null);
        assert_eq!(v["join_method"], json!("temporal"));
        assert_eq!(v["dedup_key"], json!("dk-1"));
        // outcome_ts emitted as a canonical decimal-seconds string (9 fractional
        // digits) for DateTime64(9) — exact and unambiguous to parse.
        assert_eq!(v["outcome_ts"], json!("1700000000.000000000"));

        // A single full-credit binding: weight 1.0, nil group id (the "no set" form).
        assert_eq!(v["credit_weight"], json!(1.0_f32));
        assert_eq!(
            v["contributing_set_id"],
            json!("00000000-0000-0000-0000-000000000000")
        );

        // The id_uint re-key column matches toUInt128(id).
        assert_eq!(feedback_id_uint(&f), uuid_to_uint128_string(f.id.as_uuid()));
    }

    /// A distributed-credit row serializes its partial `credit_weight` and its real
    /// (non-nil) `contributing_set_id` onto the flat columns, so a co-contributor split
    /// survives to storage and a reader can re-group the rows of one set and re-sum
    /// their weights. The nil-UUID "no group" encoding is reserved for single bindings.
    #[test]
    fn feedback_row_carries_distributed_credit() {
        let set_id = uuid::Uuid::now_v7();
        let f = Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(RolloutId::new()),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "teleop_takeover".into(),
            value: FeedbackValue::Boolean { value: false },
            join_method: JoinMethod::Temporal,
            join_confidence: 0.8,
            join_version: "join-v2".into(),
            calibration_version: "c1".into(),
            source_outcome_id: Some(OutcomeId::new()),
            delay_ms: Some(200),
            retracted: false,
            dedup_key: "dk-dc".into(),
            outcome_ts_ns: 1_700_000_000_000_000_000,
            credit_weight: 0.625,
            contributing_set_id: Some(set_id),
        };
        let v = feedback_row(&f);
        assert_eq!(v["credit_weight"], json!(0.625_f32));
        assert_eq!(v["contributing_set_id"], json!(set_id.to_string()));
        // It is NOT the nil-UUID "no group" form — this row really is in a set.
        assert_ne!(
            v["contributing_set_id"],
            json!("00000000-0000-0000-0000-000000000000")
        );
    }

    /// A real rollout serializes `provenance_kind = "real"` with an empty generator;
    /// a synthetic one carries `"synthetic"` + the generator name, so a query can
    /// separate or exclude synthetic data and audit its source.
    #[test]
    fn rollout_row_carries_provenance() {
        let real = rollout_row(&sample_rollout());
        assert_eq!(real["provenance_kind"], json!("real"));
        assert_eq!(real["provenance_generator"], json!(""));

        let syn =
            sample_rollout().with_provenance(fieldloop_types::Provenance::synthetic("cosmos-3"));
        let v = rollout_row(&syn);
        assert_eq!(v["provenance_kind"], json!("synthetic"));
        assert_eq!(v["provenance_generator"], json!("cosmos-3"));
    }

    /// An old stored row whose JSON predates the provenance field deserializes back
    /// into a `Rollout` as `Real` — confirming the storage append is backward
    /// compatible with rows written before this field existed.
    #[test]
    fn old_row_without_provenance_reads_as_real() {
        let mut json = serde_json::to_value(sample_rollout()).unwrap();
        json.as_object_mut().unwrap().remove("provenance");
        let back: Rollout = serde_json::from_value(json).unwrap();
        assert!(back.provenance.is_real());
    }

    /// A failure-class Feedback uses the categorical column, not the float column —
    /// the typed value's shape is preserved across serialization.
    #[test]
    fn feedback_row_failure_class_uses_categorical_column() {
        let f = Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Episode(EpisodeId::new()),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "failure".into(),
            value: FeedbackValue::FailureClass {
                class: fieldloop_types::FailureClass::Perception,
            },
            join_method: JoinMethod::Manual,
            join_confidence: 1.0,
            join_version: "j1".into(),
            calibration_version: "c1".into(),
            source_outcome_id: None,
            delay_ms: None,
            retracted: false,
            dedup_key: "dk-2".into(),
            outcome_ts_ns: 0,
            credit_weight: 1.0,
            contributing_set_id: None,
        };
        let v = feedback_row(&f);
        assert_eq!(v["value_type"], json!("failure_class"));
        assert_eq!(v["value_failure_class"], json!("perception"));
        assert_eq!(v["value_float"], Value::Null);
        assert_eq!(v["target_type"], json!("episode"));
        // SyntheticAbsence-style: no source outcome => null.
        assert_eq!(v["source_outcome_id"], Value::Null);
    }

    /// An OutcomeEvent with no explicit target serializes the nullable column as
    /// `null` — the common implicit case the attribution cascade must infer.
    #[test]
    fn outcome_row_implicit_target_is_null() {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 9,
            ts_wall_ns: 1,
        };
        let o = OutcomeEvent::new(
            robot,
            clock,
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        );
        let v = outcome_row(&o);
        assert_eq!(v["explicit_rollout_id"], Value::Null);
        assert_eq!(v["outcome_kind"], json!("teleop_takeover"));
        assert_eq!(v["tenant_id"], json!("acme"));
    }

    /// A rollout serialized to its storage row and parsed back reconstructs the fields
    /// the attribution cascade and calibrator read: identity, the monotonic clock spine
    /// (the attribution authority), policy version, and the embodiment calibration
    /// bucket. This is the write-side read path's inverse, proven without a database — the
    /// same string/JSON-level guarantee the query tests give the SQL.
    #[test]
    fn rollout_row_round_trips_through_parse() {
        let r = sample_rollout();
        let back = parse_rollout(&rollout_row(&r)).expect("parse rollout");
        assert_eq!(back.id, r.id);
        assert_eq!(back.robot, r.robot);
        assert_eq!(back.episode_id, r.episode_id);
        assert_eq!(back.step_index, r.step_index);
        // The clock spine survives exactly — boot scopes the monotonic timeline, mono_ns
        // is the value the engine attributes on.
        assert_eq!(back.clock, r.clock);
        assert_eq!(back.embodiment, r.embodiment);
        assert_eq!(back.policy_version, r.policy_version);
    }

    /// A synthetic rollout's provenance survives the row round-trip, so the safety path
    /// can still exclude synthetic data after a read-back (silence would otherwise read
    /// as real — provenance must be preserved, not defaulted).
    #[test]
    fn rollout_parse_preserves_synthetic_provenance() {
        let r =
            sample_rollout().with_provenance(fieldloop_types::Provenance::synthetic("cosmos-3"));
        let back = parse_rollout(&rollout_row(&r)).expect("parse synthetic rollout");
        assert!(back.provenance.is_synthetic());
        assert_eq!(
            back.provenance,
            fieldloop_types::Provenance::synthetic("cosmos-3")
        );
    }

    /// The rollout's spatial pose, frame, station, AND server anchor survive the row
    /// round-trip, so the spatial (co-location) and causal (upstream-station, on the
    /// server-anchored clock) tiers see the same target inputs after a read-back. A
    /// rollout WITHOUT a pose reads back as `None`, never a fabricated origin pose.
    #[test]
    fn rollout_parse_preserves_pose_station_and_anchor() {
        let r = sample_rollout()
            .with_pose(Se3Pose::at(3.0, 4.0, 0.0), "map")
            .at_station("pick_cell");
        let r = {
            let mut r = r;
            r.server_anchor = Some(fieldloop_types::ServerAnchor::new(
                1_700_000_000_000_000_000,
                42,
            ));
            r
        };
        let back = parse_rollout(&rollout_row(&r)).expect("parse rollout");
        assert_eq!(back.pose, Some(Se3Pose::at(3.0, 4.0, 0.0)));
        assert_eq!(back.frame_id, "map");
        assert_eq!(back.station_id, "pick_cell");
        assert_eq!(back.server_anchor, r.server_anchor);

        // A non-localized, unstationed rollout reads back with no pose / no station.
        let plain = parse_rollout(&rollout_row(&sample_rollout())).expect("parse plain rollout");
        assert!(plain.pose.is_none());
        assert!(plain.station_id.is_empty());
    }

    /// An outcome round-trips through its storage row: the clock and kind survive, and an
    /// implicit (no explicit target) outcome reads back as `None` — the common case the
    /// cascade must infer, which must never be confused with a real id.
    #[test]
    fn outcome_row_round_trips_with_implicit_target() {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 99,
            ts_wall_ns: 1_700_000_000_000_000_000,
        };
        let o = OutcomeEvent::new(robot, clock, OutcomeKind::Collision, BoundedBlob::empty());
        let back = parse_outcome(&outcome_row(&o)).expect("parse outcome");
        assert_eq!(back.id, o.id);
        assert_eq!(back.robot, o.robot);
        assert_eq!(back.clock, o.clock);
        assert_eq!(back.outcome_kind, OutcomeKind::Collision);
        assert_eq!(back.explicit_rollout_id, None);
    }

    /// An outcome carrying an explicit rollout id round-trips that id, so the cascade's
    /// Tier-1 explicit binding still fires after a read-back.
    #[test]
    fn outcome_row_round_trips_explicit_target() {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 5,
            ts_wall_ns: 1,
        };
        let explicit = RolloutId::new();
        let mut o = OutcomeEvent::new(robot, clock, OutcomeKind::EStop, BoundedBlob::empty());
        o.explicit_rollout_id = Some(explicit);
        let back = parse_outcome(&outcome_row(&o)).expect("parse outcome");
        assert_eq!(back.explicit_rollout_id, Some(explicit));
    }

    /// The spatial pose, its frame, and the causal edges survive the outcome row
    /// round-trip, so the spatial and causal tiers see the same inputs after a read-back.
    /// A station-specific lag override is preserved exactly.
    #[test]
    fn outcome_row_round_trips_pose_and_causal_parents() {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 7,
            ts_wall_ns: 1,
        };
        let o = OutcomeEvent::new(
            robot,
            clock,
            OutcomeKind::DownstreamFailure,
            BoundedBlob::empty(),
        )
        .with_pose(Se3Pose::new(1.5, -2.0, 0.25, 0.5, 0.5, 0.5, 0.5), "map")
        .with_causal_parents(vec![
            DownstreamEdge::new("inspection"),
            DownstreamEdge::with_lag("pick_cell", 4_000),
        ]);
        let back = parse_outcome(&outcome_row(&o)).expect("parse outcome");
        assert_eq!(back.pose, o.pose);
        assert_eq!(back.frame_id, "map");
        assert_eq!(back.causal_parents.len(), 2);
        assert_eq!(back.causal_parents[0].upstream_station_id, "inspection");
        assert_eq!(back.causal_parents[1].max_lag_ms, Some(4_000));
    }

    /// An outcome with NO pose reads back as `None` (not a fabricated origin pose) and
    /// with NO causal parents — the nil/empty defaults the `has_pose = 0` flag and the
    /// empty-list JSON encode. This is the "no pose vs origin pose" distinction the
    /// `has_pose` flag exists to preserve.
    #[test]
    fn outcome_row_without_pose_reads_back_as_none() {
        let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
        let clock = MonoClock {
            boot_id: BootId::new(),
            mono_ns: 1,
            ts_wall_ns: 1,
        };
        let o = OutcomeEvent::new(robot, clock, OutcomeKind::Collision, BoundedBlob::empty());
        let row = outcome_row(&o);
        // The serialized row carries the nil pose at the origin, but flagged has_pose = 0.
        assert_eq!(row.get("has_pose").and_then(|v| v.as_u64()), Some(0));
        let back = parse_outcome(&row).expect("parse outcome");
        assert!(
            back.pose.is_none(),
            "an outcome with no pose must read back as None, not an origin pose"
        );
        assert!(back.frame_id.is_empty());
        assert!(back.causal_parents.is_empty());
    }

    /// A feedback row round-trips: the typed value union, the target grain, the join
    /// method, the retraction flag, the exact-nanosecond outcome stamp, and the
    /// distributed-credit columns all survive — the calibrator fits from exactly these.
    /// The read selects `toUnixTimestamp64Nano(outcome_ts) AS outcome_ts_ns`, so the test
    /// row carries that column name (mirroring the live read) rather than the float form.
    #[test]
    fn feedback_row_round_trips_through_parse() {
        let target = RolloutId::new();
        let f = Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Rollout(target),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "task_success".into(),
            value: FeedbackValue::Boolean { value: false },
            join_method: JoinMethod::Temporal,
            join_confidence: 0.8,
            join_version: "join-v2".into(),
            calibration_version: "fitted-v1".into(),
            source_outcome_id: Some(OutcomeId::new()),
            delay_ms: Some(120),
            retracted: true,
            dedup_key: "dk-rt".into(),
            outcome_ts_ns: 1_700_000_000_123_456_789,
            credit_weight: 1.0,
            contributing_set_id: None,
        };
        // The live read projects the nanosecond column under `outcome_ts_ns`; the
        // serializer writes the float `outcome_ts`. Build the row the parser will see by
        // taking the serialized row and adding the projected integer column, exactly as
        // the SELECT returns it.
        let mut row = feedback_row(&f);
        row.as_object_mut()
            .unwrap()
            .insert("outcome_ts_ns".into(), json!(f.outcome_ts_ns.to_string()));
        let back = parse_feedback(&row).expect("parse feedback");
        assert_eq!(back.id, f.id);
        assert_eq!(back.target, f.target);
        assert_eq!(back.join_method, JoinMethod::Temporal);
        assert_eq!(back.value, FeedbackValue::Boolean { value: false });
        assert!(back.retracted);
        assert_eq!(back.outcome_ts_ns, f.outcome_ts_ns);
        assert!((back.join_confidence - 0.8).abs() < 1e-6);
    }

    /// A failure-class feedback round-trips through the categorical column, never the
    /// float column — the calibrator must see the same failure polarity it was written
    /// with, so a manual ground-truth label keeps its meaning across a read-back.
    #[test]
    fn feedback_parse_preserves_failure_class() {
        let f = Feedback {
            id: FeedbackId::new(),
            tenant_id: TenantId::new("acme"),
            target: FeedbackTarget::Episode(EpisodeId::new()),
            label_kind: LabelKind::TerminalOutcome,
            metric_name: "failure".into(),
            value: FeedbackValue::FailureClass {
                class: fieldloop_types::FailureClass::Perception,
            },
            join_method: JoinMethod::Manual,
            join_confidence: 1.0,
            join_version: "j1".into(),
            calibration_version: "c1".into(),
            source_outcome_id: None,
            delay_ms: None,
            retracted: false,
            dedup_key: "dk-fc".into(),
            outcome_ts_ns: 7,
            credit_weight: 1.0,
            contributing_set_id: None,
        };
        let mut row = feedback_row(&f);
        row.as_object_mut()
            .unwrap()
            .insert("outcome_ts_ns".into(), json!("7"));
        let back = parse_feedback(&row).expect("parse failure-class feedback");
        assert_eq!(
            back.value,
            FeedbackValue::FailureClass {
                class: fieldloop_types::FailureClass::Perception
            }
        );
        assert_eq!(
            back.target,
            FeedbackTarget::Episode(match f.target {
                FeedbackTarget::Episode(id) => id,
                _ => unreachable!(),
            })
        );
    }

    /// A missing required column is a NAMED parse error, not a silent default — a
    /// malformed stored row is auditable rather than feeding the cascade garbage.
    #[test]
    fn parse_rollout_names_a_missing_column() {
        let mut row = rollout_row(&sample_rollout());
        row.as_object_mut().unwrap().remove("embodiment");
        let err = parse_rollout(&row).unwrap_err();
        assert_eq!(err, RowParseError::MissingColumn("embodiment"));
    }

    /// `to_line` is compact single-line JSON (no pretty whitespace), matching
    /// ClickHouse's one-object-per-line `JSONEachRow` format.
    #[test]
    fn to_line_is_compact() {
        let line = to_line(&json!({"a": 1, "b": "x"}));
        assert!(!line.contains('\n'));
        assert_eq!(line, r#"{"a":1,"b":"x"}"#);
    }
}
