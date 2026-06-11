"""Closed-loop tests for the `fieldloop.attribute` binding.

These drive the real `fieldloop-join` engine through the Python boundary: build a
config + rollout/outcome dicts, call `attribute`, and assert the cascade lands the
right binding (explicit vs temporal), rejects a cross-tenant join, routes a heartbeat
to the coverage path, and surfaces caller errors as ValueError. The final test closes
the loop end-to-end: a rollout captured via `Capture.drain()` feeds straight into
`attribute` with no reshaping.
"""

import uuid

import fieldloop

# A minimal but complete attribution config: one embodiment with a wide teleop-takeover
# window (so a temporal binding is easy to land) plus the required calibration section.
CONFIG = """
[embodiments.warehouse_pick_arm]
action_space = "joint_position"

[embodiments.warehouse_pick_arm.attribution.teleop_takeover]
window_ms = 1500
requires_monotonic_colocation = false

[calibration]
version = "test-v1"
temporal_window_default_ms = 2000
heartbeat_coverage_k = 0.9
"""


def _uuid() -> str:
    return str(uuid.uuid4())


def _rollout(rollout_id: str, boot: str, episode: str, mono_ns: int) -> dict:
    """A plain rollout dict in the exact shape `attribute` (and `Capture.drain`) use."""
    return dict(
        rollout_id=rollout_id,
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=mono_ns,
        wall_ns=0,
        episode_id=episode,
        step_index=0,
        policy_version="pick@v1+abc123abc123",
        model_hash="sha256:weights",
        embodiment="warehouse_pick_arm",
        task_id="bin_pick",
        inference_us=900,
    )


def test_explicit_id_binds_at_confidence_one():
    """An outcome threading an explicit rollout id binds to it, method=explicit, 1.0."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    outcome = dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=1_200_000_000,
        outcome_kind="teleop_takeover",
        explicit_rollout_id=rid,
    )
    result = fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [outcome])

    assert len(result["feedbacks"]) == 1
    fb = result["feedbacks"][0]
    assert fb["join_method"] == "explicit"
    assert fb["join_confidence"] == 1.0
    assert fb["target_type"] == "rollout"
    assert fb["target_id"] == rid
    assert result["skipped"] == []


def test_temporal_binds_within_window_with_recency_confidence():
    """No explicit id: the latest in-window same-boot rollout binds, method=temporal."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    # 100ms after the rollout, well inside the 1500ms teleop window.
    outcome = dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=1_100_000_000,
        outcome_kind="teleop_takeover",
    )
    result = fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [outcome])

    assert len(result["feedbacks"]) == 1
    fb = result["feedbacks"][0]
    assert fb["join_method"] == "temporal"
    assert fb["target_id"] == rid
    # Recency ramp: a near-immediate outcome is highly but not fully confident.
    assert 0.0 < fb["join_confidence"] < 1.0
    assert result["skipped"] == []


def test_fitted_calibrator_threads_through_attribute():
    """A fitted calibrator built from feedback history can be passed to `attribute`, and
    the binding it produces carries the fitted calibrator's version tag (proving the
    fitted curve, not the identity default, was applied) — the OSS surface of the
    "confidence is calibrated once your labels exist" promise."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    rollout = _rollout(rid, boot, ep, 1_000_000_000)
    outcome = dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=1_100_000_000,
        outcome_kind="teleop_takeover",
    )

    # Without a calibrator: identity, so the row carries the config's calibration version.
    plain = fieldloop.attribute(CONFIG, [rollout], [outcome])
    assert plain["feedbacks"][0]["calibration_version"] == "test-v1"

    # Build enough confirmed temporal samples (each an inferred temporal binding agreeing
    # with a manual ground-truth label, same target) to learn the bucket, then fit.
    feedbacks = []
    rollouts = []
    for _ in range(6):
        r = _rollout(_uuid(), boot, _uuid(), 1_000_000_000)
        rollouts.append(r)
        for method in ("temporal", "manual"):
            feedbacks.append(
                dict(
                    feedback_id=_uuid(),
                    target_id=r["rollout_id"],
                    target_type="rollout",
                    tenant_id="acme",
                    label_kind="terminal_outcome",
                    metric_name="task_success",
                    value_type="boolean",
                    value=False,  # a failure, agreeing across both rows
                    join_method=method,
                    join_confidence=0.9 if method == "temporal" else 1.0,
                    join_version="join-v1",
                    calibration_version="test-v1",
                    dedup_key=f"dk:{r['rollout_id']}:{method}",
                    outcome_ts_ns=1,
                    credit_weight=1.0,
                )
            )
    cal = fieldloop.fit_calibrator(feedbacks, rollouts, min_labels=4)

    # With the fitted calibrator: the learned bucket applies, so the row is tagged with the
    # fitted calibrator's version, not the config default.
    fitted = fieldloop.attribute(CONFIG, [rollout], [outcome], calibrator=cal)
    assert len(fitted["feedbacks"]) == 1
    assert fitted["feedbacks"][0]["calibration_version"] == "fitted-v1"


def test_outcome_outside_window_does_not_bind():
    """An outcome past the window is recorded as a non-binding, never bound on a guess."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    # 3s after the rollout: outside the 1500ms teleop window.
    outcome = dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=4_000_000_000,
        outcome_kind="teleop_takeover",
    )
    result = fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [outcome])

    assert result["feedbacks"] == []
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "no_candidate_in_window"


def test_cross_tenant_outcome_is_a_hard_skip():
    """An outcome from a different tenant is rejected, never emitted as a binding."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    outcome = dict(
        tenant_id="evilcorp",  # different tenant than the rollout's "acme"
        robot_id="r7",
        boot_id=boot,
        mono_ns=1_100_000_000,
        outcome_kind="teleop_takeover",
    )
    result = fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [outcome])

    assert result["feedbacks"] == []
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "cross_tenant"


def test_heartbeat_is_routed_to_coverage_not_a_failure():
    """A heartbeat-kind dict rides the coverage path: it is never a failure binding and
    never appears as a skipped failure event."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    heartbeat = dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=boot,
        mono_ns=1_050_000_000,
        outcome_kind="heartbeat",
    )
    # With no heartbeat_period_ns, synthetic absence is disabled, so the heartbeat
    # produces neither a binding nor a per-outcome skip — it is pure coverage input.
    result = fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [heartbeat])
    assert result["feedbacks"] == []
    assert result["skipped"] == []


def test_invalid_config_raises_value_error():
    """A config that does not parse is a caller error surfaced as ValueError."""
    try:
        fieldloop.attribute("this is not valid toml = = =", [], [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid config")


def test_unknown_outcome_kind_raises_value_error():
    """An out-of-taxonomy outcome kind is rejected rather than silently dropped."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    bad = dict(
        tenant_id="acme", robot_id="r7", boot_id=boot, mono_ns=1, outcome_kind="exploded"
    )
    try:
        fieldloop.attribute(CONFIG, [_rollout(rid, boot, ep, 1_000_000_000)], [bad])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown outcome_kind")


def test_missing_required_field_raises_value_error():
    """A rollout dict missing a required key fails loudly before any engine work."""
    boot, ep, rid = _uuid(), _uuid(), _uuid()
    roll = _rollout(rid, boot, ep, 1_000_000_000)
    del roll["boot_id"]
    try:
        fieldloop.attribute(CONFIG, [roll], [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a missing required field")


def test_captured_rollout_feeds_attribute_directly():
    """The closed loop: capture a step, drain it, and attribute an outcome to it using
    the drained dict verbatim — no field reshaping between the two halves."""
    cap = fieldloop.Capture("acme", "r7", capacity=8)
    ctx = cap.register_context(
        policy_version="pick@v1+abc123abc123",
        model_hash="sha256:weights",
        embodiment="warehouse_pick_arm",
        task_id="bin_pick",
    )
    episode = _uuid()
    rid = cap.log_step(episode, step_index=0, ctx=ctx, inference_us=900)

    (row,) = cap.drain()
    assert row["rollout_id"] == rid
    # The drained dict already carries everything attribution needs.
    assert {"tenant_id", "robot_id", "boot_id", "mono_ns"} <= row.keys()

    # An explicit outcome on the same robot/boot binds to the captured rollout.
    outcome = dict(
        tenant_id=row["tenant_id"],
        robot_id=row["robot_id"],
        boot_id=row["boot_id"],
        mono_ns=row["mono_ns"] + 50_000_000,
        outcome_kind="teleop_takeover",
        explicit_rollout_id=row["rollout_id"],
    )
    result = fieldloop.attribute(CONFIG, [row], [outcome])

    assert len(result["feedbacks"]) == 1
    assert result["feedbacks"][0]["target_id"] == rid
    assert result["feedbacks"][0]["join_method"] == "explicit"
