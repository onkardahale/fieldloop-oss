"""Closed-loop tests for the `fieldloop.curate` binding.

These drive the real `fieldloop-curation` compiler through Python: build rollouts +
feedback dicts (the exact shape `attribute` emits), call `curate`, and assert the
confidence gate pins the trusted bindings and holds out the doubtful ones (low
confidence / retracted / no binding) under `needs_review`. The last test closes the
loop: `attribute(...)["feedbacks"]` pipes straight into `curate`.
"""

import uuid

import fieldloop

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


def _rollout(rollout_id: str, episode: str, mono_ns: int, **extra) -> dict:
    boot = extra.pop("boot", BOOT)
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
        **extra,
    )


def _feedback(rollout_id: str, *, confidence: float = 1.0, retracted: bool = False) -> dict:
    """A feedback dict in the exact shape `attribute` emits / `curate` consumes."""
    return dict(
        feedback_id=_uuid(),
        tenant_id="acme",
        target_type="rollout",
        target_id=rollout_id,
        label_kind="terminal_outcome",
        metric_name="task_success",
        value_type="boolean",
        value=True,
        join_method="temporal",
        join_confidence=confidence,
        retracted=retracted,
        outcome_ts_ns=0,
    )


BOOT = _uuid()


def test_high_confidence_pins_low_confidence_reviews():
    """A binding at/above the floor is pinned; one below it is held out for review."""
    r1, r2, ep = _uuid(), _uuid(), _uuid()
    rollouts = [_rollout(r1, ep, 1_000_000_000), _rollout(r2, ep, 2_000_000_000)]
    feedbacks = [_feedback(r1, confidence=1.0), _feedback(r2, confidence=0.90)]
    spec = dict(grain="rollout", min_confidence=0.95)

    result = fieldloop.curate(spec, rollouts, feedbacks)

    assert len(result["items"]) == 1
    assert result["items"][0]["rollout_id"] == r1
    assert len(result["needs_review"]) == 1
    assert result["needs_review"][0]["rollout_id"] == r2
    assert result["needs_review"][0]["reason"] == "below_min_confidence"
    assert result["content_hash"]


def test_retracted_binding_is_held_out():
    """A retracted (tombstoned) binding is not training data — it goes to needs-review."""
    r1, ep = _uuid(), _uuid()
    result = fieldloop.curate(
        dict(grain="rollout", min_confidence=0.5),
        [_rollout(r1, ep, 1_000_000_000)],
        [_feedback(r1, confidence=1.0, retracted=True)],
    )
    assert result["items"] == []
    assert len(result["needs_review"]) == 1
    assert result["needs_review"][0]["reason"] == "retracted"


def test_no_binding_at_all_is_held_out():
    """A rollout with no feedback has nothing authoritative to train on."""
    r1, ep = _uuid(), _uuid()
    result = fieldloop.curate(
        dict(grain="rollout"),
        [_rollout(r1, ep, 1_000_000_000)],
        [],
    )
    assert result["items"] == []
    assert result["needs_review"][0]["reason"] == "no_surviving_binding"


def test_episode_grain_rolls_up_episodes():
    """At episode grain the pinned per-step items roll up to the distinct episode set."""
    ep = _uuid()
    r1, r2 = _uuid(), _uuid()
    rollouts = [_rollout(r1, ep, 1_000_000_000), _rollout(r2, ep, 2_000_000_000)]
    feedbacks = [_feedback(r1), _feedback(r2)]

    result = fieldloop.curate(dict(grain="episode", min_confidence=0.5), rollouts, feedbacks)

    assert len(result["items"]) == 2  # two steps pinned
    assert result["episodes"] == [ep]  # rolled up to one episode


def test_content_hash_is_deterministic():
    """Identical (spec + inputs) compile to the identical content hash (reproducible)."""
    r1, ep = _uuid(), _uuid()
    args = (dict(grain="rollout", min_confidence=0.5), [_rollout(r1, ep, 1)], [_feedback(r1)])
    first = fieldloop.curate(*args)
    second = fieldloop.curate(*args)
    assert first["content_hash"] == second["content_hash"]
    assert first["content_hash"]


def test_synthetic_is_opt_in():
    """A synthetic rollout is dropped unless the spec opts in, never silently trained on."""
    r1, ep = _uuid(), _uuid()
    rollouts = [_rollout(r1, ep, 1_000_000_000, synthetic_generator="cosmos-3")]
    feedbacks = [_feedback(r1, confidence=1.0)]

    # Default: synthetic excluded entirely (not even needs-review — it never matched).
    default = fieldloop.curate(dict(grain="rollout", min_confidence=0.5), rollouts, feedbacks)
    assert default["items"] == []

    # Opt in: admitted, and tagged synthetic so a consumer can separate it.
    opted = fieldloop.curate(
        dict(grain="rollout", min_confidence=0.5, include_synthetic=True), rollouts, feedbacks
    )
    assert len(opted["items"]) == 1
    assert opted["items"][0]["provenance"] == "synthetic"


def test_attribute_feedbacks_pipe_into_curate():
    """The closed loop: attribute's output feeds curate with no reshaping."""
    r1, ep = _uuid(), _uuid()
    rollouts = [_rollout(r1, ep, 1_000_000_000)]
    outcomes = [
        dict(
            tenant_id="acme",
            robot_id="r7",
            boot_id=BOOT,
            mono_ns=1_200_000_000,
            outcome_kind="teleop_takeover",
            explicit_rollout_id=r1,
        )
    ]
    att = fieldloop.attribute(CONFIG, rollouts, outcomes)
    assert att["feedbacks"][0]["join_confidence"] == 1.0  # explicit

    # Pipe the feedbacks straight in; the explicit (1.0) binding pins.
    result = fieldloop.curate(dict(grain="rollout", min_confidence=0.8), rollouts, att["feedbacks"])
    assert len(result["items"]) == 1
    assert result["items"][0]["rollout_id"] == r1
