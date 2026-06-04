"""Closed-loop tests for the `fieldloop.select_uploads` binding.

These drive the real `fieldloop-trigger` detectors + selective-upload broker: build
rollouts/outcomes, call `select_uploads`, and assert the two non-negotiable behaviors —
a safety (reflex) trigger is ALWAYS kept even past the budget, and every budgeted
trigger that loses out is returned as a `dropped` record, never silently discarded.
"""

import uuid

import fieldloop


def _uuid() -> str:
    return str(uuid.uuid4())


BOOT = _uuid()


def _rollout(rollout_id: str, mono_ns: int, **extra) -> dict:
    return dict(
        rollout_id=rollout_id,
        tenant_id="acme",
        robot_id="r7",
        boot_id=BOOT,
        mono_ns=mono_ns,
        wall_ns=0,
        episode_id=_uuid(),
        step_index=0,
        policy_version="pick@v1+abc123abc123",
        model_hash="sha256:weights",
        embodiment="warehouse_pick_arm",
        task_id="bin_pick",
        inference_us=900,
        **extra,
    )


def _collision_outcome(rollout_id: str, mono_ns: int) -> dict:
    return dict(
        tenant_id="acme",
        robot_id="r7",
        boot_id=BOOT,
        mono_ns=mono_ns,
        outcome_kind="collision",
        explicit_rollout_id=rollout_id,
    )


def test_safety_trigger_bypasses_the_budget():
    """A reflex (collision) trigger is kept even at a zero budget; the droppable
    low-confidence pre-filter is dropped-and-counted."""
    rc, rl = _uuid(), _uuid()
    rollouts = [_rollout(rc, 1_000_000_000), _rollout(rl, 1_500_000_000, policy_confidence=0.05)]
    outcomes = [_collision_outcome(rc, 1_050_000_000)]

    result = fieldloop.select_uploads(rollouts, outcomes, max_requests=0)

    # Safety event kept despite the zero budget.
    assert len(result["requests"]) == 1
    assert result["requests"][0]["rollout_id"] == rc
    assert result["requests"][0]["tier"] == "reflex_a"
    # The risky-but-unconfirmed pre-filter was dropped — and surfaced, not lost.
    assert len(result["dropped"]) == 1
    assert result["dropped"][0]["rollout_id"] == rl
    assert result["dropped"][0]["tier"] == "pre_filter_a_prime"


def test_low_confidence_prefilter_fires_within_budget():
    """A low-confidence rollout trips the pre-filter; a generous budget keeps it."""
    rl = _uuid()
    rollouts = [_rollout(rl, 1_000_000_000, policy_confidence=0.05)]

    result = fieldloop.select_uploads(rollouts, [], max_requests=10)

    assert len(result["requests"]) == 1
    assert result["requests"][0]["tier"] == "pre_filter_a_prime"
    assert result["requests"][0]["detector_id"] == "prefilter.low_confidence"
    assert result["dropped"] == []


def test_confident_unremarkable_rollout_triggers_nothing():
    """A confident rollout with no bad outcome is unremarkable — captured by nothing."""
    r = _uuid()
    rollouts = [_rollout(r, 1_000_000_000, policy_confidence=0.95)]

    result = fieldloop.select_uploads(rollouts, [], max_requests=10)

    assert result["requests"] == []
    assert result["dropped"] == []


def test_budget_keeps_highest_priority_and_drops_the_rest():
    """Two droppable pre-filter triggers with a budget of one: the riskier is kept,
    the other dropped-and-counted (lower policy_confidence => higher priority)."""
    r_risky, r_less = _uuid(), _uuid()
    rollouts = [
        _rollout(r_risky, 1_000_000_000, policy_confidence=0.01),
        _rollout(r_less, 2_000_000_000, policy_confidence=0.20),
    ]

    result = fieldloop.select_uploads(rollouts, [], max_requests=1)

    assert len(result["requests"]) == 1
    assert len(result["dropped"]) == 1
    # The riskier (lower confidence) one wins the single slot.
    assert result["requests"][0]["rollout_id"] == r_risky
    assert result["dropped"][0]["rollout_id"] == r_less
