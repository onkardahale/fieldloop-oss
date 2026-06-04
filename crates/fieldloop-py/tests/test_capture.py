"""Closed-loop tests for the `fieldloop` Python binding.

These exercise the binding end-to-end against the compiled Rust core: construct a
Capture, register context, log steps, drain, and assert the drained dicts carry the
right ids and registered strings. They also pin the two non-negotiable behaviors:
a full queue drops-and-counts without ever blocking or raising, and a bad episode id
raises ValueError.
"""

import uuid

import fieldloop


def _episode() -> str:
    """A fresh, valid episode id as a UUID string (the binding parses it)."""
    return str(uuid.uuid4())


def test_log_steps_drain_to_matching_dicts():
    """log_step a few steps, drain, and check the dicts round-trip ids and context."""
    cap = fieldloop.Capture("tenant-a", "robot-1", capacity=64)
    ctx = cap.register_context(
        policy_version="nav@v1+abc123abc123",
        model_hash="model-sha-xyz",
        embodiment="arm6dof",
        task_id="pick-and-place",
    )
    episode = _episode()

    ids = [cap.log_step(episode, step_index=i, ctx=ctx, inference_us=100 + i) for i in range(3)]

    # Every returned id is a valid uuid string.
    for rid in ids:
        uuid.UUID(rid)

    rows = cap.drain()
    assert len(rows) == 3
    # Order preserved: drained rollouts come back in the order logged.
    for i, row in enumerate(rows):
        assert row["rollout_id"] == ids[i]
        assert row["episode_id"] == episode
        assert row["step_index"] == i
        assert row["inference_us"] == 100 + i
        # Registered context resolved back to its strings.
        assert row["policy_version"] == "nav@v1+abc123abc123"
        assert row["embodiment"] == "arm6dof"
        assert row["task_id"] == "pick-and-place"
        # mono_ns is present and non-negative (the monotonic clock reading).
        assert isinstance(row["mono_ns"], int)
        assert row["mono_ns"] >= 0


def test_distinct_contexts_resolve_independently():
    """Two registered contexts each resolve to their own strings on drain."""
    cap = fieldloop.Capture("tenant-a", "robot-1", capacity=16)
    ctx_a = cap.register_context("a@v1+aaaaaaaaaaaa", "ha", "emb-a", "task-a")
    ctx_b = cap.register_context("b@v1+bbbbbbbbbbbb", "hb", "emb-b", "task-b")
    episode = _episode()

    cap.log_step(episode, step_index=0, ctx=ctx_b, inference_us=1)
    cap.log_step(episode, step_index=1, ctx=ctx_a, inference_us=1)

    rows = cap.drain()
    assert len(rows) == 2
    assert rows[0]["embodiment"] == "emb-b"
    assert rows[0]["task_id"] == "task-b"
    assert rows[1]["embodiment"] == "emb-a"
    assert rows[1]["task_id"] == "task-a"


def test_full_queue_drops_and_counts_but_never_blocks():
    """A small queue overflows: dropped() rises, yet every log_step returns a uuid."""
    # Capacity 2: steps past the 2 buffered (undrained) overflow and are counted.
    cap = fieldloop.Capture("tenant-a", "robot-1", capacity=2)
    ctx = cap.register_context("p@v1+deadbeefcafe", "h", "emb", "task")
    episode = _episode()

    ids = [cap.log_step(episode, step_index=i, ctx=ctx, inference_us=10) for i in range(5)]

    # Nothing blocked or raised; every call returned a usable, distinct uuid string.
    assert len(ids) == 5
    for rid in ids:
        uuid.UUID(rid)
    assert len(set(ids)) == 5
    # Three records past capacity-2 were dropped and counted.
    assert cap.dropped() == 3


def test_bad_episode_id_raises_value_error():
    """A malformed episode id is a caller error surfaced as ValueError."""
    cap = fieldloop.Capture("tenant-a", "robot-1", capacity=8)
    ctx = cap.register_context("p@v1+0011223344", "h", "emb", "task")

    try:
        cap.log_step("not-a-uuid", step_index=0, ctx=ctx, inference_us=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a malformed episode_id")
