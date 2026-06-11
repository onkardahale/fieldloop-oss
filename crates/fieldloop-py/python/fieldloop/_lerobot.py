"""Optional LeRobot import (`fieldloop[lerobot]`).

Turns a recorded LeRobotDataset into the rollout + outcome dicts `fieldloop.attribute`
consumes, so a team can attribute outcomes across their existing demonstration/eval
datasets. LeRobot is a heavy optional extra (it pulls torch); it is imported lazily here
so the core paths never pay for it and a caller without the extra gets a named
ImportError instead of a traceback.

Mapping (single-boot-per-episode v1, stated): each LeRobot frame becomes one rollout
(a decision) — `episode_index` -> a per-episode boot/episode id, `frame_index` ->
`step_index`, `timestamp` (seconds) -> `mono_ns`. Each episode's final frame becomes one
terminal outcome of the caller-declared kind: the end of a LeRobot episode is its done
signal, and the caller says what that terminus means (a success, a downstream failure,
...). Per-frame outcome columns (`next.done`/`next.success`) are a future addition; v1
keeps the mapping explicit rather than guessing a column's meaning.
"""

from __future__ import annotations

import uuid
from typing import Any

_MISSING = (
    "fieldloop LeRobot import needs the optional 'lerobot' extra. Install it with:\n"
    '    pip install "fieldloop[lerobot]"'
)


def _require_lerobot():
    """Import the LeRobot dataset loader or raise an [`ImportError`] naming the missing extra."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_MISSING) from exc
    return LeRobotDataset


def load_dataset(repo_id: str, root: str):
    """Load a LeRobotDataset from a local `root` (the real loader)."""
    LeRobotDataset = _require_lerobot()
    return LeRobotDataset(repo_id, root=root)


def _as_scalar(value: Any) -> Any:
    """LeRobot frame fields are tensors; `.item()` pulls the Python scalar. Plain values
    (e.g. the `task` string) pass through unchanged."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _resolve_task(task_id: str | None, frame_task: str | None) -> str:
    """Return the task label for a rollout.

    Priority chain:
    1. caller-supplied `task_id` (explicit override, highest priority)
    2. `frame_task` resolved from the dataset's own task registry (a string)
    3. the literal string ``"task"`` — keeps a record attributable when neither is set
    """
    if task_id:
        return task_id
    if isinstance(frame_task, str):
        return frame_task
    return "task"


def _build_task_map(dataset) -> dict[int, str] | None:
    """Reverse meta.tasks into a task_index -> task_name lookup.

    LeRobotDataset v3 stores a DataFrame at `meta.tasks` whose index is the task name
    and whose `task_index` column holds the integer id. Returns None when the attribute
    is absent or the structure is unexpected, so callers fall back gracefully.
    """
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return None
    tasks_df = getattr(meta, "tasks", None)
    if tasks_df is None:
        return None
    try:
        return {int(row["task_index"]): name for name, row in tasks_df.iterrows()}
    except Exception:
        return None


def import_dataset(
    dataset,
    *,
    tenant_id: str,
    robot_id: str,
    policy_version: str,
    embodiment: str,
    terminal_outcome: str,
    task_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert a loaded LeRobotDataset into `{"rollouts": [...], "outcomes": [...]}`.

    `terminal_outcome` is the outcome kind to stamp on each episode's final frame (one of
    the engine's kinds, e.g. `task_success`/`downstream_failure`). `task_id` overrides the
    frame's own `task` label when given. A fresh `boot_id`/`episode_id` is minted per
    LeRobot episode so each episode is a self-contained, monotonic-clock-comparable unit.

    When the dataset exposes an Arrow-backed `hf_dataset`, the scalar columns
    (`episode_index`, `frame_index`, `timestamp`, `task_index`) are read from the
    columnar store directly — skipping per-frame tensor decoding. The per-frame
    `dataset[i]` path is kept as a fallback for datasets that do not expose `hf_dataset`.
    """
    task_map = _build_task_map(dataset)

    # Collect frames per episode so step ordering and per-episode ids are stable.
    by_episode: dict[int, list[dict[str, Any]]] = {}

    hf = getattr(dataset, "hf_dataset", None)
    if hf is not None:
        # Read only the scalar columns needed; skips image/video tensor decoding entirely.
        wanted = [c for c in ("episode_index", "frame_index", "timestamp", "task_index")
                  if c in hf.column_names]
        rows = hf.select_columns(wanted).to_dict()
        n = len(next(iter(rows.values())))
        for i in range(n):
            episode_index = int(rows["episode_index"][i])
            frame_index = int(rows["frame_index"][i])
            mono_ns = round(float(rows["timestamp"][i]) * 1_000_000_000)
            # Resolve task string: task_index -> task name via meta.tasks reverse map.
            if "task_index" in rows and task_map is not None:
                frame_task: str | None = task_map.get(int(rows["task_index"][i]))
            else:
                frame_task = None
            by_episode.setdefault(episode_index, []).append(
                {"frame_index": frame_index, "mono_ns": mono_ns, "task": frame_task}
            )
    else:
        # Fallback: per-frame __getitem__ decodes all columns including tensors.
        for i in range(len(dataset)):
            frame = dataset[i]
            episode_index = int(_as_scalar(frame["episode_index"]))
            frame_index = int(_as_scalar(frame["frame_index"]))
            mono_ns = round(float(_as_scalar(frame["timestamp"])) * 1_000_000_000)
            frame_task = _as_scalar(frame.get("task")) if "task" in frame else None
            by_episode.setdefault(episode_index, []).append(
                {"frame_index": frame_index, "mono_ns": mono_ns, "task": frame_task}
            )

    rollouts: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for episode_index in sorted(by_episode):
        frames = sorted(by_episode[episode_index], key=lambda f: f["frame_index"])
        boot_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        for frame in frames:
            rollouts.append(
                {
                    "rollout_id": str(uuid.uuid4()),
                    "tenant_id": tenant_id,
                    "robot_id": robot_id,
                    "boot_id": boot_id,
                    "episode_id": episode_id,
                    "step_index": frame["frame_index"],
                    "mono_ns": frame["mono_ns"],
                    "policy_version": policy_version,
                    "embodiment": embodiment,
                    "task_id": _resolve_task(task_id, frame["task"]),
                }
            )
        # The episode's final frame is its terminal outcome; stamping it at that frame's
        # clock binds it (delta 0, in-window) to the last decision of the episode.
        last = frames[-1]
        outcomes.append(
            {
                "outcome_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "boot_id": boot_id,
                "mono_ns": last["mono_ns"],
                "outcome_kind": terminal_outcome,
            }
        )
    return {"rollouts": rollouts, "outcomes": outcomes}
