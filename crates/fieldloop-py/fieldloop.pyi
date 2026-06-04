"""Type stubs for the `fieldloop` extension module.

The runtime module is the compiled Rust crate `fieldloop-py`; these stubs describe its
public surface for type checkers and editors. They are hand-maintained to match
`src/lib.rs` (the `Capture` class) and `src/attribute.rs` (the `attribute` function).
"""

from typing import Any, Optional

class Capture:
    """On-robot capture handle for a single robot's control loop.

    The hot-path call (`log_step`) never blocks and never raises on a full queue: it
    mints an id, enqueues a compact record, and drops-and-counts under backpressure.
    Heavy work (resolving context, assembling rollouts) happens off-loop in `drain`.
    """

    def __init__(self, tenant_id: str, robot_id: str, capacity: int) -> None:
        """Construct a capture instance with a bounded queue of `capacity` records."""

    def register_context(
        self,
        policy_version: str,
        model_hash: str,
        embodiment: str,
        task_id: str,
    ) -> int:
        """Register slowly-changing context once, off the hot path; returns a handle
        to pass to `log_step`."""

    def log_step(
        self, episode_id: str, step_index: int, ctx: int, inference_us: int
    ) -> str:
        """Hot-path: mint and return a rollout id, enqueueing a compact record.

        Never blocks; drops-and-counts on a full queue. Raises `ValueError` on a
        malformed `episode_id` or an unknown `ctx` handle."""

    def dropped(self) -> int:
        """Records dropped so far because the bounded queue was full."""

    def drain(self) -> list[dict[str, Any]]:
        """Drain buffered records into rollout dicts. Each dict is exactly the shape
        `attribute` consumes: `rollout_id`, `tenant_id`, `robot_id`, `boot_id`,
        `mono_ns`, `wall_ns`, `episode_id`, `step_index`, `policy_version`,
        `model_hash`, `embodiment`, `task_id`, `inference_us`."""

def attribute(
    config_toml: str,
    rollouts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    join_version: Optional[str] = ...,
    heartbeat_period_ns: Optional[int] = ...,
    absence_metric_name: Optional[str] = ...,
) -> dict[str, list[dict[str, Any]]]:
    """Attribute outcomes to the rollouts that caused them.

    `config_toml` is the embodiment/attribution-window config; `rollouts` and
    `outcomes` are flat dicts (a `heartbeat` outcome_kind routes to the coverage path).
    Returns `{"feedbacks": [...], "skipped": [...]}`. Raises `ValueError` on an invalid
    config, a malformed id, an unknown outcome kind, or a missing required field.
    """

def curate(
    spec: dict[str, Any],
    rollouts: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile a training slice: pin which (rollout, authoritative-feedback) pairs
    become training data, holding out the doubtful ones.

    `spec` is a slice-spec dict (required `grain`: "rollout"|"episode"; optional
    `min_confidence`, `policy_version`, `failure_class`, `task_id`, `site`,
    `include_synthetic`, `outcome_ts={start_ns,end_ns}`). `rollouts`/`feedbacks` are the
    dicts `attribute` consumes/emits. Returns `{"tenant_id", "content_hash", "items",
    "episodes", "needs_review"}`. Raises `ValueError` on a malformed dict.
    """

def select_uploads(
    rollouts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    max_requests: int,
    max_bytes: Optional[int] = ...,
    bytes_per_request: int = ...,
) -> dict[str, list[dict[str, Any]]]:
    """Decide which rollouts' heavy sensor payloads to pull, under a budget.

    Runs the built-in detectors (reflex on bad outcomes; low-confidence pre-filter on a
    rollout's `policy_confidence`) and the budget broker. Safety (reflex) triggers are
    kept even past `max_requests`. Returns `{"requests": [...], "dropped": [...]}`.
    Raises `ValueError` on a malformed rollout/outcome dict.
    """
