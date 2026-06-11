"""Type stubs for the compiled `fieldloop._native` extension module.

The runtime module is the compiled Rust crate `fieldloop-py`; these stubs describe its
public surface for type checkers and editors (the package `__init__` re-exports this
whole surface as the public API). They are hand-maintained to match the binding's Rust source: `src/lib.rs` (the
`Capture` class and the module surface), `src/attribute.rs` (`attribute`),
`src/curate.rs` (`curate`), and `src/trigger.rs` (`select_uploads`).
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

class Calibrator:
    """An opaque, fitted confidence calibrator (built by `fit_calibrator`).

    Pass it to `attribute(..., calibrator=...)` so binding confidences reflect observed
    accuracy instead of the raw recency/coverage score. Construct only via
    `fit_calibrator`; it has no public attributes."""

def fit_calibrator(
    feedbacks: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    min_labels: int = ...,
) -> Calibrator:
    """Fit a `Calibrator` from feedback history (curator `manual` labels as ground truth,
    bucketed per (join_method, embodiment), isotonic). Buckets with fewer than
    `min_labels` samples stay uncalibrated (identity). Pass the result to `attribute`."""

def attribute(
    config_toml: str,
    rollouts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    join_version: Optional[str] = ...,
    heartbeat_period_ns: Optional[int] = ...,
    absence_metric_name: Optional[str] = ...,
    calibrator: Optional[Calibrator] = ...,
) -> dict[str, list[dict[str, Any]]]:
    """Attribute outcomes to the rollouts that caused them.

    `config_toml` is the embodiment/attribution-window config; `rollouts` and
    `outcomes` are flat dicts (a `heartbeat` outcome_kind routes to the coverage path).
    With a `calibrator` from `fit_calibrator`, confidences are calibrated; without one,
    the honest identity default (confidence = raw score) is used.
    Returns `{"feedbacks": [...], "skipped": [...]}`. Raises `ValueError` on an invalid
    config, a malformed id, an unknown outcome kind, or a missing required field.
    """

def attribute_mcap(
    config_toml: str,
    mcap_bytes: bytes,
    mapping_toml: str,
    *,
    join_version: Optional[str] = ...,
    heartbeat_period_ns: Optional[int] = ...,
    absence_metric_name: Optional[str] = ...,
    calibrator: Optional[Calibrator] = ...,
) -> dict[str, list[dict[str, Any]]]:
    """Attribute the outcomes in an MCAP file to the decisions that caused them.

    The file-import counterpart to `attribute`: `mcap_bytes` is the raw bytes of an MCAP
    recording and `mapping_toml` is a topic-mapping TOML (identity constants, a `[clock]`
    source, and the decision/outcome topic lists). The file is decoded into typed
    rollouts/outcomes (a `heartbeat` kind routes to the coverage path), then the same
    engine runs. `config_toml` is the embodiment/attribution-window config; the mapping's
    `embodiment` must name one present in it. Returns the same
    `{"feedbacks": [...], "skipped": [...]}` shape. Raises `ValueError` on an invalid
    config, an invalid mapping, or an undecodable MCAP.
    """

def doctor(
    mcap_bytes: bytes,
    mapping_toml: str,
    *,
    clock_skew_threshold_ns: Optional[int] = ...,
) -> dict[str, Any]:
    """Check an MCAP file against a topic-mapping before attribution.

    `mcap_bytes` is the raw bytes of an MCAP recording; `mapping_toml` is the topic-mapping
    TOML. Returns a dict: `ok` (no problems), `clock_skew_threshold_ns`, `missing_topics`
    (declared in the mapping but absent from the file), `skewed_topics` (source clock
    diverges from the recorder clock past the threshold), and `topics` (per-topic `topic`,
    `role`, `message_count`, `log_time_min_ns`, `log_time_max_ns`, `max_clock_skew_ns`).
    `clock_skew_threshold_ns` overrides the 1-second default. Raises `ValueError` on an
    invalid mapping or an undecodable MCAP.
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
