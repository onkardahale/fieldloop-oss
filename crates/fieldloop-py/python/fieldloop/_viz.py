"""Optional Rerun visualization for attributed incidents (`fieldloop[viz]`).

Renders the incidents from `fieldloop attribute` onto a Rerun timeline with attributed
and unattributed outcomes under distinct entity paths. Rerun is an optional extra;
importing it lazily here means the core loop and the attribute/curate/doctor paths never
pay for it, and a caller without the extra gets a named ImportError instead of a
traceback.
"""

from __future__ import annotations

from typing import Any, Optional

# The message shown when the optional dependency is absent — names the exact install.
_MISSING = (
    "fieldloop visualization needs the optional 'viz' extra. Install it with:\n"
    '    pip install "fieldloop[viz]"'
)


def _require_rerun():
    """Import Rerun or raise an [`ImportError`] naming the missing extra."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_MISSING) from exc
    return rr


def _incident_summary(incident: dict[str, Any]) -> str:
    """One-line human description of an incident for the timeline event."""
    if incident.get("state") == "attributed":
        conf = incident.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        return (
            f"{incident.get('join_method')} -> {incident.get('bound_target_id')} "
            f"(confidence {conf_str})"
        )
    return f"unattributed: {incident.get('reason')} ({incident.get('outcome_kind')})"


def log_incidents(
    incidents: list[dict[str, Any]],
    *,
    application_id: str = "fieldloop",
    save_path: Optional[str] = None,
    spawn: bool = False,
) -> None:
    """Log `incidents` (the rows `fieldloop attribute` writes) to a Rerun recording.

    With `save_path` the recording is written to a `.rrd` file (what CI and the README
    screenshot use); with `spawn=True` the Rerun viewer is launched live. Attributed and
    unattributed outcomes are logged under separate entity-path roots so they are visually
    distinct. Raises `ImportError` (naming the `viz` extra) if Rerun is not installed.
    """
    rr = _require_rerun()
    # A local recording stream (not the global one) keeps this call self-contained: no
    # process-wide state to leak between runs, and an explicit flush at the end.
    recording = rr.RecordingStream(application_id)
    if save_path is not None:
        recording.save(save_path)
    if spawn:
        recording.spawn()

    for index, incident in enumerate(incidents):
        # A sequence timeline keyed by incident order: the rows carry no per-incident wall
        # time, and the order is already the engine's deterministic incident order.
        recording.set_time("incident", sequence=index)
        root = "attributed" if incident.get("state") == "attributed" else "unattributed"
        outcome = incident.get("outcome_id") or f"index_{index}"
        recording.log(f"{root}/{outcome}", rr.TextLog(_incident_summary(incident)))

    # Force the recording out to the sink before returning, so a caller that saved a file
    # sees a complete `.rrd` immediately rather than relying on an interpreter-exit flush.
    recording.flush()
