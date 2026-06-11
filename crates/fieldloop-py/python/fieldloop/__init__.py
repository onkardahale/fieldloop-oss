"""FieldLoop — capture robot decisions, attribute delayed outcomes, curate evidence.

The public surface re-exported here is the whole API; the compiled Rust core lives in
the internal `fieldloop._native` module and callers should never import it directly.
Re-exporting from `__init__` keeps `import fieldloop` stable while the package layers
pure-Python tooling (the `fieldloop` CLI, the bundled demo) around the Rust core.

- `Capture` — on-robot, non-blocking decision capture for the control-loop hot path.
- `attribute(config_toml, rollouts, outcomes)` — bind delayed outcomes back to the
  decisions that caused them, with an explicit method and confidence per binding.
- `attribute_mcap(config_toml, mcap_bytes, mapping_toml)` — the same attribution run
  directly on an MCAP recording plus a topic-mapping TOML (the file-import path).
- `doctor(mcap_bytes, mapping_toml)` — check a mapping against an MCAP file's real
  topics and flag cross-clock skew, before attribution is ever run.
- `curate(spec, rollouts, feedbacks)` — compile trusted bindings into a training
  slice; weak evidence is held for review, never silently included.
- `select_uploads(rollouts, outcomes, max_requests=...)` — choose which raw payloads
  are worth pulling, under a budget that only safety triggers may bypass.
"""

from fieldloop._native import (
    Calibrator,
    Capture,
    attribute,
    attribute_mcap,
    curate,
    doctor,
    fit_calibrator,
    select_uploads,
)

__all__ = [
    "Calibrator",
    "Capture",
    "attribute",
    "attribute_mcap",
    "curate",
    "doctor",
    "fit_calibrator",
    "select_uploads",
]
