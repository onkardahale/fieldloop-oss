# fieldloop-py — the `fieldloop` Python package

The Python entry point a robotics engineer (Cosmos / LeRobot / openpi) actually uses:
the on-robot capture hot path, the decision→outcome attribution engine, curation, and
selective upload, plus a `fieldloop` command-line front door. The behavior lives in the
Rust crates; this package only marshals values across the boundary and layers the CLI
and demo on top.

## Layout

Mixed [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) package: a compiled
Rust core wrapped by a small pure-Python tree.

```
crates/fieldloop-py/
├── Cargo.toml          # cdylib crate `_native`; deps: pyo3, fieldloop-capture, -types, -join, -config, -curation, -trigger
├── pyproject.toml      # maturin backend (mixed layout) + the `fieldloop` console script
├── src/                # the PyO3 binding: the `Capture` class + attribute/curate/select_uploads
├── python/fieldloop/
│   ├── __init__.py     # re-exports the public API from the compiled `_native` module
│   ├── _native.pyi     # type stubs for the compiled module (the public surface)
│   ├── _cli.py         # the `fieldloop` command (demo/init/attribute/curate)
│   ├── _demo.py        # the bundled toy-data loop (also backs `fieldloop demo`)
│   └── data/           # the packaged embodiment sample config (`fieldloop init`)
├── examples/loop.py    # the loop as a hackable script
└── tests/              # pytest closed-loop suite (built + run by `uv run pytest`)
```

The compiled crate is the internal `fieldloop._native` module; callers never import it
directly — the package `__init__` re-exports the whole API, so `import fieldloop` is
the stable surface. The underscore name lets the package carry the CLI and demo without
the Rust crate owning them.

## The `fieldloop` command

Installed as a console script with the wheel (`uv build --wheel` produces a
`pip install`-able wheel that runs with no Rust toolchain present):

```bash
fieldloop demo                 # the bundled loop, end to end (--json for stable counts)
fieldloop init                 # scaffold an embodiment config to edit
fieldloop attribute --config c.toml --rollouts r.jsonl --outcomes o.jsonl --out fb.jsonl
fieldloop curate --rollouts r.jsonl --feedbacks fb.jsonl --out slice.json
```

`attribute` and `curate` read JSON Lines (one dict per line — the exact shapes the
in-process API takes), so any logging stack can feed them with a few lines of glue.
Exit codes are scriptable: `0` success, `1` the engine rejected the inputs, `2` the
files were unreadable or unwritable.

## The library API

`fieldloop.Capture(tenant_id: str, robot_id: str, capacity: int)` — on-robot capture:

- `.register_context(policy_version, model_hash, embodiment, task_id) -> int` —
  register the slowly-changing context once (off the hot path); returns an integer
  handle.
- `.log_step(episode_id: str, step_index: int, ctx: int, inference_us: int) -> str` —
  the hot-path call; mints and returns a rollout id (uuid string). Never blocks; on a
  full queue the record is dropped-and-counted but the id is still returned. Raises
  `ValueError` on a bad `episode_id` or unknown `ctx`.
- `.dropped() -> int` — count of records dropped because the queue was full.
- `.drain() -> list[dict]` — drain buffered records into flat dicts (keys include
  `rollout_id`, `tenant_id`, `robot_id`, `boot_id`, `episode_id`, `policy_version`,
  `model_hash`, `embodiment`, `task_id`, `step_index`, `mono_ns`, `wall_ns`,
  `inference_us`), shaped to feed straight into `attribute`.

Module functions: `attribute(config_toml, rollouts, outcomes)` (bind delayed outcomes
to decisions, with method + a scored confidence), `curate(spec, rollouts, feedbacks)`
(compile a training slice; weak evidence is held for review), and
`select_uploads(rollouts, outcomes, max_requests=...)` (budgeted payload selection).

```python
import uuid, fieldloop
cap = fieldloop.Capture("tenant-a", "robot-1", capacity=1024)
ctx = cap.register_context("nav@v1+abc123abc123", "model-sha", "arm6dof", "pick")
episode = str(uuid.uuid4())
rollout_id = cap.log_step(episode, step_index=0, ctx=ctx, inference_us=180)
rows = cap.drain()  # [{'rollout_id': rollout_id, 'episode_id': episode, ...}]
```

## Build & test

The repo's single closed-loop gate builds the extension and runs these tests:

```
bash scripts/check.sh
```

Standalone: `uv run --project crates/fieldloop-py --extra dev pytest -q`.
