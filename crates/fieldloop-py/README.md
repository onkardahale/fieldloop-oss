# fieldloop-py — Python binding over the Rust capture core

This crate is **Part 3b**: a thin [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs)
binding that exposes the `fieldloop-capture` hot path to Python. It is the entry
point a robotics engineer (Cosmos / LeRobot / openpi) actually calls. It contains no
business logic of its own — it only marshals Python values to and from the Rust core,
which owns every behavioral guarantee (non-blocking minting, drop-and-count on a full
queue, off-loop draining).

## Layout

```
crates/fieldloop-py/
├── Cargo.toml          # cdylib crate; deps: pyo3, fieldloop-capture, fieldloop-types
├── pyproject.toml      # maturin build backend + uv project (pytest/ruff dev deps)
├── src/lib.rs          # the PyO3 binding: one `Capture` class, module `fieldloop`
├── tests/
│   └── test_capture.py # pytest closed-loop suite (built + run by `uv run pytest`)
└── README.md
```

The compiled crate **is** the importable `fieldloop` module (its `[lib] name` and
`#[pymodule]` are both `fieldloop`), so there is no separate Python source tree.

## API

`fieldloop.Capture(tenant_id: str, robot_id: str, capacity: int)`

- `.register_context(policy_version, model_hash, embodiment, task_id) -> int` —
  register the slowly-changing context once (off the hot path); returns an integer
  handle.
- `.log_step(episode_id: str, step_index: int, ctx: int, inference_us: int) -> str` —
  the hot-path call; mints and returns a rollout id (uuid string). Never blocks; on a
  full queue the record is dropped-and-counted but the id is still returned. Raises
  `ValueError` on a bad `episode_id` or unknown `ctx`.
- `.dropped() -> int` — count of records dropped because the queue was full.
- `.drain() -> list[dict]` — drain buffered records into flat dicts with keys
  `rollout_id`, `episode_id`, `policy_version`, `embodiment`, `task_id`, `step_index`,
  `mono_ns`, `inference_us`.

### Usage

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
