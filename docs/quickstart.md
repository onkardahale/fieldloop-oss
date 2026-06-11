# Quickstart

```bash
pip install fieldloop
fieldloop demo
```

The demo runs the whole loop on bundled data — capture, attribute, curate, select uploads —
then attributes a bundled MCAP recording: a teleop takeover and two e-stops bound back to the
decisions that preceded them, and a collision left unattributed because it fell outside its
window. No database, no cloud, no Rust toolchain.

## Attribute an MCAP you already have

You do not have to instrument anything first. If you have an MCAP recording, point `fieldloop`
at it plus a topic-mapping TOML that says which topics are decisions and which are outcomes
(`fieldloop init` scaffolds the embodiment config):

```bash
fieldloop doctor run.mcap --map mapping.toml                        # check topics + clocks first
fieldloop attribute run.mcap --map mapping.toml --config robot.toml
```

`doctor` exits non-zero if a declared topic is missing from the file, or if a topic's source
clock diverges from the recorder's (a sensor on a different time base). `attribute` writes
`incidents.jsonl` and a `report.md` binding each outcome back to the decisions that preceded
it, with method and confidence. Add `pip install "fieldloop[viz]"` and `fieldloop view
incidents.jsonl` to see it on a Rerun timeline.

A classic ROS `.bag` must be converted to MCAP first. Single-boot per file in this version.

## The engine directly (no Python)

The attribution cascade is a Rust crate; this runs it on three rollouts and three late signals
with no id attached to any outcome — the cascade reconstructs every link:

```bash
cargo run -p fieldloop-join --example quickstart
```

```
[
  { "bound_to": "decision#0 ...", "metric": "teleop_takeover", "outcome": "FAILURE", "method": "Temporal",          "confidence": 0.94 },
  { "bound_to": "decision#1 ...", "metric": "collision",       "outcome": "FAILURE (Hardware)", "method": "Temporal", "confidence": 0.60 },
  { "bound_to": "decision#2 ...", "metric": "task_success",    "outcome": "success", "method": "SyntheticAbsence",   "confidence": 1.00 }
]
```

The clean run binds a success by *absence*: heartbeats prove its window was covered and nothing
adverse happened.

## What just happened

| Step | In the example | In production |
|---|---|---|
| Capture | `Rollout::new(...)` | the recorder + sidecar agent on the robot |
| Outcome | `OutcomeEvent::new(...)` | a takeover/e-stop/collision/downstream signal, arriving late |
| Attribution | `attribute(config, rollouts, outcomes, heartbeats, opts)` | the join worker, fired on "rollout landed" |
| Result | the printed bindings | typed `Feedback` rows — the input the gate and RCA build on |

## Configuration

The per-robot attribution windows (how far back each outcome kind may bind) are declared in TOML
and validated at load. See `Config::example()` and `docs/adr/0001-tenant-query-builder.md`.

## Next

- `docs/onboard-your-robot.md` — from the demo to your own robot or recordings.
- `docs/concepts.md` — the loop: capture → join → curate → gate → deploy.
- `docs/adapter.md` — onboarding a new robot type.
- `docs/adr/` — the design decisions behind the storage, seams, and isolation model.
