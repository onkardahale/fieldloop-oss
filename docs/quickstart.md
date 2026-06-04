# Quickstart

Run the field-learning JOIN end to end, with no database to set up, in one command:

```
cargo run -p fieldloop-join --example quickstart
```

It builds three policy rollouts (decisions a robot made), three real-world signals that arrive
later (a takeover, a collision, and silence), runs the attribution cascade, and prints which
outcome bound to which decision — by what method and at what confidence:

```
[
  { "bound_to": "decision#0 ...", "metric": "teleop_takeover", "outcome": "FAILURE", "method": "Temporal",          "confidence": 0.94 },
  { "bound_to": "decision#1 ...", "metric": "collision",       "outcome": "FAILURE (Hardware)", "method": "Temporal", "confidence": 0.60 },
  { "bound_to": "decision#2 ...", "metric": "task_success",    "outcome": "success", "method": "SyntheticAbsence",   "confidence": 1.00 }
]
```

No id was attached to any outcome — the cascade reconstructed every link. The "clean" run binds a
success by *absence*: heartbeats prove its window was covered and nothing adverse happened.

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

- `docs/concepts.md` — the loop: capture → join → curate → gate → deploy.
- `docs/adapter.md` — onboarding a new robot type.
- `docs/adr/` — the design decisions behind the storage, seams, and isolation model.
