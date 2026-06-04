<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fieldloop-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/fieldloop-lockup-light.svg">
    <img alt="FieldLoop" src="assets/fieldloop-lockup-light.svg" width="320">
  </picture>
</p>

**FieldLoop binds your robot's decisions to what happened next.**

It captures the decisions a policy made, attributes delayed real-world outcomes back to
the decision that caused them (with calibrated confidence), curates the trusted ones into
a training/eval slice, and selects which raw sensor payloads are worth pulling — for one
robotics team, on its own data, with no database to set up. (The loop is fully in-memory;
persisting to ClickHouse/Postgres is an opt-in, not a prerequisite.)

## Run the loop in one command

Requires Rust, [`uv`](https://docs.astral.sh/uv/), and Python 3.12+. Check your toolchain:

```bash
./scripts/preflight.sh
```

Then run the whole loop locally (the first run compiles a Rust Python extension from
source — no PyPI install, no services):

```bash
uv run --project crates/fieldloop-py --extra dev \
  python crates/fieldloop-py/examples/loop.py
```

You should see:

```text
1. CAPTURE         captured 3 policy decisions
2. ATTRIBUTE       attributed 3 outcomes with calibrated confidence (no rollout_id given)
3. CURATE          accepted 2 feedbacks · needs_review 1 (below_min_confidence)
4. SELECT UPLOADS  selected 1 payload request (reflex_a, safety) · 2 budget drops
```

No database to set up. No cloud account. No fleet. The core idea: FieldLoop does not claim
root cause — it attributes delayed outcomes back to policy decisions *with confidence*,
then **fails closed** when confidence is too low.

## What to touch first

Most engineers only need these:

| You want to… | Start here |
|---|---|
| See the whole loop in Python | `crates/fieldloop-py/examples/loop.py` |
| Capture decisions | `fieldloop.Capture(...)` |
| Attribute outcomes to decisions | `fieldloop.attribute(config_toml, rollouts, outcomes)` |
| Build a training/eval slice | `fieldloop.curate(spec, rollouts, feedbacks)` |
| Choose which payloads to upload | `fieldloop.select_uploads(rollouts, outcomes, max_requests=…)` |
| Wire up your own robot | [`docs/onboard-your-robot.md`](docs/onboard-your-robot.md) |
| Integrate natively from Rust | `cargo run -p fieldloop-adapter --example onboard_your_robot` |

Everything else in the workspace is internals — ignore it until you need it.

## Current Status

Supported in this workspace:

- Rust capture SDK and Python binding for decision metadata.
- MCAP recorder for sensor, metadata, and safety streams.
- Ingest router, row serialization, migrations, and tenant-scoped query helpers.
- Outcome-to-rollout attribution engine.
- Incident evidence, replay, curation, evaluation, and LeRobot-style export primitives.

Not presented as finished product surfaces here: hosted fleet operations, a
fleet dashboard, enterprise workflow, production RBAC, durable audit storage, and
rollout-control operations.

## Intended Users

FieldLoop is for robotics ML, autonomy, field, and evaluation engineers working
with robot logs, MCAP recordings, teleop traces, outcome events, and policy
versions.

It is meant to answer:

- What did the robot do before this failure?
- Which policy version and model hash produced the decision?
- Which rollout was attributed to the later outcome?
- Can this incident be replayed from the recorded timeline and reviewed?
- Can this become curated training or evaluation data?

The current public surface is for one robotics team working locally with its own
data.

## Implemented Components

- Captures rollout decision metadata from the control-loop path.
- Records MCAP streams for sensor, metadata, and safety channels.
- Stores payload pointers so large sensor artifacts can stay outside the metadata plane.
- Marks outcomes such as teleop takeover, e-stop, collision, downstream failure, and task success.
- Attributes delayed outcomes back to rollout decisions with confidence and method metadata.
- Builds incident evidence bundles with confirmed, hypothesis, and excluded evidence.
- Assembles replay timelines that point back to recorded MCAP message ranges.
- Defines robot-specific adapters for action normalization, success detection, and attribution windows.
- Exports curated slices for training and evaluation workflows.

## Supported Surface

Supported integration points in this workspace:

- Rust and Python capture paths for decision metadata.
- MCAP recording for field context and replay.
- Payload pointers for large sensor artifacts kept outside the metadata plane.
- Robot-specific adapters for action normalization, success detection, and attribution windows.
- Curation and export paths for robot learning workflows, including LeRobot-style datasets.

## Capture on its own (Python)

The loop demo above is the full story; this is just the capture stage in isolation, for
when you are wiring the on-robot hot path. Requires Python 3.12+ and `uv`.

```bash
uv run --project crates/fieldloop-py --extra dev python crates/fieldloop-py/examples/quickstart.py
```

Expected output:

```text
rollout_id: <uuid>
drained: 1
dropped: 0
```

The example uses the `fieldloop` Python extension module built from
`crates/fieldloop-py`.

```python
import uuid
import fieldloop

cap = fieldloop.Capture("tenant-a", "robot-1", capacity=1024)

ctx = cap.register_context(
    policy_version="pick@v1+abc123abc123",
    model_hash="sha256:model-hash",
    embodiment="warehouse_pick_arm",
    task_id="pick-can",
)

episode_id = str(uuid.uuid4())

# Call this inside or near the robot control loop.
rollout_id = cap.log_step(
    episode_id,
    step_index=0,
    ctx=ctx,
    inference_us=180,
)

# Drain off the hot path.
rows = cap.drain()

print(rollout_id)
print(rows)
print("dropped:", cap.dropped())
```

## Capture on its own (Rust)

Run the checked-in Rust example:

```bash
cargo run -p fieldloop-capture --example quickstart
```

Expected output:

```text
rollout_id = <uuid>
drained = 1
dropped = 0
```

```rust
use fieldloop_capture::Capture;
use fieldloop_types::{
    EpisodeId, PolicyVersion, RobotId, RobotIdentity, TenantId,
};

fn main() {
    let robot = RobotIdentity::new(
        TenantId::new("tenant-a"),
        RobotId::new("robot-1"),
    );

    let (capture, drain) = Capture::new(robot, 1024);

    let ctx = capture.register_context(
        PolicyVersion::new("pick@v1+abc123abc123"),
        "sha256:model-hash",
        "warehouse_pick_arm",
        "pick-can",
    );

    let rollout_id = capture.log_step(
        EpisodeId::new(),
        0,
        ctx,
        180,
    );

    let rollouts = drain.drain_available();

    println!("rollout_id = {rollout_id}");
    println!("drained = {}", rollouts.len());
    println!("dropped = {}", capture.dropped());
}
```

## Onboard A Robot Type

Start with the adapter example:

```bash
cargo run -p fieldloop-adapter --example onboard_your_robot
```

An adapter defines:

- The robot embodiment name.
- The action space and action normalization.
- Attribution windows for each outcome kind.
- Which outcomes require monotonic-clock colocation.
- A failure-class hint for observed outcomes.
- A total success detector that can return `Unknown`.
- The feature schema used by downstream exports.

See:

- `crates/fieldloop-adapter/examples/onboard_your_robot.rs`
- `crates/fieldloop-config/examples/embodiment.sample.toml`

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

