<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fieldloop-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/fieldloop-lockup-light.svg">
    <img alt="FieldLoop" src="assets/fieldloop-lockup-light.svg" width="320">
  </picture>
</p>

**An open incident workbench for robot field failures.**

FieldLoop helps a robotics team work from the logs it already has: find the
incident window, replay the right context, see whether the failure is recurring,
and turn useful events into training or evaluation evidence.

It is built for the messy middle of robotics work:

```text
robot runs
  -> logs everything
  -> something fails later
  -> engineer finds the window
  -> FieldLoop binds the outcome back to candidate decisions
  -> useful incidents become replayable evidence and curated slices
```

FieldLoop does not replace ROS bags, MCAP, Foxglove, RViz, notebooks, or your
training stack. Those tools tell you what happened. FieldLoop helps answer:

- Which decision, planner command, mode transition, or policy action preceded
  the outcome?
- Is this failure recurring across runs, robots, sites, or versions?
- Is the attribution strong enough to learn from?
- What should be replayed, reviewed, exported, or held out?

## Run the local loop

Requires Rust, [`uv`](https://docs.astral.sh/uv/), and Python 3.12+. Check your
toolchain:

```bash
./scripts/preflight.sh
```

Then run the loop locally. The first run compiles a Rust Python extension from
source; there is no PyPI install, database, cloud account, or fleet service:

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

The demo is intentionally small. It shows the core behavior on in-memory data:
capture decisions, attach late outcomes, attribute them with confidence, hold weak
evidence for review, and select the raw payloads worth pulling.

FieldLoop does not claim automatic root cause. It builds an evidence package:
ranked leads, replayable windows, attribution method, confidence, and a path for
human confirmation.

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

## What FieldLoop does

- Captures robot decision metadata without blocking the control loop.
- Accepts outcomes such as teleop takeovers, e-stops, collisions, downstream
  failures, task success, and heartbeat coverage.
- Attributes delayed outcomes back to candidate decisions by method and
  confidence.
- Keeps confirmed evidence, inferred hypotheses, and excluded bindings distinct.
- Extracts replayable incident windows instead of asking engineers to scrub huge
  logs by hand.
- Curates trusted `(decision, outcome)` pairs into training and evaluation slices.
- Selects which raw sensor payloads are worth pulling for review.

The point is not to record more data. The point is to decide which recorded events
are useful evidence.

## Why this matters

A robot log can show:

```text
12.4s  grasp command sent
14.1s  object slipped
```

That does not prove the grasp command caused the slip. The failure might belong to
perception, planning, control, hardware, a later disturbance, or an ambiguous
chain that should not train a policy at all.

FieldLoop makes that uncertainty explicit. Every binding says how it was joined,
how confident the system is, and whether a human confirmed it, should review it,
or should exclude it.

Weak evidence should not silently become training data.

## Where it fits

```text
ROS / MCAP / custom logs
        |
        v
FieldLoop local incident workbench
        |
        v
replay bundles + recurring failure evidence + curated slices
        |
        v
LeRobot / PyTorch / Isaac / custom training and eval
```

Start from historical logs and incident timestamps. Instrumentation improves the
next run, but it is not a prerequisite for first value.

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
data. Hosted fleet operations, enterprise workflow, production RBAC, durable audit
storage, fleet-wide reporting, and rollout-control operations are not presented as
finished public surfaces here.

## Implemented components

- Captures rollout decision metadata from the control-loop path.
- Records MCAP streams for sensor, metadata, and safety channels.
- Stores payload pointers so large sensor artifacts can stay outside the metadata plane.
- Marks outcomes such as teleop takeover, e-stop, collision, downstream failure, and task success.
- Attributes delayed outcomes back to rollout decisions with confidence and method metadata.
- Builds incident evidence bundles with confirmed, hypothesis, and excluded evidence.
- Assembles replay timelines that point back to recorded MCAP message ranges.
- Defines robot-specific adapters for action normalization, success detection, and attribution windows.
- Exports curated slices for training and evaluation workflows.

## Supported surface

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

## Learn more

- [`docs/quickstart.md`](docs/quickstart.md) — the attribution cascade on in-memory data.
- [`docs/concepts.md`](docs/concepts.md) — the loop concepts and evidence states.
- [`docs/onboard-your-robot.md`](docs/onboard-your-robot.md) — move from toy data to your robot.
- [`docs/incident-workbench.md`](docs/incident-workbench.md) — the local incident workflow.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

