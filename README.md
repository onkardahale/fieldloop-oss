<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fieldloop-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/fieldloop-lockup-light.svg">
    <img alt="FieldLoop" src="assets/fieldloop-lockup-light.svg" width="320">
  </picture>
</p>

**An open incident workbench for robot field failures.**

A robot log can show:

```text
12.4s  grasp command sent
14.1s  object slipped
```

That does not prove the grasp command caused the slip. The failure might belong to
perception, planning, control, hardware, a later disturbance, or an ambiguous chain
that should not train a policy at all.

FieldLoop helps a robotics team work from the logs it already has: find the
incident window, replay the right context, see whether the failure is recurring,
and turn useful events into training or evaluation evidence. Every binding says how
it was joined, how confident the system is, and whether a human confirmed it, should
review it, or should exclude it. Weak evidence is held for review instead of
silently becoming training data.

The workflow:

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
- Is this binding reliable enough to train on?
- What should be replayed, reviewed, exported, or held out?

## Install and run

```bash
pip install fieldloop
fieldloop demo
```

`fieldloop demo` runs the bundled loop end to end, with no database, cloud account, or
Rust toolchain. `pip install "fieldloop[viz]"` and `fieldloop demo --viz` additionally
open the attributed incidents on a Rerun timeline. `fieldloop init` scaffolds a robot
config; `fieldloop attribute`, `fieldloop doctor`, and `fieldloop curate` run the stages
on your own files (`fieldloop --help`).

To build from source (to develop, or before a wheel exists for your platform) you need
Rust, [`uv`](https://docs.astral.sh/uv/), and Python 3.12+. `./scripts/preflight.sh`
checks the toolchain; `uv run --project crates/fieldloop-py --extra dev fieldloop demo`
then builds the Rust extension and runs the same command. The loop is also a script at
`crates/fieldloop-py/examples/loop.py`.

You should see (abridged):

```text
1. CAPTURE
   captured 3 policy decisions

2. ATTRIBUTE
   received 3 outcomes (2 with no rollout_id — the link is reconstructed)
   attributed 3 outcomes:
   - temporal           confidence=0.92  decision=slow_approach
   - explicit           confidence=1.00  decision=brake_before_turn
   - temporal           confidence=0.24  decision=no_recovery_action

3. CURATE
   training slice:
   - accepted 2 feedbacks
   - needs_review 1 feedbacks
     reason: below_min_confidence

4. SELECT UPLOADS  (budget: 0 — only safety may bypass it)
   selected 1 payload request
   - reflex.outcome  tier=reflex_a

5. ATTRIBUTE FROM AN MCAP FILE  (the bundled demo.mcap, the file-import path)
   MCAP demo: 4 bindings — a teleop takeover and two e-stop events traced to
   their decisions (one e-stop splits credit across a near-simultaneous pair):
   - temporal  confidence=0.90  delay=150ms  credit=1.00
   - temporal  confidence=0.60  delay=200ms  credit=1.00
   - temporal  confidence=0.80  delay=400ms  credit=0.14
   - temporal  confidence=0.80  delay=100ms  credit=0.86
   refused: a collision outside its window stays unattributed
   (no_candidate_in_window) — never guessed
```

The demo is intentionally small. Stages 1–4 show the core behavior on in-memory data:
capture decisions, attach late outcomes, attribute them with an explicit method and
confidence, hold weak evidence for review, and select the raw payloads worth
pulling. The 0.24 binding is the point: it is attributed, but it does not enter the
training slice — it goes to review. Stage 5 runs the *same* engine on a real MCAP
binary (~3,400 messages: 10 Hz decision bursts with jitter, three sensor topics)
through the file-import path (`fieldloop attribute run.mcap --map mapping.toml`):
a clean takeover binding, the 0.60 e-stop, a credit split across a near-simultaneous
pair, and a collision that is refused rather than guessed — the same behaviors you
get on your own recordings. A second bundled file, `demo-skew.mcap`, carries a
sensor topic 32 s off its recorder clock so `fieldloop doctor` has a real catch to
show.

FieldLoop does not claim automatic root cause. It builds an evidence package:
ranked leads, replayable windows, attribution method, confidence, and a path for
human confirmation.

## Attribute your own MCAP

Point `fieldloop` at a recording plus a topic-mapping TOML that declares which topics are
decisions and which are outcomes (`fieldloop init` scaffolds the embodiment config):

```toml
# mapping.toml
tenant_id = "acme"
robot_id = "robot-01"
boot_id = "00000000-0000-7000-8000-0000000000d0"     # one boot per file (v1)
episode_id = "00000000-0000-7000-8000-0000000000e0"
embodiment = "warehouse_pick_arm"
policy_version = "pi-2025.10"
task_id = "pick-place"

[[decisions]]
topic = "/policy/action"

[[outcomes]]
topic = "/safety/estop"
outcome_kind = "e_stop"
```

```bash
fieldloop doctor run.mcap --map mapping.toml                        # check topics + clocks first
fieldloop attribute run.mcap --map mapping.toml --config robot.toml
```

`doctor` exits non-zero if a declared topic is missing from the file, or if a topic's
source clock diverges from the recorder's (a sensor on a different time base). `attribute`
writes `incidents.jsonl` and a `report.md` that bind each outcome back to the decisions
that preceded it, with method and confidence.

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

The point is not to record more data. The point is data triage: deciding which
recorded events are worth keeping as evidence.

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
next run, but is not required to start.

## Intended Users

FieldLoop is for robotics ML, autonomy, field, and evaluation engineers working
with robot logs, MCAP recordings, teleop traces, outcome events, and policy
versions.

It answers:

- What did the robot do before this failure?
- Which policy version and model hash produced the decision?
- Which rollout preceded this incident?
- Can this incident be replayed from the recorded timeline and reviewed?
- Can this become curated training or evaluation data?

## What is solid today, and what is not

Robotics tooling is judged on honesty.

Solid today, with tests:

- The in-memory loop end to end: capture → attribute → curate → select uploads,
  from Rust and Python.
- Non-blocking capture safe to call inside a control loop.
- The attribution cascade with explicit method and confidence per binding, and
  fail-closed curation (weak evidence goes to review, never silently to training).
- MCAP recording, replay-window assembly, robot-type adapters, and LeRobot-style
  slice export.
- MCAP import: a recorded file plus a topic-mapping TOML becomes attributable rollouts
  and outcomes, with a `doctor` pre-check for topic coverage and clock skew. The reader
  is validated on a real rosbag2 MCAP (`libmcap`, ROS 2 profile, cdr).
- LeRobot import: `fieldloop import-lerobot` turns a recorded LeRobot dataset into
  rollouts and outcomes, validated end to end through the real loader on a public Hub
  dataset (`pusht_keypoints`: load to import to attribute).

Not yet, stated plainly:

- **Confidence is scored, not yet field-calibrated.** The default reports the raw
  recency/coverage score unchanged. A calibration curve fit from curator-confirmed
  labels (isotonic, per robot type and join method) is implemented and takes over
  once your team has confirmed enough bindings — until then a 0.9 means "close and
  well-covered", not "90% of such bindings proved correct".
- **Attribution is demonstrated on bundled and synthetic recordings, not yet on a public
  robot bag.** The reader handles real rosbag2 MCAP and the LeRobot import round-trips a
  real Hub dataset (above), but attributing a public bag end to end needs one with genuine
  decision and outcome topics, and most public bags are sensor-only. `fieldloop attribute
  run.mcap --map mapping.toml` works today, single-boot per file; a classic `.bag` must be
  converted to MCAP first. You can also feed decisions and outcomes as plain JSON Lines
  records (see the onboarding guide).
- The current public surface is for one robotics team working locally with its own
  fleet data. Hosted fleet operations, enterprise workflow, production RBAC, durable
  audit storage, fleet-wide reporting, and rollout-control operations are not
  presented as finished public surfaces here.

## Implemented components

- Captures rollout decision metadata from the control-loop path.
- Records MCAP streams for sensor, metadata, and safety channels.
- Stores payload pointers so large sensor artifacts can stay outside the metadata plane.
- Marks outcomes such as teleop takeover, e-stop, collision, downstream failure, and task success.
- Attributes delayed outcomes back to rollout decisions with confidence and method metadata.
- Builds incident evidence bundles with confirmed, hypothesis, and excluded evidence.
- Assembles replay timelines that point back to recorded MCAP message ranges.
- Defines robot-specific adapters for action normalization, success detection, and attribution windows.
- Exports curated slices for training and evaluation workflows, including LeRobot-style datasets.

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

