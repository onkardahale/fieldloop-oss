# Onboard your robot

You ran [`examples/loop.py`](../crates/fieldloop-py/examples/loop.py) on toy data. This
walks you from that demo to your robot. You do not need to read the internals or the ADRs.

The loop is four calls. Onboarding is: feed them your data instead of the toy data.

```
capture → attribute → curate → select_uploads
```

---

## Step A — Describe your embodiment

Attribution needs to know, per robot type, how far after a decision each kind of outcome
may still bind back to it. That lives in a small TOML config. Start from the sample:

```bash
cp crates/fieldloop-config/examples/embodiment.sample.toml my_robot.toml
```

Change these first; leave the rest until you need it:

- the embodiment table name (`[embodiments.<your_robot_type>]`) — this is the string you
  pass as `embodiment` when you capture;
- `action_space` — your control interface;
- the per-kind `window_ms` — how long after a decision an outcome of that kind may still
  be its consequence (a takeover can trail by seconds; a collision is tight);
- `requires_monotonic_colocation` — keep it `true` for tight-timing safety events
  (collision, e-stop) so they only bind when the decision and outcome share a boot and can
  be compared on the skew-free monotonic clock.

In `loop.py` the config is the inline `CONFIG_TOML` string — replace it with your file's
contents (or read the file in). The embodiment name there must match the one you capture
under.

---

## Step B — Capture your real decisions

In `loop.py`, `capture_decisions()` logs three toy steps. Replace them with your policy
loop. The contract that matters: **`log_step` is non-blocking** — it mints an id and
enqueues a compact record, then returns. It never waits on a slow consumer, so it is safe
to call inside a 50 Hz control loop.

```python
capture = fieldloop.Capture("your_team", "robot_01", capacity=1024)
ctx = capture.register_context(
    policy_version="your_policy@v3+<hash>",
    model_hash="sha256:<weights>",
    embodiment="your_robot_type",   # must match the config table name from Step A
    task_id="your_task",
)

# In your control loop:
#   1. compute the action
#   2. log_step(...)   <- compact, non-blocking; returns immediately
#   3. continue control
rollout_id = capture.log_step(episode_id, step_index=i, ctx=ctx, inference_us=measured_us)

# Off the hot path (a different thread / a timer), pull the records out:
rollouts = capture.drain()
```

`drain()` returns plain dicts carrying everything attribution needs (`rollout_id`,
`tenant_id`, `robot_id`, `boot_id`, `mono_ns`, `policy_version`, `embodiment`, …). The
demo pins `mono_ns` for reproducibility; on a real robot it is the capture clock — leave
it as drained.

---

## Step C — Feed in your real outcomes

An outcome is just a dict of what happened, when. The important part: **it does not need a
rollout id.** That is the whole point — a teleop takeover or e-stop arrives after the causing decision
with no pointer back, and FieldLoop reconstructs which decision it belongs to from the
clock and the attribution window.

```python
outcomes = [
    {
        "tenant_id": "your_team",
        "robot_id": "robot_01",
        "boot_id": rollout["boot_id"],     # same boot as the decisions it followed
        "mono_ns": event_mono_ns,          # when the event happened, on the same clock
        "outcome_kind": "teleop_takeover", # or e_stop | collision | downstream_failure
    },
]
```

`outcome_kind` is a closed set: `teleop_takeover`, `e_stop`, `collision`,
`downstream_failure`, and `heartbeat` (a coverage sample, not a failure). If a detector
*did* identify the exact decision, you may add `"explicit_rollout_id": "<rollout_id>"` —
that binds directly at confidence `1.0`. Most real outcomes will not have it.

---

## Step D — Run attribute and curate, and read the confidence

Nothing else changes — the same two calls now run on your data:

```python
attributed = fieldloop.attribute(open("my_robot.toml").read(), rollouts, outcomes)
for fb in attributed["feedbacks"]:
    print(fb["join_method"], fb["join_confidence"], "->", fb["target_id"])

curated = fieldloop.curate({"grain": "rollout", "min_confidence": 0.7}, rollouts,
                           attributed["feedbacks"])
print("accepted:", len(curated["items"]), "needs_review:", len(curated["needs_review"]))
```

Two things to watch, because they are the point of the system:

- **Confidence is scored, not asserted.** A binding far from its decision (near the
  window edge) comes back with low confidence. Move an outcome's `mono_ns` closer to a
  decision and watch the confidence rise. The default score is the raw recency/coverage
  signal; a calibration curve fit from your team's confirmed labels replaces it once
  enough ground truth exists.
- **Curation fails closed.** Raise or lower `min_confidence` and watch bindings move
  between the training slice (`items`) and `needs_review`. A label you are not confident in
  never silently enters training data.

---

## Native Rust integration

If your robot integration is Rust-native rather than Python, the same capture path and an
embodiment *adapter* (action normalization, success detection, attribution windows, a
feature schema) are shown in:

```bash
cargo run -p fieldloop-adapter --example onboard_your_robot
```

Use Python (the loop demo) to understand the semantics; use the Rust adapter when you wire
the integration into native robot code. Both speak the same schema.

---

## Where to go next

- [`quickstart.md`](quickstart.md) — the attribution cascade on its own, in Rust.
- [`concepts.md`](concepts.md) — the loop end to end: capture → join → curate → gate → deploy.
- [`adapter.md`](adapter.md) — the full adapter contract for a new robot type.
