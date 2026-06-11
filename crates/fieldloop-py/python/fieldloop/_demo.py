"""The bundled demo scenario: the whole OSS loop on toy data, in one module.

    capture -> attribute -> curate -> select_uploads

Lives inside the package (not only in `examples/`) so the installed `fieldloop demo`
command works with nothing but the wheel — no checkout, no example files. Everything
here uses the real `fieldloop` calls on toy data: the attribution is reconstructed,
never faked.

The scenario: a warehouse robot makes three decisions. Later, three things happen in
the world (a near-miss takeover, a collision, and a late takeover) — none of them
carrying a pointer back to the decision that caused it. FieldLoop reconstructs the
links with confidence, keeps only the trusted ones for training, and decides which raw
sensor payloads are worth pulling.
"""

import json
import uuid
from importlib import resources

import fieldloop

# Per-robot attribution config: how far after a decision each outcome kind may still bind
# back to it. Parsed and validated by the real Config loader. (This is the OSS schema —
# the same TOML the Rust engine reads.)
CONFIG_TOML = """
[embodiments.diff_drive_demo]
action_space = "joint_position"

[embodiments.diff_drive_demo.attribution.teleop_takeover]
window_ms = 250
requires_monotonic_colocation = false

[embodiments.diff_drive_demo.attribution.collision]
window_ms = 250
requires_monotonic_colocation = true

[calibration]
version = "demo-v1"
temporal_window_default_ms = 2000
heartbeat_coverage_k = 0.9
"""

# What the curation gate demands. grain is the one required choice; min_confidence is the
# floor below which a binding is held out of training rather than silently trusted.
CURATION_SPEC = {"grain": "rollout", "min_confidence": 0.70}

MS = 1_000_000  # nanoseconds per millisecond


def capture_decisions():
    """Stage 1 — capture three policy decisions on the robot, then drain them off the
    hot path into plain rollout dicts. This uses the real non-blocking capture SDK."""
    capture = fieldloop.Capture("demo_team", "robot_01", capacity=16)
    ctx = capture.register_context(
        policy_version="nav@v12+abc123abc123",
        model_hash="sha256:demo-weights",
        embodiment="diff_drive_demo",
        task_id="aisle_nav",
    )
    episode = str(uuid.uuid4())
    for step in range(3):
        capture.log_step(episode, step_index=step, ctx=ctx, inference_us=180)

    rollouts = capture.drain()

    # Pin a clean, evenly-spaced timeline so the demo is reproducible. In production these
    # `mono_ns` values come straight from the capture clock; here we set them (keeping the
    # real rollout ids and boot id) so the attribution windows land deterministically.
    # We also attach the on-robot pre-filter confidence the upload stage reads.
    decisions = ["slow_approach", "brake_before_turn", "no_recovery_action"]
    confidences = [0.15, 0.90, 0.08]
    for i, rollout in enumerate(rollouts):
        rollout["mono_ns"] = (1 + i) * 500 * MS  # 0.5s apart, comfortably > the 250ms window
        rollout["policy_confidence"] = confidences[i]
        rollout["action"] = decisions[i]  # demo-only label, ignored by the engine
    return rollouts, capture.dropped()


def observed_outcomes(rollouts):
    """The three things that happened later. Only the collision carries a rollout id
    (a detector localized it); the takeovers arrive with NO link — the cascade has to
    reconstruct which decision they belong to."""
    r0, r1, r2 = rollouts
    base = dict(tenant_id="demo_team", robot_id="robot_01", boot_id=r0["boot_id"])
    return [
        # A near-miss takeover 20ms after decision 0 — close, so high-confidence temporal.
        {**base, "mono_ns": r0["mono_ns"] + 20 * MS, "outcome_kind": "teleop_takeover"},
        # A collision a detector tied to decision 1 — explicit link, confidence 1.0.
        {
            **base,
            "mono_ns": r1["mono_ns"] + 30 * MS,
            "outcome_kind": "collision",
            "explicit_rollout_id": r1["rollout_id"],
        },
        # A takeover 190ms after decision 2 — near the window edge, so LOW confidence.
        {**base, "mono_ns": r2["mono_ns"] + 190 * MS, "outcome_kind": "teleop_takeover"},
    ]


def run_mcap_demo():
    """Attribute the bundled demo MCAP through the real file-import path.

    The in-process loop above feeds the engine dicts; this path feeds it an actual
    finalized MCAP binary (`data/demo.mcap`) plus its topic-mapping, exercising
    `attribute_mcap` end to end. The bundled file holds three decisions and an e-stop
    200ms after the last, which the engine binds via the temporal tier."""
    data = resources.files("fieldloop").joinpath("data")
    config_toml = data.joinpath("embodiment.sample.toml").read_text(encoding="utf-8")
    mapping_toml = data.joinpath("demo.map.toml").read_text(encoding="utf-8")
    mcap_bytes = data.joinpath("demo.mcap").read_bytes()
    return fieldloop.attribute_mcap(config_toml, mcap_bytes, mapping_toml)


def run_loop():
    """The whole loop, returning every stage's real result."""
    rollouts, dropped = capture_decisions()
    outcomes = observed_outcomes(rollouts)

    # Stage 2 — attribute outcomes back to the decisions that caused them.
    attributed = fieldloop.attribute(CONFIG_TOML, rollouts, outcomes)

    # Stage 3 — curate: pin the trusted bindings into a training slice, hold out the rest.
    curated = fieldloop.curate(CURATION_SPEC, rollouts, attributed["feedbacks"])

    # Stage 4 — select which payloads to pull, under a tight budget. A safety (reflex)
    # trigger is kept even at a zero budget; the risky-but-unconfirmed ones are dropped.
    uploads = fieldloop.select_uploads(rollouts, outcomes, max_requests=0)

    return {
        "rollouts": rollouts,
        "dropped": dropped,
        "attribute": attributed,
        "curate": curated,
        "select_uploads": uploads,
        # The same engine, run on a real MCAP file rather than in-process dicts.
        "mcap_demo": run_mcap_demo(),
    }


def summarize(result):
    """The stable, machine-readable counts — safe to snapshot in CI (no decimals)."""
    return {
        "capture": {"rollouts": len(result["rollouts"]), "dropped": result["dropped"]},
        "attribute": {
            "feedbacks": len(result["attribute"]["feedbacks"]),
            "skipped": len(result["attribute"]["skipped"]),
        },
        "curate": {
            "accepted": len(result["curate"]["items"]),
            "needs_review": len(result["curate"]["needs_review"]),
        },
        "select_uploads": {
            "selected": len(result["select_uploads"]["requests"]),
            "dropped": len(result["select_uploads"]["dropped"]),
        },
        "mcap_demo": {
            "attributed": len(result["mcap_demo"]["feedbacks"]),
            "unattributed": len(result["mcap_demo"]["skipped"]),
        },
    }


def print_human(result):
    rollouts = result["rollouts"]
    decision_of = {r["rollout_id"]: r["action"] for r in rollouts}

    print("FieldLoop local loop demo")
    print("no database to set up · no cloud · everything runs locally in-process")
    print()

    print("1. CAPTURE")
    print(f"   captured {len(rollouts)} policy decisions")
    print(f"   drained {len(rollouts)} rollout records")
    print(f"   dropped {result['dropped']}")
    print()

    feedbacks = result["attribute"]["feedbacks"]
    # Derive the counts so this line stays true if the scenario is edited: an explicit
    # binding is one whose outcome carried a rollout id; the rest arrived with none and
    # had their link reconstructed.
    received = len(feedbacks) + len(result["attribute"]["skipped"])
    no_id = sum(1 for fb in feedbacks if fb["join_method"] != "explicit")
    print("2. ATTRIBUTE")
    print(f"   received {received} outcomes ({no_id} with no rollout_id — the link is reconstructed)")
    print(f"   attributed {len(feedbacks)} outcomes:")
    for fb in feedbacks:
        decision = decision_of.get(fb["target_id"], fb["target_id"])
        print(
            f"   - {fb['join_method']:<18} confidence={fb['join_confidence']:.2f}"
            f"  decision={decision}"
        )
    print()

    curate = result["curate"]
    print("3. CURATE")
    print("   training slice:")
    print(f"   - accepted {len(curate['items'])} feedbacks")
    print(f"   - needs_review {len(curate['needs_review'])} feedbacks")
    for item in curate["needs_review"]:
        print(f"     reason: {item['reason']}")
    print()

    uploads = result["select_uploads"]
    print("4. SELECT UPLOADS  (budget: 0 — only safety may bypass it)")
    print(f"   selected {len(uploads['requests'])} payload request")
    for request in uploads["requests"]:
        print(f"   - {request['detector_id']}  tier={request['tier']}")
    print(f"   budget drops returned: {len(uploads['dropped'])}")
    print()

    mcap = result["mcap_demo"]
    print("5. ATTRIBUTE FROM AN MCAP FILE  (the bundled demo.mcap, the file-import path)")
    feedbacks = mcap["feedbacks"]
    print(
        f"   MCAP demo: {len(feedbacks)} bindings — a teleop takeover and two e-stop"
        " events traced to their decisions (one e-stop splits credit across a"
        " near-simultaneous pair):"
    )
    for fb in feedbacks:
        print(
            f"   - {fb['join_method']:<9} confidence={fb['join_confidence']:.2f}"
            f"  delay={fb['delay_ms']}ms  credit={fb['credit_weight']:.2f}"
        )
    for skip in mcap["skipped"]:
        print(
            f"   refused: a {skip['outcome_kind']} outside its window stays"
            f" unattributed ({skip['reason']}) — never guessed"
        )
    print(
        "   (same engine as above, run on a real MCAP binary — try it on yours:"
        " fieldloop attribute run.mcap --map mapping.toml --config embodiment.toml)"
    )
    print()

    print("Loop complete.")
    print(
        "You just saw: decisions -> outcomes -> attributed feedback (with confidence)"
        " -> curated training slice -> upload selection -> the same attribution on a real"
        " MCAP file."
    )


def run(json_output=False, viz=False):
    """Run the demo end to end and print it — the entry the CLI and example share.

    With `viz`, the MCAP-demo incidents also open on a Rerun timeline (the optional
    `viz` extra); a missing extra is reported after the demo output, exit 1."""
    result = run_loop()
    if json_output:
        print(json.dumps(summarize(result), indent=2, sort_keys=True))
    else:
        print_human(result)
    if viz:
        import sys

        from fieldloop import _report, _viz

        incidents = _report.build_incidents(
            result["mcap_demo"]["feedbacks"], result["mcap_demo"]["skipped"]
        )
        try:
            _viz.log_incidents(incidents, spawn=True)
        except ImportError as e:
            print(f"fieldloop: error: {e}", file=sys.stderr)
            return 1
    return 0
