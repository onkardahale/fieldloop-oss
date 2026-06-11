"""CLI behavior, asserted through the real entry point (`fieldloop._cli.main`).

Runs in-process rather than via subprocess so failures give a Python traceback, not
an opaque exit code — and so the suite exercises exactly what the console script
wrapper calls. File-shape errors and exit codes are part of the public contract here:
scripts will branch on them, so they are pinned by test.
"""

import json
from pathlib import Path

import pytest

from fieldloop import _cli, _demo


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def demo_files(tmp_path):
    """Real rollouts/outcomes (from the demo scenario) serialized the way a user's
    log-conversion script would produce them, plus the demo's attribution config."""
    rollouts, _ = _demo.capture_decisions()
    outcomes = _demo.observed_outcomes(rollouts)
    config = tmp_path / "config.toml"
    config.write_text(_demo.CONFIG_TOML, encoding="utf-8")
    r_path = tmp_path / "rollouts.jsonl"
    o_path = tmp_path / "outcomes.jsonl"
    _write_jsonl(r_path, rollouts)
    _write_jsonl(o_path, outcomes)
    return config, r_path, o_path


def test_demo_runs_end_to_end(capsys):
    assert _cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Loop complete." in out
    assert "needs_review 1" in out  # the fail-closed holdout is the demo's point


def test_demo_json_is_stable(capsys):
    assert _cli.main(["demo", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["attribute"]["feedbacks"] == 3
    assert summary["curate"] == {"accepted": 2, "needs_review": 1}


def test_demo_includes_mcap_attribution(capsys):
    """The demo also runs the bundled real MCAP through the file-import path, so a fresh
    install sees attribution working on an actual MCAP binary, not only the in-process loop."""
    assert _cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "MCAP demo:" in out
    assert "e-stop" in out


def test_demo_json_includes_mcap_demo(capsys):
    assert _cli.main(["demo", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mcap_demo"] == {"attributed": 4, "unattributed": 1}


def test_demo_viz_flag_opens_the_timeline(monkeypatch, capsys):
    """`demo --viz` hands the MCAP-demo incidents to the Rerun path with spawn=True. The
    viewer itself is monkeypatched: the test pins the wiring, not the window."""
    calls = []

    def fake_log_incidents(incidents, **kwargs):
        calls.append((incidents, kwargs))

    from fieldloop import _viz

    monkeypatch.setattr(_viz, "log_incidents", fake_log_incidents)
    assert _cli.main(["demo", "--viz"]) == 0
    assert "Loop complete." in capsys.readouterr().out
    assert len(calls) == 1
    incidents, kwargs = calls[0]
    assert kwargs.get("spawn") is True
    states = [i["state"] for i in incidents]
    assert states.count("attributed") == 4 and states.count("unattributed") == 1


def test_demo_viz_without_rerun_is_a_clean_error(monkeypatch, capsys):
    """Without the extra, `demo --viz` still prints the demo, then reports the install
    line and exits non-zero — never an ImportError traceback."""
    import sys

    monkeypatch.setitem(sys.modules, "rerun", None)
    code = _cli.main(["demo", "--viz"])
    captured = capsys.readouterr()
    assert code == _cli.EXIT_ENGINE_ERROR
    assert "Loop complete." in captured.out  # the demo itself still ran
    assert "fieldloop[viz]" in captured.err


def test_attribute_then_curate_roundtrip(demo_files, tmp_path, capsys):
    """The file-based pipeline mirrors the in-process one: 3 bindings, then the
    sub-threshold one held out by the curation gate."""
    config, r_path, o_path = demo_files
    fb_path = tmp_path / "feedbacks.jsonl"

    code = _cli.main(
        [
            "attribute",
            "--config",
            str(config),
            "--rollouts",
            str(r_path),
            "--outcomes",
            str(o_path),
            "--out",
            str(fb_path),
        ]
    )
    assert code == 0
    assert "attributed 3 outcomes" in capsys.readouterr().out
    feedbacks = [json.loads(line) for line in fb_path.read_text().splitlines()]
    assert len(feedbacks) == 3

    slice_path = tmp_path / "slice.json"
    code = _cli.main(
        [
            "curate",
            "--rollouts",
            str(r_path),
            "--feedbacks",
            str(fb_path),
            "--out",
            str(slice_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "accepted 2 · needs_review 1" in out
    result = json.loads(slice_path.read_text())
    assert len(result["items"]) == 2


def test_attribute_missing_file_is_exit_2(tmp_path, capsys):
    code = _cli.main(
        [
            "attribute",
            "--config",
            str(tmp_path / "nope.toml"),
            "--rollouts",
            str(tmp_path / "nope.jsonl"),
            "--outcomes",
            str(tmp_path / "nope.jsonl"),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    assert "error:" in capsys.readouterr().err


def test_attribute_malformed_jsonl_names_the_line(demo_files, tmp_path, capsys):
    config, r_path, _ = demo_files
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    code = _cli.main(
        [
            "attribute",
            "--config",
            str(config),
            "--rollouts",
            str(r_path),
            "--outcomes",
            str(bad),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    assert "line 2" in capsys.readouterr().err


def test_attribute_out_to_unwritable_path_is_clean_exit_2(demo_files, tmp_path, capsys):
    """A successful attribution whose --out write fails must still exit cleanly with
    code 2, not crash with a traceback — the work printed, but a downstream script
    must not march past a file that was never written."""
    config, r_path, o_path = demo_files
    code = _cli.main(
        [
            "attribute",
            "--config",
            str(config),
            "--rollouts",
            str(r_path),
            "--outcomes",
            str(o_path),
            "--out",
            str(tmp_path / "no_such_dir" / "x.jsonl"),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    captured = capsys.readouterr()
    assert "attributed 3 outcomes" in captured.out  # the work still surfaced
    assert "error:" in captured.err  # the failure was reported, not raised


def test_attribute_bad_config_is_engine_error(demo_files, tmp_path):
    _, r_path, o_path = demo_files
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text("[embodiments]\n", encoding="utf-8")  # missing calibration
    code = _cli.main(
        [
            "attribute",
            "--config",
            str(bad_config),
            "--rollouts",
            str(r_path),
            "--outcomes",
            str(o_path),
        ]
    )
    assert code == _cli.EXIT_ENGINE_ERROR


def test_curate_out_to_unwritable_path_is_clean_exit_2(demo_files, tmp_path, capsys):
    """Same write-failure contract as attribute: the gate result printed, but an --out
    write into a missing directory exits 2 cleanly rather than raising."""
    config, r_path, o_path = demo_files
    fb_path = tmp_path / "fb.jsonl"
    assert _cli.main(
        ["attribute", "--config", str(config), "--rollouts", str(r_path),
         "--outcomes", str(o_path), "--out", str(fb_path)]
    ) == 0
    capsys.readouterr()
    code = _cli.main(
        ["curate", "--rollouts", str(r_path), "--feedbacks", str(fb_path),
         "--out", str(tmp_path / "no_such_dir" / "slice.json")]
    )
    assert code == _cli.EXIT_BAD_FILE
    captured = capsys.readouterr()
    assert "accepted" in captured.out  # the work still surfaced
    assert "error:" in captured.err


def test_init_out_to_unwritable_path_is_clean_exit_2(tmp_path, capsys):
    """init writes nothing it cannot write: a missing parent directory is reported as
    a clean exit 2, not a traceback."""
    code = _cli.main(["init", "--out", str(tmp_path / "no_such_dir" / "robot.toml")])
    assert code == _cli.EXIT_BAD_FILE
    assert "error:" in capsys.readouterr().err


def test_init_writes_sample_and_refuses_overwrite(tmp_path):
    dest = tmp_path / "my_robot.toml"
    assert _cli.main(["init", "--out", str(dest)]) == 0
    text = dest.read_text(encoding="utf-8")
    assert "[embodiments.warehouse_pick_arm]" in text
    assert "[calibration]" in text

    # Refuses to clobber a hand-edited config without --force.
    assert _cli.main(["init", "--out", str(dest)]) == _cli.EXIT_BAD_FILE
    assert _cli.main(["init", "--out", str(dest), "--force"]) == 0


def _bundled(name):
    """Resolve a packaged data file to a real filesystem path for the CLI to read."""
    from importlib.resources import as_file, files

    return as_file(files("fieldloop").joinpath(f"data/{name}"))


def test_attribute_mcap_binds_estop_to_a_decision(tmp_path, capsys):
    """The file-import path on the bundled demo MCAP: the e-stop binds back to the
    decision it followed, producing one attributed incident and zero unattributed."""
    inc_path = tmp_path / "incidents.jsonl"
    report_path = tmp_path / "report.md"
    with _bundled("demo.mcap") as mcap, _bundled("demo.map.toml") as mapping, _bundled(
        "embodiment.sample.toml"
    ) as config:
        code = _cli.main(
            [
                "attribute",
                str(mcap),
                "--map",
                str(mapping),
                "--config",
                str(config),
                "--out",
                str(inc_path),
                "--report",
                str(report_path),
            ]
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "4 attributed, 1 unattributed" in out

    incidents = [json.loads(line) for line in inc_path.read_text().splitlines()]
    assert len(incidents) == 5
    attributed = [i for i in incidents if i["state"] == "attributed"]
    refused = [i for i in incidents if i["state"] == "unattributed"]
    # No explicit ids ride the file, so every binding comes from the temporal tier; the
    # out-of-window collision is refused with the engine's own reason, never guessed.
    assert len(attributed) == 4
    assert all(i["join_method"] == "temporal" for i in attributed)
    assert len(refused) == 1
    assert refused[0]["reason"] == "no_candidate_in_window"

    report = report_path.read_text(encoding="utf-8")
    assert "# FieldLoop attribution report" in report
    # The honesty rail must survive into the shipped report.
    assert "not a field-calibrated probability" in report
    assert "outcomes attributed: 4" in report


def test_attribute_mcap_without_map_is_exit_2(capsys):
    with _bundled("demo.mcap") as mcap, _bundled("embodiment.sample.toml") as config:
        code = _cli.main(["attribute", str(mcap), "--config", str(config)])
    assert code == _cli.EXIT_BAD_FILE
    assert "--map" in capsys.readouterr().err


def test_attribute_with_no_inputs_is_exit_2(tmp_path, capsys):
    """Neither an MCAP source nor JSONL inputs: a clear error, not a crash."""
    config = tmp_path / "config.toml"
    config.write_text(_demo.CONFIG_TOML, encoding="utf-8")
    code = _cli.main(["attribute", "--config", str(config)])
    assert code == _cli.EXIT_BAD_FILE
    assert "error:" in capsys.readouterr().err


def test_doctor_clean_demo_is_ok(capsys):
    """The bundled demo mapping matches the bundled demo MCAP and its clocks agree."""
    with _bundled("demo.mcap") as mcap, _bundled("demo.map.toml") as mapping:
        code = _cli.main(["doctor", str(mcap), "--map", str(mapping)])
    assert code == 0
    out = capsys.readouterr().out
    assert "ok:" in out
    # every demo topic is classified.
    assert "/policy/action" in out
    assert "/safety/estop" in out


def test_doctor_flags_a_declared_topic_absent_from_the_file(tmp_path, capsys):
    """A mapping that declares an outcome topic the file does not contain is the common
    onboarding mistake; doctor names it and exits nonzero."""
    mapping = tmp_path / "bad.map.toml"
    mapping.write_text(
        """
tenant_id = "demo"
robot_id = "demo-arm-01"
boot_id = "00000000-0000-7000-8000-0000000000d0"
episode_id = "00000000-0000-7000-8000-0000000000e0"
embodiment = "warehouse_pick_arm"
policy_version = "demo-policy-v1"
task_id = "pick-place"
[[decisions]]
topic = "/policy/action"
[[outcomes]]
topic = "/gripper/fault"
outcome_kind = "collision"
""",
        encoding="utf-8",
    )
    with _bundled("demo.mcap") as mcap:
        code = _cli.main(["doctor", str(mcap), "--map", str(mapping)])
    assert code == _cli.EXIT_DOCTOR_PROBLEMS
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "/gripper/fault" in out


def test_view_writes_rrd_when_rerun_present(tmp_path):
    """With the optional 'viz' extra installed, `view --save` writes a real .rrd. Skips
    where rerun is absent (the gate's dev env); the real path is also verified manually."""
    pytest.importorskip("rerun")
    incidents = tmp_path / "incidents.jsonl"
    incidents.write_text(
        '{"state": "attributed", "outcome_id": "o1", "join_method": "temporal", '
        '"bound_target_id": "d2", "confidence": 0.6}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.rrd"
    assert _cli.main(["view", str(incidents), "--save", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_view_without_rerun_is_a_clean_error(tmp_path, capsys, monkeypatch):
    """Without the extra, `view` reports the install line and exits non-zero — never a
    raw ImportError traceback."""
    import sys

    # Force `import rerun` to raise even if the extra happens to be installed.
    monkeypatch.setitem(sys.modules, "rerun", None)
    incidents = tmp_path / "incidents.jsonl"
    incidents.write_text('{"state": "attributed", "outcome_id": "o1"}\n', encoding="utf-8")
    code = _cli.main(["view", str(incidents)])
    assert code == _cli.EXIT_ENGINE_ERROR
    assert 'fieldloop[viz]' in capsys.readouterr().err


_LEROBOT_CONFIG = """
[embodiments.lerobot_bot]
action_space = "joint_position"
[embodiments.lerobot_bot.attribution.downstream_failure]
window_ms = 5000
requires_monotonic_colocation = true
[calibration]
version = "lerobot-v1"
temporal_window_default_ms = 2000
heartbeat_coverage_k = 0.9
"""


def test_lerobot_roundtrip_load_import_attribute(tmp_path):
    """Acceptance: a LeRobotDataset built via the REAL lerobot API round-trips
    load -> import -> attribute end to end. Skips where the heavy 'lerobot' extra is absent
    (the gate's dev env); verified manually with --extra lerobot."""
    pytest.importorskip("lerobot")
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import fieldloop
    from fieldloop import _lerobot

    root = tmp_path / "ds"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["dx", "dy"]},
    }
    ds = LeRobotDataset.create(
        repo_id="fieldloop/test", fps=10, root=str(root), features=features, use_videos=False
    )
    for ep in range(2):
        for i in range(3):
            ds.add_frame(
                {
                    "observation.state": np.array([float(i), float(ep)], dtype=np.float32),
                    "action": np.array([1.0, 0.0], dtype=np.float32),
                    "task": "pick",
                }
            )
        ds.save_episode()
    ds.finalize()

    loaded = _lerobot.load_dataset("fieldloop/test", str(root))
    result = _lerobot.import_dataset(
        loaded,
        tenant_id="t",
        robot_id="r",
        policy_version="v",
        embodiment="lerobot_bot",
        terminal_outcome="downstream_failure",
    )
    assert len(result["rollouts"]) == 6  # 2 episodes x 3 frames
    assert len(result["outcomes"]) == 2  # one terminal outcome per episode

    # The imported records attribute end to end through the real engine. Both episode
    # terminal outcomes are attributed; the engine distributes credit across the in-window
    # decisions of each episode, so there are >= 2 feedbacks, every one pointing at an
    # imported rollout.
    report = fieldloop.attribute(_LEROBOT_CONFIG, result["rollouts"], result["outcomes"])
    feedbacks = report["feedbacks"]
    assert {f["source_outcome_id"] for f in feedbacks} == {
        o["outcome_id"] for o in result["outcomes"]
    }
    rollout_ids = {r["rollout_id"] for r in result["rollouts"]}
    assert feedbacks and all(f["target_id"] in rollout_ids for f in feedbacks)


def test_import_lerobot_without_extra_is_a_clean_error(tmp_path, capsys, monkeypatch):
    """Without the extra, `import-lerobot` reports the install line and exits non-zero."""
    import sys

    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", None)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)
    code = _cli.main(
        [
            "import-lerobot",
            str(tmp_path),
            "--repo-id",
            "x/y",
            "--tenant-id",
            "t",
            "--robot-id",
            "r",
            "--policy-version",
            "v",
            "--embodiment",
            "e",
            "--terminal-outcome",
            "task_success",
            "--out",
            str(tmp_path / "r.jsonl"),
            "--outcomes-out",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert code == _cli.EXIT_ENGINE_ERROR
    assert "fieldloop[lerobot]" in capsys.readouterr().err


def test_attribute_binary_config_is_exit_2(tmp_path, capsys):
    config = tmp_path / "bin.toml"
    config.write_bytes(b"\x89MCAP0\r\n\xff\xfe")
    code = _cli.main(
        [
            "attribute",
            "--config",
            str(config),
            "--rollouts",
            str(tmp_path / "r.jsonl"),
            "--outcomes",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    assert "error:" in capsys.readouterr().err


def test_doctor_negative_threshold_is_exit_2(capsys):
    with _bundled("demo.mcap") as mcap, _bundled("demo.map.toml") as mapping:
        code = _cli.main(
            [
                "doctor",
                str(mcap),
                "--map",
                str(mapping),
                "--clock-skew-threshold-ns",
                "-1",
            ]
        )
    assert code == _cli.EXIT_BAD_FILE
    assert "error:" in capsys.readouterr().err


def test_attribute_conflicting_inputs_is_exit_2(tmp_path, capsys):
    code = _cli.main(
        [
            "attribute",
            str(tmp_path / "run.mcap"),
            "--config",
            str(tmp_path / "config.toml"),
            "--map",
            str(tmp_path / "m.toml"),
            "--rollouts",
            str(tmp_path / "r.jsonl"),
            "--outcomes",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    assert "not both" in capsys.readouterr().err


def test_import_lerobot_dataset_error_is_exit_2(tmp_path, capsys, monkeypatch):
    from fieldloop import _lerobot

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(_lerobot, "load_dataset", _boom)
    code = _cli.main(
        [
            "import-lerobot",
            str(tmp_path),
            "--repo-id",
            "x/y",
            "--tenant-id",
            "t",
            "--robot-id",
            "r",
            "--policy-version",
            "v",
            "--embodiment",
            "e",
            "--terminal-outcome",
            "task_success",
            "--out",
            str(tmp_path / "r.jsonl"),
            "--outcomes-out",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert code == _cli.EXIT_BAD_FILE
    assert "could not import dataset" in capsys.readouterr().err


def test_doctor_flags_the_bundled_skewed_recording(capsys):
    """demo-skew.mcap carries a sensor topic publishing 32s behind its log time; doctor
    must flag it and exit nonzero."""
    with _bundled("demo-skew.mcap") as mcap, _bundled("demo.map.toml") as mapping:
        code = _cli.main(["doctor", str(mcap), "--map", str(mapping)])
    assert code == _cli.EXIT_DOCTOR_PROBLEMS
    out = capsys.readouterr().out
    assert "CLOCK SKEW" in out
    assert "/range/front" in out


def test_packaged_sample_matches_config_crate_sample():
    """The packaged scaffold must stay byte-identical to the config crate's sample —
    they document the same schema, and drift would mean `fieldloop init` teaches a
    different config than the Rust docs do."""
    from importlib import resources

    packaged = (
        resources.files("fieldloop")
        .joinpath("data/embodiment.sample.toml")
        .read_text(encoding="utf-8")
    )
    crate_sample = (
        Path(__file__).parents[2] / "fieldloop-config" / "examples" / "embodiment.sample.toml"
    )
    if not crate_sample.exists():
        pytest.skip("config crate sample not present in this tree")
    assert packaged == crate_sample.read_text(encoding="utf-8")
