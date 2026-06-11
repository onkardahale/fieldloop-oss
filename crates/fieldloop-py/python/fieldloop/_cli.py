"""The `fieldloop` command-line front door.

Four subcommands, mirroring the loop's stages on files instead of in-process dicts:

    fieldloop demo                          # the bundled toy-data loop, end to end
    fieldloop init [--out my_robot.toml]    # scaffold an embodiment config to edit
    fieldloop attribute run.mcap --map mapping.toml --config embodiment.toml  # MCAP in
    fieldloop attribute --config ... --rollouts ... --outcomes ...   # JSONL in
    fieldloop curate --rollouts ... --feedbacks ...                  # gate evidence

`attribute` takes either an MCAP recording (a positional file plus a `--map`
topic-mapping TOML) and writes an incidents JSONL + a Markdown report, or the plain
JSON Lines inputs (one dict per line for rollouts/outcomes/feedbacks — the exact dict
shapes `fieldloop.attribute`/`curate` take) so any logging stack can produce them with a
few lines of glue and no SDK.

Exit codes are scriptable: 0 success, 1 the engine rejected the inputs (e.g. invalid
config), 2 the files themselves were unreadable or malformed.
"""

import argparse
import json
import sys
from importlib import metadata, resources
from pathlib import Path

# The engine rejected semantically-bad input (bad config, bad shapes): the caller's
# data is wrong. Distinct from EXIT_BAD_FILE so scripts can tell "fix my data" from
# "fix my paths/encoding".
EXIT_ENGINE_ERROR = 1
# The input files were missing or not parseable at all.
EXIT_BAD_FILE = 2
# `doctor` ran fine but found problems (a declared topic missing from the file, or a
# topic on a divergent clock). A distinct meaning from EXIT_ENGINE_ERROR despite the
# shared value: nonzero so a CI step or onboarding script halts on a bad mapping/file.
EXIT_DOCTOR_PROBLEMS = 1


def _fail(code, message):
    print(f"fieldloop: error: {message}", file=sys.stderr)
    return code


def _read_jsonl(path):
    """Read a JSON Lines file into a list of dicts, naming the exact offending line on
    failure — 'line 37: expecting value' beats a bare traceback when the file came out
    of someone's log-conversion script."""
    rows = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: line {lineno}: {e.msg}") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {lineno}: expected a JSON object")
        rows.append(row)
    return rows


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _cmd_demo(args):
    from fieldloop import _demo

    return _demo.run(json_output=args.json, viz=args.viz)


def _cmd_init(args):
    """Write the packaged embodiment sample config for the user to edit. Refuses to
    overwrite without --force: the file is meant to be hand-edited, so clobbering it
    silently could destroy someone's tuned attribution windows."""
    dest = Path(args.out)
    if dest.exists() and not args.force:
        return _fail(EXIT_BAD_FILE, f"{dest} already exists (use --force to overwrite)")
    sample = (
        resources.files("fieldloop")
        .joinpath("data/embodiment.sample.toml")
        .read_text(encoding="utf-8")
    )
    try:
        dest.write_text(sample, encoding="utf-8")
    except OSError as e:
        return _fail(EXIT_BAD_FILE, str(e))
    print(f"wrote {dest}")
    print("edit the embodiment name, action_space, and per-kind window_ms, then:")
    print(f"  fieldloop attribute --config {dest} --rollouts r.jsonl --outcomes o.jsonl")
    return 0


def _cmd_attribute(args):
    """Two input modes share one subcommand. An MCAP file (positional `source`, with
    `--map`) is the file-import path; `--rollouts`/`--outcomes` is the JSONL path. The
    config is needed by both, so it is read once here before the mode split."""
    if args.source is not None and (args.rollouts or args.outcomes):
        return _fail(EXIT_BAD_FILE, "pass an MCAP source or --rollouts/--outcomes, not both")
    try:
        config_toml = Path(args.config).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    if args.source is not None:
        return _attribute_mcap(args, config_toml)
    if args.rollouts and args.outcomes:
        return _attribute_jsonl(args, config_toml)
    return _fail(
        EXIT_BAD_FILE,
        "provide an MCAP file with --map, or both --rollouts and --outcomes",
    )


def _attribute_jsonl(args, config_toml):
    """The dict-in path: rollout/outcome JSONL through `fieldloop.attribute`, writing
    feedbacks to --out."""
    try:
        rollouts = _read_jsonl(args.rollouts)
        outcomes = _read_jsonl(args.outcomes)
    except (OSError, ValueError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    import fieldloop

    try:
        result = fieldloop.attribute(config_toml, rollouts, outcomes)
    except ValueError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))

    feedbacks = result["feedbacks"]
    skipped = result["skipped"]
    print(f"attributed {len(feedbacks)} outcomes ({len(skipped)} skipped):")
    for fb in feedbacks:
        print(
            f"  - {fb['join_method']:<18} confidence={fb['join_confidence']:.2f}"
            f"  -> {fb['target_id']}"
        )
    for skip in skipped:
        print(f"  skipped: {skip.get('reason', skip)}")

    # The engine already printed its result above, so an --out write failure here
    # loses nothing — but it must still exit non-zero (and cleanly), or a script
    # piping `--out` into a later stage would march on past a file that was never
    # written.
    if args.out:
        try:
            _write_jsonl(args.out, feedbacks)
        except OSError as e:
            return _fail(EXIT_BAD_FILE, str(e))
        print(f"wrote {len(feedbacks)} feedbacks to {args.out}")
    return 0


def _attribute_mcap(args, config_toml):
    """The file-import path: an MCAP recording + a topic-mapping TOML through
    `fieldloop.attribute_mcap`, writing an incidents JSONL (--out) and/or a Markdown
    report (--report)."""
    if not args.mapping:
        return _fail(
            EXIT_BAD_FILE,
            "an MCAP source needs --map pointing at a topic-mapping TOML",
        )
    try:
        mapping_toml = Path(args.mapping).read_text(encoding="utf-8")
        mcap_bytes = Path(args.source).read_bytes()
    except (OSError, UnicodeDecodeError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    import fieldloop
    from fieldloop import _report

    try:
        result = fieldloop.attribute_mcap(config_toml, mcap_bytes, mapping_toml)
    except ValueError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))

    feedbacks = result["feedbacks"]
    skipped = result["skipped"]
    incidents = _report.build_incidents(feedbacks, skipped)
    print(
        f"imported {args.source}: {len(feedbacks)} attributed, {len(skipped)} unattributed"
    )
    for inc in incidents:
        if inc["state"] == "attributed":
            conf = inc["confidence"]
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
            print(
                f"  - {inc['join_method']:<18} confidence={conf_str}"
                f"  outcome {inc['outcome_id']} -> {inc['bound_target_id']}"
            )
        else:
            print(f"  unattributed: {inc.get('reason')}  ({inc.get('outcome_kind')})")

    # As in the JSONL path, the result already printed, so a write failure must still
    # exit cleanly and non-zero rather than crash a downstream script.
    if args.out:
        try:
            _write_jsonl(args.out, incidents)
        except OSError as e:
            return _fail(EXIT_BAD_FILE, str(e))
        print(f"wrote {len(incidents)} incidents to {args.out}")
    if args.report:
        try:
            Path(args.report).write_text(
                _report.build_report_md(incidents), encoding="utf-8"
            )
        except OSError as e:
            return _fail(EXIT_BAD_FILE, str(e))
        print(f"wrote report to {args.report}")
    return 0


def _cmd_doctor(args):
    """Check an MCAP file against a topic-mapping before attribution: report each topic's
    role + clock skew, list declared-but-absent topics and divergent-clock topics, and
    exit nonzero if any problem was found so a script can halt on a bad mapping/file."""
    if args.clock_skew_threshold_ns is not None and args.clock_skew_threshold_ns < 0:
        return _fail(EXIT_BAD_FILE, "clock-skew threshold must be >= 0")
    try:
        mapping_toml = Path(args.mapping).read_text(encoding="utf-8")
        mcap_bytes = Path(args.source).read_bytes()
    except (OSError, UnicodeDecodeError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    import fieldloop

    try:
        report = fieldloop.doctor(
            mcap_bytes,
            mapping_toml,
            clock_skew_threshold_ns=args.clock_skew_threshold_ns,
        )
    except ValueError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))

    print(f"doctor: {args.source}")
    for topic in report["topics"]:
        skew_s = topic["max_clock_skew_ns"] / 1e9
        print(
            f"  {topic['role']:<9} {topic['topic']}  "
            f"({topic['message_count']} msgs, max clock skew {skew_s:.3f}s)"
        )
    if report["missing_topics"]:
        print("  MISSING — declared in the mapping but absent from the file:")
        for topic in report["missing_topics"]:
            print(f"    - {topic}")
    if report["skewed_topics"]:
        threshold_s = report["clock_skew_threshold_ns"] / 1e9
        print(
            f"  CLOCK SKEW > {threshold_s:.3f}s — source clock diverges from the recorder "
            "clock (a sensor on a different time base):"
        )
        for topic in report["skewed_topics"]:
            print(f"    - {topic}")

    if report["ok"]:
        print("ok: every declared topic is present and all clocks agree")
        return 0
    return _fail(EXIT_DOCTOR_PROBLEMS, "doctor found problems (see above)")


def _cmd_curate(args):
    try:
        rollouts = _read_jsonl(args.rollouts)
        feedbacks = _read_jsonl(args.feedbacks)
    except (OSError, ValueError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    import fieldloop

    spec = {"grain": args.grain, "min_confidence": args.min_confidence}
    try:
        result = fieldloop.curate(spec, rollouts, feedbacks)
    except ValueError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))

    items = result["items"]
    review = result["needs_review"]
    print(f"accepted {len(items)} · needs_review {len(review)}")
    for held in review:
        print(f"  needs_review: {held.get('reason', '')}  target={held.get('target_id', '')}")

    if args.out:
        try:
            Path(args.out).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as e:
            return _fail(EXIT_BAD_FILE, str(e))
        print(f"wrote slice to {args.out}")
    return 0


def _cmd_view(args):
    """Render an incidents JSONL on a Rerun timeline. Needs the optional `viz` extra; if
    Rerun is absent, the import raises and we report the install line as a clean exit, not
    a traceback."""
    try:
        incidents = _read_jsonl(args.incidents)
    except (OSError, ValueError) as e:
        return _fail(EXIT_BAD_FILE, str(e))

    from fieldloop import _viz

    try:
        # No --save: open the live viewer. With --save: write a .rrd (the CI/screenshot path).
        _viz.log_incidents(incidents, save_path=args.save, spawn=args.save is None)
    except ImportError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))

    if args.save:
        print(f"wrote Rerun recording to {args.save}")
    else:
        print("opened Rerun viewer")
    return 0


def _cmd_import_lerobot(args):
    """Load a LeRobot dataset and write its frames as rollouts + episode-terminal outcomes,
    ready for `fieldloop attribute`. Needs the optional 'lerobot' extra; a missing extra is
    reported cleanly, not as a traceback."""
    from fieldloop import _lerobot

    try:
        dataset = _lerobot.load_dataset(args.repo_id, args.root)
        result = _lerobot.import_dataset(
            dataset,
            tenant_id=args.tenant_id,
            robot_id=args.robot_id,
            policy_version=args.policy_version,
            embodiment=args.embodiment,
            terminal_outcome=args.terminal_outcome,
            task_id=args.task_id,
        )
    except ImportError as e:
        return _fail(EXIT_ENGINE_ERROR, str(e))
    except Exception as e:
        return _fail(EXIT_BAD_FILE, f"could not import dataset: {e}")
    rollouts, outcomes = result["rollouts"], result["outcomes"]
    print(
        f"imported {len(rollouts)} rollouts and {len(outcomes)} outcomes from {args.root}"
    )
    try:
        _write_jsonl(args.out, rollouts)
        _write_jsonl(args.outcomes_out, outcomes)
    except OSError as e:
        return _fail(EXIT_BAD_FILE, str(e))
    print(f"wrote {args.out} and {args.outcomes_out}")
    print(
        f"  next: fieldloop attribute --config <embodiment.toml> "
        f"--rollouts {args.out} --outcomes {args.outcomes_out}"
    )
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="fieldloop",
        description=(
            "Open incident workbench for robot field failures: attribute delayed "
            "outcomes back to the decisions that caused them, with confidence, and "
            "gate weak evidence out of training data."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"fieldloop {metadata.version('fieldloop')}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="run the bundled toy-data loop end to end")
    p_demo.add_argument("--json", action="store_true", help="emit stable summary counts")
    p_demo.add_argument(
        "--viz",
        action="store_true",
        help="open the MCAP-demo incidents in Rerun (needs the optional 'viz' extra)",
    )
    p_demo.set_defaults(func=_cmd_demo)

    p_init = sub.add_parser("init", help="scaffold an embodiment config to edit")
    p_init.add_argument("--out", default="my_robot.toml", help="where to write the config")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_init.set_defaults(func=_cmd_init)

    p_attr = sub.add_parser(
        "attribute",
        help="bind delayed outcomes to decisions, from an MCAP file or JSONL",
    )
    p_attr.add_argument(
        "source",
        nargs="?",
        help="an MCAP recording to import (use with --map); omit for the JSONL inputs",
    )
    p_attr.add_argument("--config", required=True, help="embodiment TOML (see: fieldloop init)")
    p_attr.add_argument(
        "--map",
        dest="mapping",
        help="topic-mapping TOML (required when a source MCAP is given)",
    )
    p_attr.add_argument("--rollouts", help="rollout dicts, one JSON per line (JSONL mode)")
    p_attr.add_argument("--outcomes", help="outcome dicts, one JSON per line (JSONL mode)")
    p_attr.add_argument(
        "--out",
        help="write output JSONL here (incidents in MCAP mode, feedbacks in JSONL mode)",
    )
    p_attr.add_argument("--report", help="write a Markdown incident report here (MCAP mode)")
    p_attr.set_defaults(func=_cmd_attribute)

    p_doc = sub.add_parser(
        "doctor",
        help="check a mapping against an MCAP file's topics + clocks (exit 1 on problems)",
    )
    p_doc.add_argument("source", help="the MCAP file to check")
    p_doc.add_argument("--map", dest="mapping", required=True, help="topic-mapping TOML")
    p_doc.add_argument(
        "--clock-skew-threshold-ns",
        dest="clock_skew_threshold_ns",
        type=int,
        help="flag a topic whose source/recorder clock gap exceeds this (default 1s)",
    )
    p_doc.set_defaults(func=_cmd_doctor)

    p_cur = sub.add_parser(
        "curate", help="gate attributed feedbacks into a training slice (fail-closed)"
    )
    p_cur.add_argument("--rollouts", required=True, help="rollout dicts, one JSON per line")
    p_cur.add_argument("--feedbacks", required=True, help="feedbacks from `fieldloop attribute`")
    p_cur.add_argument("--grain", default="rollout", help="slice grain (default: rollout)")
    p_cur.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="confidence floor; weaker bindings go to needs_review (default: 0.70)",
    )
    p_cur.add_argument("--out", help="write the full slice result here as JSON")
    p_cur.set_defaults(func=_cmd_curate)

    p_view = sub.add_parser(
        "view",
        help="visualize incidents on a Rerun timeline (needs the optional 'viz' extra)",
    )
    p_view.add_argument("incidents", help="incidents JSONL from `fieldloop attribute --out`")
    p_view.add_argument(
        "--save", help="write a .rrd recording to this path instead of opening the viewer"
    )
    p_view.set_defaults(func=_cmd_view)

    p_lr = sub.add_parser(
        "import-lerobot",
        help="import a LeRobot dataset into rollouts + outcomes (needs the 'lerobot' extra)",
    )
    p_lr.add_argument("root", help="local LeRobot dataset root directory")
    p_lr.add_argument(
        "--repo-id", dest="repo_id", required=True, help="the dataset's repo id (see its meta/info.json)"
    )
    p_lr.add_argument("--tenant-id", dest="tenant_id", required=True)
    p_lr.add_argument("--robot-id", dest="robot_id", required=True)
    p_lr.add_argument("--policy-version", dest="policy_version", required=True)
    p_lr.add_argument("--embodiment", required=True)
    p_lr.add_argument(
        "--terminal-outcome",
        dest="terminal_outcome",
        required=True,
        help="outcome kind stamped on each episode's final frame (e.g. task_success)",
    )
    p_lr.add_argument("--task-id", dest="task_id", help="override the per-frame task label")
    p_lr.add_argument("--out", required=True, help="write rollouts JSONL here")
    p_lr.add_argument(
        "--outcomes-out", dest="outcomes_out", required=True, help="write outcomes JSONL here"
    )
    p_lr.set_defaults(func=_cmd_import_lerobot)

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
