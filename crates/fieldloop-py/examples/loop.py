"""Run the whole FieldLoop OSS loop locally, in one file.

    capture -> attribute -> curate -> select_uploads

No database to set up. No cloud account. No robot required. The scenario itself lives
in the installed package (`fieldloop._demo`) so the same code also backs `fieldloop
demo`; this file is the thin runnable wrapper. To hack on the scenario — swap in your
own decisions and outcomes — copy `fieldloop/_demo.py` here and call its functions
directly instead of re-importing them.

Run it:

    uv run --project crates/fieldloop-py --extra dev \
        python crates/fieldloop-py/examples/loop.py

Add --json for a machine-readable summary (stable counts, for snapshots/CI).
"""

import argparse

from fieldloop._demo import (  # noqa: F401  (re-exported for readers and tests)
    CONFIG_TOML,
    CURATION_SPEC,
    capture_decisions,
    observed_outcomes,
    print_human,
    run,
    run_loop,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit stable summary counts")
    args = parser.parse_args()
    return run(json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
