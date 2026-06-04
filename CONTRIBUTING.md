# Contributing to FieldLoop

Thanks for your interest. FieldLoop is a Rust workspace plus a Python binding, and the whole
loop runs locally with no database or services to set up.

## Setup

Check your toolchain (Rust, [`uv`](https://docs.astral.sh/uv/), Python 3.12+):

```bash
./scripts/preflight.sh
```

Then run the whole loop to see what the project does end to end:

```bash
uv run --project crates/fieldloop-py --extra dev \
  python crates/fieldloop-py/examples/loop.py
```

## Before you open a PR

Run the same checks CI does:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude fieldloop-py
uv run --project crates/fieldloop-py --extra dev ruff check crates/fieldloop-py
uv run --project crates/fieldloop-py --extra dev pytest -q crates/fieldloop-py/tests
```

## Conventions

- **Conventional Commits**: `<type>: <subject>` (`feat`, `fix`, `docs`, `refactor`, `test`,
  `chore`, …) — imperative, lower-case, no trailing period.
- **Comments explain intent** — why, invariants, tradeoffs — not the mechanics of the code.
- **Doc comments are the caller contract**: keep them precise and self-contained.

## Project shape

The loop is four stages: **capture → attribute → curate → select_uploads**. Most changes
touch a single crate — each crate's `//!` module docs say what it owns, and `docs/` covers
the concepts (`docs/concepts.md`), onboarding a robot (`docs/onboard-your-robot.md`), and the
design decisions (`docs/adr/`).

By contributing you agree your contributions are licensed under the repository's Apache-2.0
license (`LICENSE`).
