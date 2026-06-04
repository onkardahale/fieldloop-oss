#!/usr/bin/env bash
# FieldLoop onboarding preflight: check the three tools the first run needs, and say
# EXACTLY what is missing + how to install it. FieldLoop OSS is local and source-built —
# the first run compiles a Rust Python extension — so a missing toolchain must read as
# "install this", never as a confusing "maturin failed" later.
#
#   ./scripts/preflight.sh
set -u
cd "$(dirname "$0")/.." || exit 1
# A rustup toolchain may be installed but not yet on this shell's PATH.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# Edition 2024 needs rustc >= 1.85; Python 3.12 is the abi3 floor the extension targets.
MIN_RUSTC_MINOR=85
MIN_PY_MINOR=12

ok() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$1"; }
missing=0

echo "FieldLoop preflight"
echo

# --- rustc (edition 2024) ---------------------------------------------------
if command -v rustc >/dev/null 2>&1; then
  ver=$(rustc --version | awk '{print $2}')
  minor=$(echo "$ver" | cut -d. -f2)
  if [ "${minor:-0}" -ge "$MIN_RUSTC_MINOR" ]; then
    ok "rustc $ver"
  else
    bad "rustc $ver is too old (need >= 1.${MIN_RUSTC_MINOR} for edition 2024)"
    echo "      Update: rustup update stable"
    missing=1
  fi
else
  bad "rustc not found"
  echo "      Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  missing=1
fi

# --- Python 3.12+ -----------------------------------------------------------
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -n "$PY" ]; then
  pyver=$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)
  pymin=$("$PY" -c 'import sys; print(sys.version_info[1])' 2>/dev/null)
  if [ "${pymin:-0}" -ge "$MIN_PY_MINOR" ]; then
    ok "Python $pyver"
  else
    bad "Python $pyver is too old (need >= 3.${MIN_PY_MINOR})"
    echo "      uv can install one: uv python install 3.12"
    missing=1
  fi
else
  bad "Python not found"
  echo "      uv can install one: uv python install 3.12"
  missing=1
fi

# --- uv (build + run driver) ------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}')"
else
  bad "uv not found"
  echo "      Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  missing=1
fi

# --- repo sanity ------------------------------------------------------------
if [ -f crates/fieldloop-py/examples/loop.py ]; then
  ok "repo root: $(basename "$(pwd)")/"
else
  bad "run this from the repository root (crates/fieldloop-py/examples/loop.py not found)"
  missing=1
fi

echo
if [ "$missing" -eq 0 ]; then
  echo "Ready. The first run compiles the Rust Python extension from source, then runs the loop:"
  echo
  echo "  uv run --project crates/fieldloop-py --extra dev \\"
  echo "    python crates/fieldloop-py/examples/loop.py"
  exit 0
else
  echo "Install the missing tool(s) above, then re-run ./scripts/preflight.sh"
  exit 1
fi
