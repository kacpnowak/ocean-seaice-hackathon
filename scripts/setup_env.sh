#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the project virtual environment.
#
#   bash scripts/setup_env.sh          # normal setup
#   FORCE=1 bash scripts/setup_env.sh  # delete .venv and start over
#
# Timing: ~30 s warm, ~5 min cold.  Measured on JUPITER with uv's wheel cache
# already populated: 20 s for a brand-new .venv, 8 s to re-check an existing one.
# The first build on a machine has to download ~2 GB of PyTorch, which is the
# ~5 minutes; every build after that reuses the cache.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# CUDA build of torch to use.  aarch64 (GH200) wheels are NOT on plain PyPI --
# the PyPI aarch64 wheel is CPU-only -- so we must point at PyTorch's own index.
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
PYTHON_VERSION="3.12"

echo "==> Project root: $REPO_ROOT"

# --- 1. make sure uv is available -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found, installing it into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version | awk '{print $2}')"

# --- 2. create the venv ------------------------------------------------------
if [[ "${FORCE:-0}" == "1" && -d .venv ]]; then
    echo "==> FORCE=1, removing existing .venv"
    rm -rf .venv
fi

if [[ ! -d .venv ]]; then
    echo "==> Creating .venv with Python ${PYTHON_VERSION}"
    uv venv --python "${PYTHON_VERSION}" .venv
else
    echo "==> Reusing existing .venv"
fi

# --- 3. install ---------------------------------------------------------------
# --index-strategy unsafe-best-match lets uv pick torch from the CUDA index while
# still taking everything else from PyPI.
echo "==> Installing oceanarches + geoarches (this pulls torch, ~2 GB) ..."
uv pip install \
    --python .venv/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url "$TORCH_INDEX" \
    --overrides overrides.txt \
    -e ".[dev]"

echo
echo "==> Done.  You do NOT need to activate anything: every make target and"
echo "    every command in the docs runs .venv/bin/python directly."
echo
echo "    Next, once per clone (about 8 minutes):"
echo "      make stats     # masks, normalisation statistics, climatology"
echo "      make doctor    # FAILs until make stats has run"
