#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the project virtual environment.
#
#   bash scripts/setup_env.sh          # normal setup
#   FORCE=1 bash scripts/setup_env.sh  # delete .venv and start over
#
# Timing: ~30 s warm, ~5 min cold.  Measured with uv's wheel cache
# already populated: 20 s for a brand-new .venv, 8 s to re-check an existing one.
# The first build on a machine has to download ~2 GB of PyTorch, which is the
# ~5 minutes; every build after that reuses the cache.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# CUDA build of torch, from PyTorch's own index rather than PyPI.
# Naming the index keeps the CUDA version explicit rather than trusting whatever
# PyPI happens to serve for this platform.
#
# A `.venv` is not portable between machines of different architecture -- build
# it here, which is what this script does.
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
PYTHON_VERSION="3.12"

echo "==> Project root: $REPO_ROOT"

# --- 0. keep uv off $HOME ----------------------------------------------------
# JURECA's $HOME is inode-quotaed, and the quota is small: measured at ~2050
# files total, of which a fresh account already uses ~480.  uv defaults to
# unpacking its managed CPython into ~/.local/share/uv/python (several thousand
# files) and its wheel cache into ~/.cache/uv (tens of thousands), so a stock
# `uv venv` dies partway through the interpreter with
#
#     Failed to extract archive: cpython-3.12.14-...tar.gz
#       Caused by: Disk quota exceeded (os error 122)
#
# which says nothing about $HOME, nothing about inodes, and happens before torch
# is even considered.  df reports terabytes free, because the limit is on the
# number of files and not on their size.  Both directories go under the repo --
# which is on scratch -- unless the caller has already chosen somewhere else.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv/cache}"
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"
echo "==> uv python  -> $UV_PYTHON_INSTALL_DIR"
echo "==> uv cache   -> $UV_CACHE_DIR"

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
