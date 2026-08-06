#!/usr/bin/env bash
# Phase 1: create the environment and verify it. See docs/01_env.md.
#
# Safe to run on the CPU login node: nothing here compiles CUDA.
#
#   bash run_01_env.sh              # create/sync the venv, then check
#   bash run_01_env.sh --upgrade    # also re-resolve the lockfile to newer versions
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH (expected ~/.local/bin/uv)" >&2
    echo "install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "=== uv $(uv --version | awk '{print $2}')"

if [[ "${1:-}" == "--upgrade" ]]; then
    echo "=== re-resolving lockfile (--upgrade)"
    uv lock --upgrade
fi

echo "=== creating venv (python 3.12)"
uv venv --python 3.12 --seed --allow-existing

echo "=== syncing dependencies (core + dev + attn + log)"
# `attn` installs `kernels`, which lets transformers fetch a PREBUILT flash-attn from the
# Hub at load time -- no CUDA toolkit, no compiler, no MAX_JOBS. The default backend is
# still sdpa (fused cuDNN attention on Blackwell), which needs nothing at all.
uv sync --extra dev --extra attn --extra log

echo "=== environment check"
uv run --no-sync python -m steerable_t2l.utils.env

echo
echo "=== model weights (needed for GPU runs; compute nodes run HF_HUB_OFFLINE=1)"
if uv run --no-sync python scripts/prefetch_models.py --check; then
    echo "  all model weights are cached"
else
    echo "  fetching missing weights (needs network) ..."
    uv run --no-sync python scripts/prefetch_models.py
fi

echo
echo "=== phase 1 complete"
echo "next: bash run_02_model.sh"
