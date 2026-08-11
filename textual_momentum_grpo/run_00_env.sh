#!/usr/bin/env bash
# Create/sync this project's own isolated venv. See docs/build_and_run_guide.md.
#
# This venv (textual_momentum_grpo/.venv) is completely separate from
# /home/dg793/steerable-text-to-lora's own .venv/pyproject.toml/uv.lock -- this script never
# touches those. It only installs the lightweight, CPU-only deps this project's own harness
# code (tmgrpo/, scripts/) needs. The heavy RL training stack (verl + torch + vllm + ray) is
# NOT installed here -- it runs from the official verl docker image on a GPU node.
#
#   bash run_00_env.sh              # create/sync the venv, then check
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH (expected ~/.local/bin/uv)" >&2
    echo "install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "=== uv $(uv --version | awk '{print $2}')"
echo "=== creating venv at ${PROJECT_DIR}/.venv (python 3.12)"
uv venv --python 3.12 --seed --allow-existing

echo "=== syncing dependencies (core + dev)"
uv sync --extra dev

echo "=== sanity check"
uv run --no-sync python -c "import tmgrpo; print('tmgrpo import OK')"

echo
echo "=== environment ready"
echo "next: uv run --no-sync pytest tests   (CPU-only unit tests)"
echo "for actual GPU training runs: see docs/build_and_run_guide.md (verl docker, separate node)"
