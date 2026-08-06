#!/usr/bin/env bash
# Phase 2: verify the model implementation. See docs/02_model.md.
#
# Runs entirely on CPU. No GPU, no CUDA toolkit, no network.
#
#   bash run_02_model.sh            # lint + tests + self-check          (~15s)
#   bash run_02_model.sh --full     # ... plus the REAL Qwen2.5 3B/1.5B  (~3 min)
#
# The default path uses tiny synthetic models. --full loads the actual backbone and target
# from the HF cache and runs a real forward+backward through the hooks, which is the same
# code path a GPU run takes -- so almost everything can be verified before spending GPU time.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [[ ! -d .venv ]]; then
    echo "error: no .venv -- run 'bash run_01_env.sh' first" >&2
    exit 1
fi

# Weights and tokenizers are cached; do not reach out to the Hub.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

echo "=== lint"
uv run --no-sync ruff check src tests scripts

echo
echo "=== tests"
uv run --no-sync python -m pytest tests -q

echo
echo "=== model self-check (synthetic models)"
uv run --no-sync python -m steerable_t2l.hypernet --self-check

if [[ $FULL -eq 1 ]]; then
    echo
    echo "=== real-model smoke check on CPU (Qwen2.5-3B-Instruct -> Qwen2.5-1.5B-Instruct)"
    uv run --no-sync python scripts/smoke_check.py --batch 2 --seq 64 --backward
fi

echo
echo "=== phase 2 complete"
if [[ $FULL -eq 0 ]]; then
    echo "real-model check (optional, ~3 min): bash run_02_model.sh --full"
fi
echo "next: docs/03_training_validation.md is a specification only -- implement it in a new session"
