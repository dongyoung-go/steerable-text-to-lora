#!/usr/bin/env bash
# Phase 4: downstream accuracy eval. See docs/04_downstream_eval.md.
#
#   bash run_04_downstream_eval.sh            # lint + full pytest suite (CPU-only, tiny
#                                               # synthetic fixtures, no network, no GPU, no
#                                               # real weights -- gold-answer join, parsing,
#                                               # per-condition scoring, and the lora_hooks
#                                               # multi-token generate() coverage check)
#   bash run_04_downstream_eval.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                               # eval_downstream_accuracy.py against the real
#                                               # 13-task GSM8K data, the real trained
#                                               # hypernetwork checkpoint, and real Qwen2.5
#                                               # weights. Long-running (real generation, not
#                                               # teacher forcing, up to max-new-tokens per row
#                                               # per condition).
#
# Same philosophy as run_03_training_validation.sh: no slurm, no DAG runner, a plain
# sequential bash script. --full needs docs/03's real run already done (a trained hypernet
# checkpoint and, for the oracle condition, outputs/oracle_loras) -- it does not train
# anything itself.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [[ ! -d .venv ]]; then
    echo "error: no .venv -- run 'bash run_01_env.sh' first" >&2
    exit 1
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}  # gold-answer join needs one live GSM8K fetch (or an HF cache hit)
export TOKENIZERS_PARALLELISM=false

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

echo "=== lint"
uv run --no-sync ruff check src tests scripts

echo
echo "=== tests"
uv run --no-sync python -m pytest tests -q

if [[ $FULL -eq 1 ]]; then
    echo
    echo "=== REAL run on B200 -- long-running, run manually and monitor ==="

    TASKS_ROOT="${TASKS_ROOT:-/home/dg793/text-to-lora/tasks}"
    TRAIN_TASKS="${TRAIN_TASKS:-textgrad_repro_gsm8k_*}"
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_warmstart/latest.pt}"
    ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_loras}"
    OUT="${OUT:-outputs/eval/downstream_accuracy.json}"

    if [[ ! -f data/splits.json ]]; then
        echo "error: data/splits.json not found -- run 'bash run_03_training_validation.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained hypernet checkpoint" >&2
        exit 1
    fi

    echo "--- downstream accuracy eval"
    uv run --no-sync python scripts/eval_downstream_accuracy.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "$TRAIN_TASKS" --splits data/splits.json \
        --oracle-dir "$ORACLE_DIR" --out "$OUT"
fi

echo
echo "=== phase 4 complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04_downstream_eval.sh --full"
fi
echo "next: see docs/04_downstream_eval.md changelog for what was verified"
