#!/usr/bin/env bash
# Phase 5 (v3 dataset): inference-time iterative TextGrad-style refinement pilot with T2L. See
# docs/07_iterative_t2l_application_v3.md.
#
# Unlike run_04c_downstream_eval_v3.sh (one fixed LoRA per task, scored once), this closes the
# loop: round 0 solves with the task's own best training description as a literal prompt (no
# LoRA); every later round critiques the previous round's generations via Qwen3-14B, feeds the
# resulting text into T2L for a fresh LoRA (replacing, not stacking, the previous round's), and
# re-scores. Pilot scope: 3 tasks spanning answer formats (gsm8k/integer, aqua/MCQ,
# strategyqa/yes-no), 5 refinement rounds each -- see docs/07 for why this scope, not the full
# 38-task v3 suite, was chosen for the first run.
#
#   bash run_05_iterative_t2l_v3.sh            # lint + full pytest suite, CPU-safe
#   bash run_05_iterative_t2l_v3.sh --full     # RUN THIS MANUALLY ON THE B200 NODE.
#                                               # Loads Qwen3-14B (vLLM), the v3 hypernet
#                                               # checkpoint + its Qwen2.5-3B backbone, and the
#                                               # Qwen2.5-1.5B target model all at once -- see
#                                               # docs/07's open item #1 on GPU memory, not yet
#                                               # validated anywhere else in this repo. Long-
#                                               # running (real generation, not teacher forcing,
#                                               # up to max-new-tokens per row, x2 pools x every
#                                               # round x 3 tasks).
#
# Needs run_03c_training_validation_v3.sh --full already done (data/splits_v3.json and a trained
# v3 hypernet checkpoint). Defaults HYPERNET_CKPT to sft_scratch_v3 -- the checkpoint
# docs/03/docs/04 §14 recommend trusting on v3 data, NOT sft_warmstart_v3 (recon warm-start
# collapsed there).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [[ ! -d .venv ]]; then
    echo "error: no .venv -- run 'bash run_01_env.sh' first" >&2
    exit 1
fi

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
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_scratch_v3/latest.pt}"
    FEEDBACK_MODEL="${FEEDBACK_MODEL:-Qwen/Qwen3-14B}"
    TASKS="${TASKS:-textgrad_repro_v3_gsm8k_d4 textgrad_repro_v3_aqua_d9 textgrad_repro_v3_strategyqa_d8}"
    MODE="${MODE:-prompt}"
    N_ROUNDS="${N_ROUNDS:-5}"
    OUT="${OUT:-outputs/eval/iterative_t2l_v3_pilot.json}"

    if [[ ! -f data/splits_v3.json ]]; then
        echo "error: data/splits_v3.json not found -- run 'bash run_03c_training_validation_v3.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained v3 hypernet checkpoint" >&2
        exit 1
    fi

    read -ra TASKS_ARR <<< "$TASKS"

    echo "--- iterative T2L pilot ($MODE mode, $N_ROUNDS rounds, tasks: ${TASKS_ARR[*]})"
    uv run --no-sync python scripts/eval_iterative_t2l_v3.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --tasks "${TASKS_ARR[@]}" --splits data/splits_v3.json \
        --feedback-model "$FEEDBACK_MODEL" --mode "$MODE" --n-rounds "$N_ROUNDS" --out "$OUT"
fi

echo
echo "=== phase 5 (v3 iterative application pilot) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_05_iterative_t2l_v3.sh --full"
fi
echo "next: see docs/07_iterative_t2l_application_v3.md for what to log"
