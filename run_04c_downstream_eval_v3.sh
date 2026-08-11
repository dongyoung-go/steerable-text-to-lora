#!/usr/bin/env bash
# Phase 4, v3 dataset: same eval as run_04b_downstream_eval_v2.sh, but pointed at the v3
# hypernet checkpoint/splits/oracle dir produced by run_03c_training_validation_v3.sh --full. See
# docs/03/docs/04 for what v1/v2 were; v3 additionally pools reverted/repeated-prompt iterations
# (see run_all_v3.sh) and trains one oracle LoRA per distinct instruction (task dirs suffixed
# _d0, _d1, ... -- see the builders' docstrings).
#
# Both eval scripts default to the SAME task scope: only the single winning-instruction task dir
# per original task/algorithm -- "the output of TextGrad and GEPA" itself, not every
# rejected/reverted `_dK` variant a task's optimization run also tried. Computed fresh each run
# by scripts/select_best_prompt_tasks_v3.py into data/best_prompt_tasks_v3.txt (fast, CPU-only,
# reads each source dir's own best_prompt.json + iterations.jsonl -- see that script's docstring
# for exactly how "winning" is resolved, including its fallback when the literal winning
# instruction's group didn't survive --min-samples). Since a single task list feeds every
# condition, this scope applies uniformly to base/prompted/oracle/t2l_train_desc/
# t2l_other_task_desc/t2l_gibberish_desc for both scripts.
#
# eval_downstream_accuracy.py's earlier default was every successful
# textgrad_repro_v3_*/gepa_repro_v3_* task dir (unlike v2's hardcoded 8-task allowlist) -- honest
# full-variant coverage, but ~576 task dirs is a ~10x multiplier on top of already-expensive
# per-condition generation (up to max-new-tokens per row). Set TRAIN_TASKS="textgrad_repro_v3_*
# gepa_repro_v3_*" below to get that back for this script specifically.
#
#   bash run_04c_downstream_eval_v3.sh            # lint + full pytest suite (same as
#                                                   # run_04b_downstream_eval_v2.sh -- no
#                                                   # v3-specific tests, same code path)
#   bash run_04c_downstream_eval_v3.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                   # Runs BOTH eval_downstream_accuracy.py
#                                                   # (small Q-holdout, all 6 conditions incl.
#                                                   # oracle) and eval_downstream_accuracy_full.py
#                                                   # (full official test sets, all 6 conditions
#                                                   # incl. oracle) against the real v3 tasks, the
#                                                   # v3 sft_warmstart hypernet checkpoint, and
#                                                   # real Qwen2.5 weights. Long-running (real
#                                                   # generation, not teacher forcing, up to
#                                                   # max-new-tokens per row per condition).
#
# Same philosophy as run_03c_training_validation_v3.sh: no slurm, no DAG runner, a plain
# sequential bash script. --full needs run_03c_training_validation_v3.sh --full already done (a
# trained v3 hypernet checkpoint, data/splits_v3.json, and outputs/oracle_loras_v3 for the oracle
# condition) -- it does not train anything itself.
#
# Resumable: both eval_downstream_accuracy*.py scripts resume from --out, reusing (task,
# condition) pairs already recorded there (real generation is expensive) -- pass
# FORCE=1 to redo everything, on both scripts at once. Point OUT/OUT_FULL at fresh files to
# keep a run's results separate from a prior run's output JSON.
#
# GEN_BATCH_SIZE (default 64) is forwarded to both scripts' --gen-batch-size: docs/04 §13 found
# the --gen-batch-size 8 default leaves a B200 at ~26-30% utilization, while 64 gives ~9x
# throughput at only 5.6GB peak memory -- see docs/04's "batch size matters a lot" note.
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
    BEST_PROMPT_TASKS_FILE="${BEST_PROMPT_TASKS_FILE:-data/best_prompt_tasks_v3.txt}"

    # Both TRAIN_TASKS_ARR and FULL_EVAL_TRAIN_TASKS_ARR default to the same winning-instruction
    # list (see header comment) -- computed once here, fresh every run (cheap, CPU-only, no
    # GPU/network needed), unless both TRAIN_TASKS and FULL_EVAL_TRAIN_TASKS are already set, in
    # which case neither script needs it.
    if [[ -z "${TRAIN_TASKS:-}" || -z "${FULL_EVAL_TRAIN_TASKS:-}" ]]; then
        echo "--- selecting winning-instruction task dirs (default scope for both eval scripts)"
        uv run --no-sync python scripts/select_best_prompt_tasks_v3.py \
            --textgrad-src-root data/textgrad_repro --gepa-src-root data/gepa_repro \
            --tasks-out "$TASKS_ROOT" --out "$BEST_PROMPT_TASKS_FILE"
    fi

    # Set TRAIN_TASKS to a space-separated list of patterns (glob or literal) to override, e.g.
    # TRAIN_TASKS="textgrad_repro_v3_* gepa_repro_v3_*" for every instruction variant (the old
    # default) instead of just the winner.
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        mapfile -t TRAIN_TASKS_ARR < "$BEST_PROMPT_TASKS_FILE"
    fi

    if [[ -n "${FULL_EVAL_TRAIN_TASKS:-}" ]]; then
        read -ra FULL_EVAL_TRAIN_TASKS_ARR <<< "$FULL_EVAL_TRAIN_TASKS"
    else
        mapfile -t FULL_EVAL_TRAIN_TASKS_ARR < "$BEST_PROMPT_TASKS_FILE"
    fi

    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_warmstart_v3/latest.pt}"
    ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_loras_v3}"
    OUT="${OUT:-outputs/eval/downstream_accuracy_v3.json}"
    OUT_FULL="${OUT_FULL:-outputs/eval/downstream_accuracy_full_v3.json}"
    GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
    FORCE_FLAG=()
    [[ "${FORCE:-0}" -eq 1 ]] && FORCE_FLAG=(--force)

    if [[ ! -f data/splits_v3.json ]]; then
        echo "error: data/splits_v3.json not found -- run 'bash run_03c_training_validation_v3.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained v3 hypernet checkpoint" >&2
        exit 1
    fi

    echo "--- downstream accuracy eval (v3, small Q-holdout, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT" \
        "${FORCE_FLAG[@]}"

    echo
    echo "--- downstream accuracy eval (v3, full official test sets, winning instruction only, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy_full.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${FULL_EVAL_TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT_FULL" \
        "${FORCE_FLAG[@]}"
fi

echo
echo "=== phase 4 (v3 dataset) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04c_downstream_eval_v3.sh --full"
fi
echo "next: see docs/03_training_validation.md / docs/04_downstream_eval.md for what to log"
