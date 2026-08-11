#!/usr/bin/env bash
# Phase 4, v4 experiment: same eval as run_04c_downstream_eval_v3.sh, but pointed at the v4
# hypernet checkpoint/splits/oracle dir produced by run_03_training_validation_v4.sh --full, and
# at the comprehensive_feedback_v4_* task namespace (comprehensive-feedback T2L input instead of
# the optimized prompt text -- see docs/05_comprehensive_feedback_v4.md).
#
# Deliberate simplification vs run_04c_downstream_eval_v3.sh: v3's full-official-test-set eval
# (eval_downstream_accuracy_full.py) narrows to a single "winning instruction" task dir per
# original task, computed by scripts/select_best_prompt_tasks_v3.py matching literal
# best_prompt.json text against instruction groups. There is no feedback-side equivalent of
# "winning instruction" -- comprehensive feedback isn't itself scored the way a prompt is -- so no
# analogous selector was built. Both eval_downstream_accuracy.py (small Q-holdout) and
# eval_downstream_accuracy_full.py (full official test sets) here default to EVERY successful
# comprehensive_feedback_v4_* task dir (== every group that survived the builder's own
# --min-samples filter), same scope for both scripts, unlike v3's narrower full-eval scope.
#
#   bash run_04_downstream_eval_v4.sh            # lint + full pytest suite (same code path as
#                                                  # run_04c -- no v4-specific tests)
#   bash run_04_downstream_eval_v4.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                  # Runs BOTH eval_downstream_accuracy.py (small
#                                                  # Q-holdout, all 6 conditions incl. oracle) and
#                                                  # eval_downstream_accuracy_full.py (full
#                                                  # official test sets, all 6 conditions incl.
#                                                  # oracle) against the real v4 tasks, the v4
#                                                  # sft_warmstart hypernet checkpoint, and real
#                                                  # Qwen2.5 weights. Long-running.
#
# Same philosophy as run_04c_downstream_eval_v3.sh: no slurm, no DAG runner, a plain sequential
# bash script. --full needs run_03_training_validation_v4.sh --full already done (a trained v4
# hypernet checkpoint, data/splits_v4.json, and outputs/oracle_loras_v4 for the oracle condition)
# -- it does not train anything itself.
#
# Resumable: both eval_downstream_accuracy*.py scripts resume from --out, reusing (task,
# condition) pairs already recorded there. Pass FORCE=1 to redo everything, on both scripts at
# once. Point OUT/OUT_FULL at fresh files to keep a run's results separate from a prior run's.
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
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(comprehensive_feedback_v4_*)
    fi

    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_warmstart_v4/latest.pt}"
    ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_loras_v4}"
    OUT="${OUT:-outputs/eval/downstream_accuracy_v4.json}"
    OUT_FULL="${OUT_FULL:-outputs/eval/downstream_accuracy_full_v4.json}"
    GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
    FORCE_FLAG=()
    [[ "${FORCE:-0}" -eq 1 ]] && FORCE_FLAG=(--force)

    if [[ ! -f data/splits_v4.json ]]; then
        echo "error: data/splits_v4.json not found -- run 'bash run_03_training_validation_v4.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained v4 hypernet checkpoint" >&2
        exit 1
    fi

    echo "--- downstream accuracy eval (v4, small Q-holdout, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v4.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT" \
        "${FORCE_FLAG[@]}"

    echo
    echo "--- downstream accuracy eval (v4, full official test sets, all successful groups, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy_full.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v4.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT_FULL" \
        "${FORCE_FLAG[@]}"
fi

echo
echo "=== phase 4 (v4 experiment) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04_downstream_eval_v4.sh --full"
fi
echo "compare against outputs/eval/downstream_accuracy_v3.json / ..._full_v3.json to see whether"
echo "comprehensive-feedback-as-input changes downstream accuracy vs prompt-as-input"
