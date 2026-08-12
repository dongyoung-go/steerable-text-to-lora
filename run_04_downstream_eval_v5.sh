#!/usr/bin/env bash
# Phase 4, v5 experiment: same eval as run_04c_downstream_eval_v3.sh, but pointed at the v5
# hypernet checkpoint/splits/oracle dir produced by run_03_training_validation_v5.sh --full, and at
# the description-paraphrase-augmented textgrad_repro_v5_*/gepa_repro_v5_* task namespace (see
# docs/06_description_augmentation_v5.md). This is deliberately v3's `_c` eval script's scope, not
# v4's: outputs/eval/downstream_accuracy_full_v3.json -- the intended comparison target for this
# experiment (does more descriptions per task fix the description-independent-LoRA collapse
# docs/06 diagnosed against v3?) -- was produced restricted to one winning-instruction task dir per
# task/algorithm, and v5 needs that identical scope to be a fair comparison.
#
# scripts/select_best_prompt_tasks_v3.py hardcodes the textgrad_repro_v3/gepa_repro_v3 prefixes in
# its own source (it derives dir_name() from a literal "textgrad_repro_v3"/"gepa_repro_v3" prefix,
# not from an argument) -- it cannot itself emit _v5_-named winners. But v5's task dirs are exact
# name-for-name copies of v3's (same winning instruction per task, identical underlying data, only
# `descriptions` differs -- see docs/06), so the fix is a substitution, not a rerun: regenerate
# data/best_prompt_tasks_v3.txt fresh (cheap, CPU-only, no GPU/network), then derive
# data/best_prompt_tasks_v5.txt by replacing "_v3_" -> "_v5_" in each line, verifying every derived
# name actually exists under $TASKS_ROOT (should be exact 1:1 -- errors loudly if not).
#
#   bash run_04_downstream_eval_v5.sh            # lint + full pytest suite (same code path as
#                                                  # run_04c/run_04 v3/v4 -- no v5-specific tests)
#   bash run_04_downstream_eval_v5.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                  # Runs BOTH eval_downstream_accuracy.py (small
#                                                  # Q-holdout, all 6 conditions incl. oracle) and
#                                                  # eval_downstream_accuracy_full.py (full official
#                                                  # test sets, all 6 conditions incl. oracle)
#                                                  # against the real v5 tasks, the v5 sft_warmstart
#                                                  # hypernet checkpoint, and real Qwen2.5 weights.
#                                                  # Long-running.
#
# Same philosophy as run_04c_downstream_eval_v3.sh: no slurm, no DAG runner, a plain sequential
# bash script. --full needs run_03_training_validation_v5.sh --full already done (a trained v5
# hypernet checkpoint, data/splits_v5.json, and outputs/oracle_loras_v5 for the oracle condition)
# -- it does not train anything itself.
#
# Resumable: both eval_downstream_accuracy*.py scripts resume from --out, reusing (task, condition)
# pairs already recorded there. Pass FORCE=1 to redo everything, on both scripts at once. Point
# OUT/OUT_FULL at fresh files to keep a run's results separate from a prior run's output JSON.
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
    BEST_PROMPT_TASKS_V3_FILE="${BEST_PROMPT_TASKS_V3_FILE:-data/best_prompt_tasks_v3.txt}"
    BEST_PROMPT_TASKS_FILE="${BEST_PROMPT_TASKS_FILE:-data/best_prompt_tasks_v5.txt}"

    if [[ -n "${TRAIN_TASKS:-}" && -n "${FULL_EVAL_TRAIN_TASKS:-}" ]]; then
        : # both already set below, no need to derive the winner list
    else
        echo "--- selecting winning-instruction task dirs, v3 scope (fresh, cheap, CPU-only)"
        uv run --no-sync python scripts/select_best_prompt_tasks_v3.py \
            --textgrad-src-root data/textgrad_repro --gepa-src-root data/gepa_repro \
            --tasks-out "$TASKS_ROOT" --out "$BEST_PROMPT_TASKS_V3_FILE"

        echo "--- deriving v5 winner list from v3's (name-for-name copies, see header)"
        sed 's/_v3_/_v5_/' "$BEST_PROMPT_TASKS_V3_FILE" > "$BEST_PROMPT_TASKS_FILE"
        MISSING=0
        while IFS= read -r name; do
            [[ -z "$name" ]] && continue
            if [[ ! -d "$TASKS_ROOT/$name" ]]; then
                echo "  error: derived v5 task dir $TASKS_ROOT/$name does not exist" >&2
                MISSING=1
            fi
        done < "$BEST_PROMPT_TASKS_FILE"
        if [[ $MISSING -eq 1 ]]; then
            echo "error: not every v3 winner has a v5 counterpart -- see missing entries above" >&2
            exit 1
        fi
    fi

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
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_warmstart_v5/latest.pt}"
    ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_loras_v5}"
    OUT="${OUT:-outputs/eval/downstream_accuracy_v5.json}"
    OUT_FULL="${OUT_FULL:-outputs/eval/downstream_accuracy_full_v5.json}"
    GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
    FORCE_FLAG=()
    [[ "${FORCE:-0}" -eq 1 ]] && FORCE_FLAG=(--force)

    if [[ ! -f data/splits_v5.json ]]; then
        echo "error: data/splits_v5.json not found -- run 'bash run_03_training_validation_v5.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained v5 hypernet checkpoint" >&2
        exit 1
    fi

    echo "--- downstream accuracy eval (v5, small Q-holdout, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v5.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT" \
        "${FORCE_FLAG[@]}"

    echo
    echo "--- downstream accuracy eval (v5, full official test sets, winning instruction only, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy_full.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${FULL_EVAL_TRAIN_TASKS_ARR[@]}" --splits data/splits_v5.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT_FULL" \
        "${FORCE_FLAG[@]}"
fi

echo
echo "=== phase 4 (v5 experiment) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04_downstream_eval_v5.sh --full"
fi
echo "compare against outputs/eval/downstream_accuracy_full_v3.json with:"
echo "  python scripts/compare_downstream_eval.py outputs/eval/downstream_accuracy_full_v3.json outputs/eval/downstream_accuracy_full_v5.json --labels v3 v5"
