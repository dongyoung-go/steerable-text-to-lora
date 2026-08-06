#!/usr/bin/env bash
# Phase 4, v2 dataset: same eval as run_04_downstream_eval.sh, but pointed at the
# textgrad_repro_v2 hypernet checkpoint/splits/oracle dir produced by
# run_03b_training_validation_v2.sh --full. See docs/04_downstream_eval.md and docs/03's
# 2026-08-04 changelog entry for what v2 is.
#
#   bash run_04b_downstream_eval_v2.sh            # lint + full pytest suite (same as
#                                                   # run_04_downstream_eval.sh -- no
#                                                   # v2-specific tests, same code path)
#   bash run_04b_downstream_eval_v2.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                   # Runs BOTH eval_downstream_accuracy.py
#                                                   # (small Q-holdout, all 6 conditions incl.
#                                                   # oracle) and eval_downstream_accuracy_full.py
#                                                   # (full official test sets, all 6 conditions
#                                                   # incl. oracle -- see docs/04 §13) against
#                                                   # the real textgrad_repro_v2 tasks, the v2
#                                                   # sft_warmstart hypernet checkpoint, and
#                                                   # real Qwen2.5 weights. Long-running (real
#                                                   # generation, not teacher forcing, up to
#                                                   # max-new-tokens per row per condition).
#
# Same philosophy as run_03b_training_validation_v2.sh: no slurm, no DAG runner, a plain
# sequential bash script. --full needs run_03b_training_validation_v2.sh --full already done
# (a trained v2 hypernet checkpoint, data/splits_v2.json, and outputs/oracle_loras_v2 for the
# oracle condition) -- it does not train anything itself.
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
    # Default is the explicit 8-task set docs/04 §13 documents results for -- NOT a
    # 'textgrad_repro_v2_*' glob. tasks-root now also holds ~12 additional bbh_* task dirs
    # (exposed by the --min-samples 0 metadata rebuild, docs/03's 2026-08-05 changelog) that
    # the hypernet may have trained on but that have no outputs/oracle_loras_v2/ adapter and
    # were never part of this documented run -- a glob here would silently pull all of them
    # into both eval calls below (real generation on ~12 extra tasks, oracle=n/a for each).
    # Set TRAIN_TASKS to a space-separated list of patterns (glob or literal) to override,
    # e.g. TRAIN_TASKS='textgrad_repro_v2_*' to deliberately run the full, wider set.
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(
            textgrad_repro_v2_aqua
            textgrad_repro_v2_bbh_causal_judgement
            textgrad_repro_v2_bbh_date_understanding
            textgrad_repro_v2_bbh_dyck_languages
            textgrad_repro_v2_bbh_formal_fallacies
            textgrad_repro_v2_bbh_geometric_shapes
            textgrad_repro_v2_bbh_logical_deduction_seven_objects
            textgrad_repro_v2_bbh_movie_recommendation
        )
    fi
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    HYPERNET_CKPT="${HYPERNET_CKPT:-outputs/checkpoints/sft_warmstart_v2/latest.pt}"
    ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_loras_v2}"
    OUT="${OUT:-outputs/eval/downstream_accuracy_v2.json}"
    OUT_FULL="${OUT_FULL:-outputs/eval/downstream_accuracy_full_v2.json}"
    GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
    FORCE_FLAG=()
    [[ "${FORCE:-0}" -eq 1 ]] && FORCE_FLAG=(--force)

    if [[ ! -f data/splits_v2.json ]]; then
        echo "error: data/splits_v2.json not found -- run 'bash run_03b_training_validation_v2.sh --full' first" >&2
        exit 1
    fi
    if [[ ! -f "$HYPERNET_CKPT" ]]; then
        echo "error: $HYPERNET_CKPT not found -- set HYPERNET_CKPT to a trained v2 hypernet checkpoint" >&2
        exit 1
    fi

    echo "--- downstream accuracy eval (v2, small Q-holdout, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v2.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT" \
        "${FORCE_FLAG[@]}"

    echo
    echo "--- downstream accuracy eval (v2, full official test sets, incl. oracle -- docs/04 §13)"
    # oracle-dir passed through here too: run_downstream_eval supports it regardless of
    # rows_for_task, and the whole point of docs/04 §13's TODO was checking whether the
    # small-Q-holdout oracle numbers above hold up on a bigger, disjoint test set.
    uv run --no-sync python scripts/eval_downstream_accuracy_full.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v2.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT_FULL" \
        "${FORCE_FLAG[@]}"
fi

echo
echo "=== phase 4 (v2 dataset) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04b_downstream_eval_v2.sh --full"
fi
echo "next: see docs/04_downstream_eval.md changelog for what was verified"
