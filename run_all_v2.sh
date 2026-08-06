#!/usr/bin/env bash
# The whole v2 (10-domain, non-<think>) pipeline, end to end: build tasks from
# data/textgrad_repro/ -> phase 3b (splits/oracle/canon/recon/SFT x2/ablation) -> phase 4b
# (downstream accuracy eval). Mirrors run_all.sh's shape but for the *_v2 pipeline
# (run_03b_training_validation_v2.sh / run_04b_downstream_eval_v2.sh) -- see docs/03's
# 2026-08-04 changelog entry and docs/04_downstream_eval.md §11 for what v1 vs v2 means.
#
#   bash run_all_v2.sh                       # lint + tests only for both phases (CPU-safe)
#   bash run_all_v2.sh --full                 # ... PLUS every real, long-running stage on a
#                                               # GPU node. Hours-long; run on the B200 node.
#   bash run_all_v2.sh --full --no-filter-correct
#   bash run_all_v2.sh --full --min-samples 30
#
# --filter-correct / --no-filter-correct (default: on) and --min-samples N (default: 50) are
# forwarded to scripts/build_tasks_from_textgrad_repro_v2.py: the correctness filter keeps
# only rows where the textgrad-repro teacher's response was correct (these rows double as
# SFT/oracle training targets, so incorrect ones would train the model to imitate wrong
# reasoning) and --min-samples drops a task entirely, rather than shipping a near-empty task
# dir, if fewer than that many rows survive the filter (only enforced when the filter is on).
#
# This script owns the ONE build_tasks_from_textgrad_repro_v2.py call for a given run --
# it runs before run_03b_training_validation_v2.sh, which then sees the task dirs already on
# disk and skips its own (unfiltered-by-default) internal build step. Re-running this script
# with different filter settings after task dirs already exist is a no-op for the build step
# (existing task dirs are left alone) -- pass --force-rebuild to remove and rebuild them, or
# clear the TASKS_OUT directory yourself first.
#
# Safe to re-run --full after an interruption: every downstream stage script skips work
# already on disk (see run_03b_training_validation_v2.sh's and
# run_04b_downstream_eval_v2.sh's own headers).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

FULL=0
FILTER_CORRECT=1
MIN_SAMPLES=50
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FULL=1; shift ;;
        --filter-correct) FILTER_CORRECT=1; shift ;;
        --no-filter-correct) FILTER_CORRECT=0; shift ;;
        --min-samples) MIN_SAMPLES="$2"; shift 2 ;;
        --force-rebuild) FORCE_REBUILD=1; shift ;;
        *) echo "error: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

FULL_FLAG=()
[[ $FULL -eq 1 ]] && FULL_FLAG=(--full)

echo "############################################"
echo "### phase 1: environment"
echo "############################################"
bash run_01_env.sh

if [[ ! -d .venv ]]; then
    echo "error: no .venv -- run 'bash run_01_env.sh' first" >&2
    exit 1
fi

if [[ $FULL -eq 1 ]]; then
    echo
    echo "############################################"
    echo "### build tasks from data/textgrad_repro/ (filter_correct=$FILTER_CORRECT, min_samples=$MIN_SAMPLES)"
    echo "############################################"

    TASKS_OUT="${TASKS_OUT:-/home/dg793/text-to-lora/tasks}"
    JSONL_OUT="${JSONL_OUT:-/home/dg793/text-to-lora/data/textgrad_repro_v2_t2l}"

    if [[ $FORCE_REBUILD -eq 1 ]]; then
        echo "--- --force-rebuild: removing existing textgrad_repro_v2_* task dirs under $TASKS_OUT"
        rm -rf "$TASKS_OUT"/textgrad_repro_v2_*
    fi

    SKIP_BUILD=0
    for task_dir in "$TASKS_OUT"/textgrad_repro_v2_*; do
        [[ -d "$task_dir" ]] && { echo "  tasks already built under $TASKS_OUT, skipping build step (pass --force-rebuild to redo)"; SKIP_BUILD=1; break; }
    done
    if [[ $SKIP_BUILD -eq 0 ]]; then
        FILTER_FLAG="--filter-correct"
        [[ "$FILTER_CORRECT" -eq 0 ]] && FILTER_FLAG="--no-filter-correct"
        uv run --no-sync python scripts/build_tasks_from_textgrad_repro_v2.py \
            --src-root data/textgrad_repro --jsonl-out "$JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    # Threaded through in case run_03b/run_04b are invoked below without task dirs already
    # present (e.g. first run on a fresh TASKS_OUT) -- see run_03b's own SKIP_BUILD logic.
    export FILTER_CORRECT MIN_SAMPLES
fi

echo
echo "############################################"
echo "### phase 3b: training & validation (v2 dataset)"
echo "############################################"
bash run_03b_training_validation_v2.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 4b: downstream accuracy eval (v2 dataset, small Q-holdout + full official"
echo "### test sets, both incl. oracle -- see docs/04 §13)"
echo "############################################"
# Needs a trained v2 hypernet checkpoint -- phase 3b writes the warm-started SFT arm to
# outputs/checkpoints/sft_warmstart_v2/latest.pt by default; override via HYPERNET_CKPT if
# your phase-3b run used different --out paths (same env vars run_04b_downstream_eval_v2.sh
# itself reads: HYPERNET_CKPT, ORACLE_DIR, OUT, OUT_FULL, GEN_BATCH_SIZE, FORCE).
bash run_04b_downstream_eval_v2.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### v2 pipeline complete"
echo "############################################"
