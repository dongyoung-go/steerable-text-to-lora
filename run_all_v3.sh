#!/usr/bin/env bash
# The whole v3 pipeline, end to end: build tasks from BOTH data/textgrad_repro/ and
# data/gepa_repro/ -> phase 3c (splits/oracle/canon/recon/SFT x2/ablation) -> phase 4c
# (downstream accuracy eval). Mirrors run_all_v2.sh's shape but for the *_v3 pipeline
# (run_03c_training_validation_v3.sh / run_04c_downstream_eval_v3.sh).
#
# What's new vs v2:
#   - builds tasks from data/gepa_repro/ in addition to data/textgrad_repro/ (two separate
#     builder scripts, two separate task namespaces -- textgrad_repro_v3_<task>_d<K> and
#     gepa_repro_v3_<task>_d<K> stay distinct tasks/LoRAs even where the task name is shared
#     between the two algorithms; no cross-algorithm merging).
#   - ONE TASK DIR (== ONE ORACLE LORA) PER DISTINCT INSTRUCTION, not one task dir per task
#     name. v2 picked a single best-scoring iteration/candidate per task and used only its
#     responses, with every other instruction phrasing tried during the run listed in
#     metadata.yaml's "descriptions" purely as unpaired alternates for D-axis eval. v3 instead
#     groups each task's forward-pass rows by their own literal instruction text (in
#     first-appearance order: _d0, _d1, _d2, ...) and writes a separate task dir per group, each
#     with exactly one description. This matters because the hypernetwork/oracle training loader
#     (PerTaskDescDataset) samples a description at random from a task's WHOLE descriptions pool
#     per training row, independent of which description produced that row's response --
#     pooling genuinely different (non-paraphrase) instructions under one task's descriptions
#     list would silently teach false instruction->response associations. Rows generated across
#     iterations that reused the *identical* instruction text (textgrad's "reverted" mechanic;
#     gepa re-evaluating a candidate at multiple search points) are still pooled together within
#     their group, since that's the same description being reused, not a different one.
#   - dedup changes from "1 row per question" to "1 row per (question, response) pair" in both
#     builders, so pooling reused-identical-instruction rows actually grows sample counts instead
#     of being immediately collapsed back down to one row per question.
#   - side effect: every task dir has exactly one description, so the D-axis (description
#     holdout) eval split is universally N/A for v3 -- same degenerate case the codebase already
#     warns about for legacy single-description tasks. Accepted tradeoff until
#     description-augmentation is built as its own feature.
#   - phase 4c has no hardcoded task allowlist (v2's run_04b hardcoded 8 tasks) -- it defaults to
#     every successful task dir from both namespaces, where "successful" already means "passed
#     the builder's own --min-samples filter".
#
#   bash run_all_v3.sh                        # lint + tests only for both phases (CPU-safe)
#   bash run_all_v3.sh --full                  # ... PLUS every real, long-running stage.
#                                               # Hours-long -- RUN ON THE B200 GPU NODE, NOT
#                                               # this CPU node. Nothing in --full is CPU-safe:
#                                               # oracle/recon/SFT training and real generation
#                                               # for downstream eval all need GPU.
#   bash run_all_v3.sh --full --no-filter-correct
#   bash run_all_v3.sh --full --min-samples 30
#
# --filter-correct / --no-filter-correct (default: on) and --min-samples N (default: 50) are
# forwarded to BOTH scripts/build_tasks_from_textgrad_repro_v3.py and
# scripts/build_tasks_from_gepa_repro_v3.py: the correctness filter keeps only rows where the
# teacher's response was correct (these rows double as SFT/oracle training targets, so incorrect
# ones would train the model to imitate wrong reasoning) and --min-samples drops a task entirely,
# rather than shipping a near-empty task dir, if fewer than that many rows survive the filter
# (only enforced when the filter is on).
#
# This script owns the ONE pair of build_tasks_from_*_v3.py calls for a given run -- it runs
# before run_03c_training_validation_v3.sh, which then sees the task dirs already on disk and
# skips its own (unfiltered-by-default) internal build step. Re-running this script with
# different filter settings after task dirs already exist is a no-op for the build step (existing
# task dirs are left alone) -- pass --force-rebuild to remove and rebuild them, or clear the
# relevant TASKS_OUT/*_v3_* dirs yourself first.
#
# Safe to re-run --full after an interruption: every downstream stage script skips work already
# on disk (see run_03c_training_validation_v3.sh's and run_04c_downstream_eval_v3.sh's own
# headers).
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
    TASKS_OUT="${TASKS_OUT:-/home/dg793/text-to-lora/tasks}"
    TG_JSONL_OUT="${TG_JSONL_OUT:-/home/dg793/text-to-lora/data/textgrad_repro_v3_t2l}"
    GEPA_JSONL_OUT="${GEPA_JSONL_OUT:-/home/dg793/text-to-lora/data/gepa_repro_v3_t2l}"

    if [[ $FORCE_REBUILD -eq 1 ]]; then
        echo "--- --force-rebuild: removing existing textgrad_repro_v3_* and gepa_repro_v3_* task dirs under $TASKS_OUT"
        rm -rf "$TASKS_OUT"/textgrad_repro_v3_* "$TASKS_OUT"/gepa_repro_v3_*
    fi

    FILTER_FLAG="--filter-correct"
    [[ "$FILTER_CORRECT" -eq 0 ]] && FILTER_FLAG="--no-filter-correct"

    echo
    echo "############################################"
    echo "### build tasks from data/textgrad_repro/ (filter_correct=$FILTER_CORRECT, min_samples=$MIN_SAMPLES)"
    echo "############################################"
    SKIP_TG_BUILD=0
    for task_dir in "$TASKS_OUT"/textgrad_repro_v3_*; do
        [[ -d "$task_dir" ]] && { echo "  tasks already built under $TASKS_OUT, skipping build step (pass --force-rebuild to redo)"; SKIP_TG_BUILD=1; break; }
    done
    if [[ $SKIP_TG_BUILD -eq 0 ]]; then
        uv run --no-sync python scripts/build_tasks_from_textgrad_repro_v3.py \
            --src-root data/textgrad_repro --jsonl-out "$TG_JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    echo
    echo "############################################"
    echo "### build tasks from data/gepa_repro/ (filter_correct=$FILTER_CORRECT, min_samples=$MIN_SAMPLES)"
    echo "############################################"
    SKIP_GEPA_BUILD=0
    for task_dir in "$TASKS_OUT"/gepa_repro_v3_*; do
        [[ -d "$task_dir" ]] && { echo "  tasks already built under $TASKS_OUT, skipping build step (pass --force-rebuild to redo)"; SKIP_GEPA_BUILD=1; break; }
    done
    if [[ $SKIP_GEPA_BUILD -eq 0 ]]; then
        uv run --no-sync python scripts/build_tasks_from_gepa_repro_v3.py \
            --src-root data/gepa_repro --jsonl-out "$GEPA_JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    # Threaded through in case run_03c/run_04c are invoked below without task dirs already
    # present (e.g. first run on a fresh TASKS_OUT) -- see run_03c's own SKIP_*_BUILD logic.
    export FILTER_CORRECT MIN_SAMPLES
fi

echo
echo "############################################"
echo "### phase 3c: training & validation (v3 dataset)"
echo "############################################"
bash run_03c_training_validation_v3.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 4c: downstream accuracy eval (v3 dataset, small Q-holdout + full official"
echo "### test sets, both incl. oracle, all successful tasks from both algorithms)"
echo "############################################"
# Needs a trained v3 hypernet checkpoint -- phase 3c writes the warm-started SFT arm to
# outputs/checkpoints/sft_warmstart_v3/latest.pt by default; override via HYPERNET_CKPT if
# your phase-3c run used different --out paths (same env vars run_04c_downstream_eval_v3.sh
# itself reads: HYPERNET_CKPT, ORACLE_DIR, OUT, OUT_FULL, GEN_BATCH_SIZE, FORCE).
bash run_04c_downstream_eval_v3.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### v3 pipeline complete"
echo "############################################"
