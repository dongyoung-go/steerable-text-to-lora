#!/usr/bin/env bash
# Phase 4, v3 dataset: same eval as run_04b_downstream_eval_v2.sh, but pointed at the v3
# hypernet checkpoint/splits/oracle dir produced by run_03c_training_validation_v3.sh --full, and
# defaulting to ALL successful textgrad_repro_v3_* and gepa_repro_v3_* task dirs (unlike v2's
# hardcoded 8-task allowlist) -- "successful" already means "survived the builders' own
# --min-samples filter", so no further curation is needed here. See docs/03/docs/04 for what
# v1/v2 were; v3 additionally pools reverted/repeated-prompt iterations (see run_all_v3.sh).
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
    # Default is ALL successful v3 tasks from both algorithms -- unlike v2's hardcoded 8-task
    # allowlist, there's no curated subset here: any task dir the builders wrote already passed
    # --min-samples, so a glob over both namespaces is exactly "every successful dataset". Set
    # TRAIN_TASKS to a space-separated list of patterns (glob or literal) to override, e.g. to
    # restrict to one algorithm only.
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(textgrad_repro_v3_* gepa_repro_v3_*)
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
    echo "--- downstream accuracy eval (v3, full official test sets, incl. oracle)"
    uv run --no-sync python scripts/eval_downstream_accuracy_full.py \
        --hypernet "$HYPERNET_CKPT" --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --oracle-dir "$ORACLE_DIR" --gen-batch-size "$GEN_BATCH_SIZE" --out "$OUT_FULL" \
        "${FORCE_FLAG[@]}"
fi

echo
echo "=== phase 4 (v3 dataset) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_04c_downstream_eval_v3.sh --full"
fi
echo "next: see docs/03_training_validation.md / docs/04_downstream_eval.md for what to log"
