#!/usr/bin/env bash
# Phase 3, v3 dataset: same pipeline as run_03b_training_validation_v2.sh, but builds tasks from
# BOTH data/textgrad_repro/ (scripts/build_tasks_from_textgrad_repro_v3.py) and data/gepa_repro/
# (scripts/build_tasks_from_gepa_repro_v3.py), and trains across both task namespaces at once
# (textgrad_repro_v3_* and gepa_repro_v3_*, kept as separate tasks/LoRAs even where the task name
# overlaps between the two algorithms). v3's builders pool rows across iterations that reused an
# identical winning prompt/candidate (reverted textgrad rounds, repeated gepa candidates) instead
# of using only a single best iteration -- see the builders' docstrings and run_all_v3.sh's header.
#
#   bash run_03c_training_validation_v3.sh            # lint + full pytest suite (same as
#                                                       # run_03b_training_validation_v2.sh -- no
#                                                       # v3-specific tests, same code path)
#   bash run_03c_training_validation_v3.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                       # build_tasks (both sources) ->
#                                                       # make_splits -> train_oracle_loras ->
#                                                       # canonicalize -> train_recon ->
#                                                       # train_sft (x2) -> run_ablation.
#                                                       # Long-running.
#
# Safe to re-run --full after an interruption: every stage script gracefully skips work already
# on disk (same skip logic as run_03b_training_validation_v2.sh). Pass --force to any individual
# script below to redo that stage anyway.
#
# Writes to *_v3-suffixed paths only -- outputs/oracle_loras_v3, outputs/oracle_loras_canon_v3,
# outputs/checkpoints/{recon,sft_scratch,sft_warmstart}_v3, data/splits_v3.json -- so v1/v2's
# checkpoints and splits files are never touched. data/textgrad_repro/ and data/gepa_repro/ are
# read-only to this whole script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [[ ! -d .venv ]]; then
    echo "error: no .venv -- run 'bash run_01_env.sh' first" >&2
    exit 1
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
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

    TASKS_OUT="${TASKS_OUT:-/home/dg793/text-to-lora/tasks}"
    TG_JSONL_OUT="${TG_JSONL_OUT:-/home/dg793/text-to-lora/data/textgrad_repro_v3_t2l}"
    GEPA_JSONL_OUT="${GEPA_JSONL_OUT:-/home/dg793/text-to-lora/data/gepa_repro_v3_t2l}"
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(textgrad_repro_v3_* gepa_repro_v3_*)
    fi
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    # FILTER_CORRECT/MIN_SAMPLES only matter the first time this script actually builds tasks
    # (see the SKIP_BUILD checks below) -- run_all_v3.sh builds tasks itself before calling this
    # script, so these are here for when this script is invoked directly instead of via the
    # wrapper.
    FILTER_CORRECT="${FILTER_CORRECT:-1}"
    MIN_SAMPLES="${MIN_SAMPLES:-50}"
    FILTER_FLAG="--filter-correct"
    [[ "$FILTER_CORRECT" -eq 0 ]] && FILTER_FLAG="--no-filter-correct"

    echo "--- build tasks from data/textgrad_repro/ (skips if textgrad_repro_v3_* dirs already exist)"
    SKIP_TG_BUILD=0
    for task_dir in "$TASKS_OUT"/textgrad_repro_v3_*; do
        [[ -d "$task_dir" ]] && { echo "  textgrad_repro_v3_* tasks already built under $TASKS_OUT, skipping"; SKIP_TG_BUILD=1; break; }
    done
    if [[ $SKIP_TG_BUILD -eq 0 ]]; then
        uv run --no-sync python scripts/build_tasks_from_textgrad_repro_v3.py \
            --src-root data/textgrad_repro --jsonl-out "$TG_JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    echo "--- build tasks from data/gepa_repro/ (skips if gepa_repro_v3_* dirs already exist)"
    SKIP_GEPA_BUILD=0
    for task_dir in "$TASKS_OUT"/gepa_repro_v3_*; do
        [[ -d "$task_dir" ]] && { echo "  gepa_repro_v3_* tasks already built under $TASKS_OUT, skipping"; SKIP_GEPA_BUILD=1; break; }
    done
    if [[ $SKIP_GEPA_BUILD -eq 0 ]]; then
        uv run --no-sync python scripts/build_tasks_from_gepa_repro_v3.py \
            --src-root data/gepa_repro --jsonl-out "$GEPA_JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    echo
    echo "--- splits"
    uv run --no-sync python scripts/make_splits.py --tasks-root "$TASKS_OUT" \
        --train-tasks "${TRAIN_TASKS_ARR[@]}" --seed 0 --out data/splits_v3.json

    echo
    echo "--- oracle LoRAs (sequential here; shard externally via --tasks for parallel jobs)"
    uv run --no-sync python scripts/train_oracle_loras.py --config configs/oracle.yaml \
        --data-config configs/data_v3.yaml --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --target-dir "$TARGET_DIR" --splits data/splits_v3.json --out outputs/oracle_loras_v3

    echo
    echo "--- canonicalize oracles"
    uv run --no-sync python scripts/canonicalize_oracles.py --oracle-dir outputs/oracle_loras_v3 \
        --target-dir "$TARGET_DIR" --out outputs/oracle_loras_canon_v3

    echo
    echo "--- recon warm-start"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras_v3 --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --splits data/splits_v3.json --out outputs/checkpoints/recon_v3

    echo
    echo "--- SFT ablation: from-scratch"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --data-config configs/data_v3.yaml --oracle-dir outputs/oracle_loras_v3 \
        --out outputs/checkpoints/sft_scratch_v3

    echo
    echo "--- SFT ablation: recon-warm-started"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --data-config configs/data_v3.yaml --oracle-dir outputs/oracle_loras_v3 \
        --init-from outputs/checkpoints/recon_v3/best.pt --out outputs/checkpoints/sft_warmstart_v3

    echo
    echo "--- ablation report"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch_v3/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart_v3/latest.pt
fi

echo
echo "=== phase 3 (v3 dataset) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_03c_training_validation_v3.sh --full"
fi
