#!/usr/bin/env bash
# Phase 3, v2 dataset: same pipeline as run_03_training_validation.sh, but pointed at the
# textgrad_repro_v2 tasks (data/textgrad_repro/ converted by
# scripts/build_tasks_from_textgrad_repro_v2.py -- no <think> tokens, 10 domains instead of
# gsm8k-only). See docs/03_training_validation.md's 2026-08-04 changelog entry.
#
#   bash run_03b_training_validation_v2.sh            # lint + full pytest suite (same as
#                                                       # run_03_training_validation.sh -- no
#                                                       # v2-specific tests, same code path)
#   bash run_03b_training_validation_v2.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                       # build_tasks -> make_splits ->
#                                                       # train_oracle_loras -> canonicalize ->
#                                                       # train_recon -> train_sft (x2) ->
#                                                       # run_ablation. Long-running.
#
# Safe to re-run --full after an interruption: every stage script gracefully skips work
# already on disk (same skip logic as run_03_training_validation.sh -- see that file's
# comment). Pass --force to any individual script below to redo that stage anyway.
#
# Writes to *_v2-suffixed paths only -- outputs/oracle_loras_v2, outputs/oracle_loras_canon_v2,
# outputs/checkpoints/{recon,sft_scratch,sft_warmstart}_v2, data/splits_v2.json -- so the
# original run's checkpoints and data/splits.json are never touched. The source dataset at
# data/textgrad_repro/ is read-only to this whole script.
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
    JSONL_OUT="${JSONL_OUT:-/home/dg793/text-to-lora/data/textgrad_repro_v2_t2l}"
    TRAIN_TASKS="${TRAIN_TASKS:-textgrad_repro_v2_*}"
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    # FILTER_CORRECT/MIN_SAMPLES only matter the first time this script actually builds tasks
    # (see SKIP_BUILD below) -- run_all_v2.sh builds tasks itself before calling this script,
    # so these are here for when this script is invoked directly instead of via the wrapper.
    FILTER_CORRECT="${FILTER_CORRECT:-1}"
    MIN_SAMPLES="${MIN_SAMPLES:-50}"

    echo "--- build tasks from data/textgrad_repro/ (skips a task dir if its metadata.yaml already exists)"
    for task_dir in "$TASKS_OUT"/textgrad_repro_v2_*; do
        [[ -d "$task_dir" ]] && { echo "  tasks already built under $TASKS_OUT, skipping build step"; SKIP_BUILD=1; break; }
    done
    if [[ "${SKIP_BUILD:-0}" -ne 1 ]]; then
        FILTER_FLAG="--filter-correct"
        [[ "$FILTER_CORRECT" -eq 0 ]] && FILTER_FLAG="--no-filter-correct"
        uv run --no-sync python scripts/build_tasks_from_textgrad_repro_v2.py \
            --src-root data/textgrad_repro --jsonl-out "$JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    echo
    echo "--- splits"
    uv run --no-sync python scripts/make_splits.py --tasks-root "$TASKS_OUT" \
        --train-tasks "$TRAIN_TASKS" --seed 0 --out data/splits_v2.json

    echo
    echo "--- oracle LoRAs (sequential here; shard externally via --tasks for parallel jobs)"
    uv run --no-sync python scripts/train_oracle_loras.py --config configs/oracle.yaml \
        --data-config configs/data_v2.yaml --tasks-root "$TASKS_OUT" --train-tasks "$TRAIN_TASKS" \
        --target-dir "$TARGET_DIR" --splits data/splits_v2.json --out outputs/oracle_loras_v2

    echo
    echo "--- canonicalize oracles"
    uv run --no-sync python scripts/canonicalize_oracles.py --oracle-dir outputs/oracle_loras_v2 \
        --target-dir "$TARGET_DIR" --out outputs/oracle_loras_canon_v2

    echo
    echo "--- recon warm-start"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras_v2 --tasks-root "$TASKS_OUT" --train-tasks "$TRAIN_TASKS" \
        --splits data/splits_v2.json --out outputs/checkpoints/recon_v2

    echo
    echo "--- SFT ablation: from-scratch"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "$TRAIN_TASKS" --splits data/splits_v2.json \
        --data-config configs/data_v2.yaml --oracle-dir outputs/oracle_loras_v2 \
        --out outputs/checkpoints/sft_scratch_v2

    echo
    echo "--- SFT ablation: recon-warm-started"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "$TRAIN_TASKS" --splits data/splits_v2.json \
        --data-config configs/data_v2.yaml --oracle-dir outputs/oracle_loras_v2 \
        --init-from outputs/checkpoints/recon_v2/latest.pt --out outputs/checkpoints/sft_warmstart_v2

    echo
    echo "--- ablation report"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch_v2/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart_v2/latest.pt
fi

echo
echo "=== phase 3 (v2 dataset) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_03b_training_validation_v2.sh --full"
fi
