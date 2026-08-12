#!/usr/bin/env bash
# Phase 3, v4 experiment: same pipeline shape as run_03c_training_validation_v3.sh, but trains on
# comprehensive-feedback task dirs (scripts/build_tasks_from_comprehensive_feedback_v4.py) instead
# of the prompt-text-based textgrad_repro_v3/gepa_repro_v3 task dirs. Building those task dirs is
# itself a two-stage process specific to v4:
#   1. scripts/generate_comprehensive_feedback_v4.py (GPU/vLLM, Qwen3-14B) reads each TextGrad run's
#      iterations.jsonl and derives a comprehensive, generalized feedback chain -- see that
#      script's and docs/05_comprehensive_feedback_v4.md's docstrings for the full algorithm.
#   2. scripts/build_tasks_from_comprehensive_feedback_v4.py (CPU) joins that chain against
#      forward_outputs.jsonl and writes comprehensive_feedback_v4_<task>_d<K> task dirs, one per
#      distinct feedback text, same shape as v3's task dirs.
#
#   bash run_03_training_validation_v4.sh            # lint + full pytest suite (same code path
#                                                       # as run_03c -- no v4-specific tests)
#   bash run_03_training_validation_v4.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                       # generate_comprehensive_feedback_v4 ->
#                                                       # build_tasks_from_comprehensive_feedback_v4
#                                                       # -> make_splits -> train_oracle_loras ->
#                                                       # canonicalize -> train_recon -> train_sft
#                                                       # (x2) -> run_ablation. Long-running.
#
# Safe to re-run --full after an interruption: every stage script skips work already on disk
# (generate_comprehensive_feedback_v4.py skips any task whose comprehensive_feedback_v4.jsonl
# already exists; the rest inherit the same skip logic as run_03c_training_validation_v3.sh).
# Pass --force to any individual script below to redo that stage anyway.
#
# Writes to *_v4-suffixed paths only -- outputs/oracle_loras_v4, outputs/oracle_loras_canon_v4,
# outputs/checkpoints/{recon,sft_scratch,sft_warmstart}_v4, data/splits_v4.json,
# data/textgrad_repro_comprehensive_feedback_v4/ -- so v1/v2/v3's checkpoints, splits files, and
# task dirs are never touched. data/textgrad_repro/ is read-only to this whole script (both new
# stage scripts only ever read iterations.jsonl / forward_outputs.jsonl from it).
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
    FEEDBACK_ROOT="${FEEDBACK_ROOT:-data/textgrad_repro_comprehensive_feedback_v4}"
    CF_JSONL_OUT="${CF_JSONL_OUT:-/home/dg793/text-to-lora/data/comprehensive_feedback_v4_t2l}"
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(comprehensive_feedback_v4_*)
    fi
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    FEEDBACK_MODEL="${FEEDBACK_MODEL:-Qwen/Qwen3-14B}"
    FILTER_CORRECT="${FILTER_CORRECT:-1}"
    MIN_SAMPLES="${MIN_SAMPLES:-50}"
    FILTER_FLAG="--filter-correct"
    [[ "$FILTER_CORRECT" -eq 0 ]] && FILTER_FLAG="--no-filter-correct"

    echo "--- generate comprehensive feedback (skips any source dir with a "
    echo "    comprehensive_feedback_v4.jsonl already present under $FEEDBACK_ROOT)"
    # Ephemeral uv overlay, not --no-sync -- vllm is not in the persistent .venv (deliberately:
    # see pyproject.toml's "gen" extra comment). Same pins as scripts/textgrad_repro_run.sh /
    # guide_rest/run.sh: this machine's CUDA 12.8 driver needs vllm==0.11.0 specifically (newer
    # releases link against CUDA 13), which in turn needs transformers<5 (and a huggingface_hub-
    # compatible kernels release) -- none of this touches pyproject.toml/uv.lock/the persistent
    # .venv, it's resolved fresh into an overlay each invocation (cached after the first run).
    uv run \
        --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
        --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
        python scripts/generate_comprehensive_feedback_v4.py \
        --src-root data/textgrad_repro --out-root "$FEEDBACK_ROOT" --model "$FEEDBACK_MODEL"

    echo "--- build tasks from comprehensive feedback (skips if comprehensive_feedback_v4_* dirs already exist)"
    SKIP_CF_BUILD=0
    for task_dir in "$TASKS_OUT"/comprehensive_feedback_v4_*; do
        [[ -d "$task_dir" ]] && { echo "  tasks already built under $TASKS_OUT, skipping build step"; SKIP_CF_BUILD=1; break; }
    done
    if [[ $SKIP_CF_BUILD -eq 0 ]]; then
        uv run --no-sync python scripts/build_tasks_from_comprehensive_feedback_v4.py \
            --src-root data/textgrad_repro --feedback-root "$FEEDBACK_ROOT" \
            --jsonl-out "$CF_JSONL_OUT" --tasks-out "$TASKS_OUT" \
            "$FILTER_FLAG" --min-samples "$MIN_SAMPLES"
    fi

    echo
    echo "--- splits"
    uv run --no-sync python scripts/make_splits.py --tasks-root "$TASKS_OUT" \
        --train-tasks "${TRAIN_TASKS_ARR[@]}" --seed 0 --out data/splits_v4.json

    echo
    echo "--- oracle LoRAs (sequential here; shard externally via --tasks for parallel jobs)"
    uv run --no-sync python scripts/train_oracle_loras.py --config configs/oracle.yaml \
        --data-config configs/data_v4.yaml --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --target-dir "$TARGET_DIR" --splits data/splits_v4.json --out outputs/oracle_loras_v4

    echo
    echo "--- canonicalize oracles"
    uv run --no-sync python scripts/canonicalize_oracles.py --oracle-dir outputs/oracle_loras_v4 \
        --target-dir "$TARGET_DIR" --out outputs/oracle_loras_canon_v4

    echo
    echo "--- recon warm-start"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras_v4 --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --splits data/splits_v4.json --out outputs/checkpoints/recon_v4

    echo
    echo "--- SFT ablation: from-scratch"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v4.json \
        --data-config configs/data_v4.yaml --oracle-dir outputs/oracle_loras_v4 \
        --out outputs/checkpoints/sft_scratch_v4

    echo
    echo "--- SFT ablation: recon-warm-started"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v4.json \
        --data-config configs/data_v4.yaml --oracle-dir outputs/oracle_loras_v4 \
        --init-from outputs/checkpoints/recon_v4/best.pt --out outputs/checkpoints/sft_warmstart_v4

    echo
    echo "--- ablation report"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch_v4/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart_v4/latest.pt
fi

echo
echo "=== phase 3 (v4 experiment) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_03_training_validation_v4.sh --full"
fi
