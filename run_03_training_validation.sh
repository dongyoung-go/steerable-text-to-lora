#!/usr/bin/env bash
# Phase 3: training & validation pipeline. See docs/03_training_validation.md.
#
#   bash run_03_training_validation.sh            # lint + full pytest suite (CPU-only,
#                                                   # tiny synthetic fixtures, no network,
#                                                   # no GPU, no real weights)
#   bash run_03_training_validation.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                   # profile_lengths -> make_splits ->
#                                                   # train_oracle_loras -> canonicalize_oracles
#                                                   # -> train_recon -> train_sft (x2: scratch,
#                                                   # warmstart) -> run_ablation. Long-running.
#
# There is no synthetic end-to-end driver script here (unlike run_02_model.sh's --full smoke
# check): confidence in the CPU-only default path comes entirely from the per-module unit
# test suite (tests/test_data_*.py, test_validation.py, test_canonicalize.py,
# test_train_oracle.py, test_recon_*.py, test_sft_*.py, test_grad_accum_order.py,
# test_run_ablation.py). --full reads real task data in place from TASKS_ROOT -- nothing is
# copied into this repo -- and is a plain sequential bash script: no slurm, no DAG runner,
# same philosophy as run_01_env.sh / run_02_model.sh.
#
# Safe to re-run --full after an interruption (crash, killed job, accidentally shut-down node):
# every stage script below gracefully skips work that's already on disk --
# train_oracle_loras.py skips tasks with an existing adapter, canonicalize_oracles.py skips
# tasks with an existing .pt, train_recon.py/train_sft.py skip if their checkpoint already
# reached the configured step count, make_splits.py skips if splits.json already exists. Pass
# --force to any individual script to redo that stage anyway.
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

    TASKS_ROOT="${TASKS_ROOT:-/home/dg793/text-to-lora/tasks}"
    TRAIN_TASKS="${TRAIN_TASKS:-textgrad_repro_gsm8k_*}"
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"

    echo "--- length profiling (advisory -- hand-copy the recommendation into configs/data.yaml)"
    uv run --no-sync python scripts/profile_lengths.py --tasks-root "$TASKS_ROOT" \
        --train-tasks "$TRAIN_TASKS" --tokenizer "$TARGET_DIR"

    echo
    echo "--- splits"
    uv run --no-sync python scripts/make_splits.py --tasks-root "$TASKS_ROOT" \
        --train-tasks "$TRAIN_TASKS" --seed 0 --out data/splits.json

    echo
    echo "--- oracle LoRAs (sequential here; shard externally via --tasks for parallel jobs)"
    uv run --no-sync python scripts/train_oracle_loras.py --config configs/oracle.yaml \
        --data-config configs/data.yaml --tasks-root "$TASKS_ROOT" --train-tasks "$TRAIN_TASKS" \
        --target-dir "$TARGET_DIR" --splits data/splits.json --out outputs/oracle_loras

    echo
    echo "--- canonicalize oracles"
    uv run --no-sync python scripts/canonicalize_oracles.py --oracle-dir outputs/oracle_loras \
        --target-dir "$TARGET_DIR" --out outputs/oracle_loras_canon

    echo
    echo "--- recon warm-start"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras --tasks-root "$TASKS_ROOT" --train-tasks "$TRAIN_TASKS" \
        --splits data/splits.json --out outputs/checkpoints/recon

    echo
    echo "--- SFT ablation: from-scratch"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "$TRAIN_TASKS" --splits data/splits.json \
        --data-config configs/data.yaml --oracle-dir outputs/oracle_loras \
        --out outputs/checkpoints/sft_scratch

    echo
    echo "--- SFT ablation: recon-warm-started"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_ROOT" --train-tasks "$TRAIN_TASKS" --splits data/splits.json \
        --data-config configs/data.yaml --oracle-dir outputs/oracle_loras \
        --init-from outputs/checkpoints/recon/latest.pt --out outputs/checkpoints/sft_warmstart

    echo
    echo "--- ablation report"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart/latest.pt
fi

echo
echo "=== phase 3 complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_03_training_validation.sh --full"
fi
echo "next: see docs/03_training_validation.md changelog for what was verified"
