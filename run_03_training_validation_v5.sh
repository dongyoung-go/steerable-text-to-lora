#!/usr/bin/env bash
# Phase 3, v5 experiment: same pipeline shape as run_03_training_validation_v4.sh, but trains on
# the description-paraphrase-augmented copies of v3's task dirs (textgrad_repro_v5_*/gepa_repro_v5_*
# -- see docs/06_description_augmentation_v5.md and scripts/paraphrase_descs.py). No task-building
# stage here: those task dirs already exist on disk (a v3 collapse-diagnosis session's
# scripts/paraphrase_descs.py run built them by copying v3's pristine task dirs and appending
# paraphrased descriptions -- v3 itself was left untouched).
#
# Oracle LoRAs are NOT retrained here. v5's task dirs share v3's exact (question, response) rows
# (only `descriptions` differs), so v3's already-trained oracle LoRAs are numerically identical to
# what training against v5 would produce (docs/06's own "What's not built yet" section calls this
# out explicitly). scripts/reuse_oracle_loras.py symlinks outputs/oracle_loras_v5/<v5_task_name> ->
# outputs/oracle_loras_v3/<v3_task_name> (and the canonicalized .pt equivalents) instead of running
# train_oracle_loras.py/canonicalize_oracles.py -- this requires outputs/oracle_loras_v3 and
# outputs/oracle_loras_canon_v3 to already exist (from a prior `bash run_03c_training_validation_v3.sh
# --full`) and fails loudly if any v5 task has no v3 counterpart.
#
#   bash run_03_training_validation_v5.sh            # lint + full pytest suite (same code path
#                                                       # as run_03c/run_03 v3/v4 -- no
#                                                       # v5-specific tests besides
#                                                       # test_reuse_oracle_loras.py)
#   bash run_03_training_validation_v5.sh --full      # RUN THIS MANUALLY ON THE B200 NODE.
#                                                       # make_splits -> reuse_oracle_loras ->
#                                                       # train_recon -> train_sft (x2) ->
#                                                       # run_ablation. Long-running (recon/SFT
#                                                       # still train for real; only oracle
#                                                       # training is skipped).
#
# Safe to re-run --full after an interruption: make_splits/train_recon/train_sft skip work already
# on disk, same as v3/v4; reuse_oracle_loras.py leaves existing symlinks alone unless --force is
# added to its own invocation below.
#
# Writes to *_v5-suffixed paths only -- outputs/oracle_loras_v5, outputs/oracle_loras_canon_v5
# (symlink trees, see above), outputs/checkpoints/{recon,sft_scratch,sft_warmstart}_v5,
# data/splits_v5.json -- so v1/v2/v3/v4's checkpoints, splits files, and task dirs are never
# touched. outputs/oracle_loras_v3 and outputs/oracle_loras_canon_v3 are read-only to this whole
# script (only ever symlinked into, never written into).
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
    if [[ -n "${TRAIN_TASKS:-}" ]]; then
        read -ra TRAIN_TASKS_ARR <<< "$TRAIN_TASKS"
    else
        TRAIN_TASKS_ARR=(textgrad_repro_v5_* gepa_repro_v5_*)
    fi
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"
    V3_ORACLE_DIR="${V3_ORACLE_DIR:-outputs/oracle_loras_v3}"
    V3_CANON_DIR="${V3_CANON_DIR:-outputs/oracle_loras_canon_v3}"

    if [[ ! -d "$V3_ORACLE_DIR" || ! -d "$V3_CANON_DIR" ]]; then
        echo "error: $V3_ORACLE_DIR / $V3_CANON_DIR not found -- run 'bash run_03c_training_validation_v3.sh --full' first (v5 reuses v3's oracle LoRAs, see docs/06)" >&2
        exit 1
    fi

    echo "--- splits (this is the step that finally exercises v5's >1 description per task)"
    uv run --no-sync python scripts/make_splits.py --tasks-root "$TASKS_OUT" \
        --train-tasks "${TRAIN_TASKS_ARR[@]}" --seed 0 --out data/splits_v5.json

    echo
    echo "--- reuse v3's oracle LoRAs under v5 task names (no retraining -- see header)"
    uv run --no-sync python scripts/reuse_oracle_loras.py --tasks-root "$TASKS_OUT" \
        --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v5.json \
        --source-oracle-dir "$V3_ORACLE_DIR" --source-canon-dir "$V3_CANON_DIR" \
        --out-oracle-dir outputs/oracle_loras_v5 --out-canon-dir outputs/oracle_loras_canon_v5 \
        --from-substr _v5_ --to-substr _v3_

    echo
    echo "--- recon warm-start"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras_v5 --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --splits data/splits_v5.json --out outputs/checkpoints/recon_v5

    echo
    echo "--- SFT ablation: from-scratch"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v5.json \
        --data-config configs/data_v5.yaml --oracle-dir outputs/oracle_loras_v5 \
        --out outputs/checkpoints/sft_scratch_v5

    echo
    echo "--- SFT ablation: recon-warm-started"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v5.json \
        --data-config configs/data_v5.yaml --oracle-dir outputs/oracle_loras_v5 \
        --init-from outputs/checkpoints/recon_v5/best.pt --out outputs/checkpoints/sft_warmstart_v5

    echo
    echo "--- ablation report"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch_v5/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart_v5/latest.pt
fi

echo
echo "=== phase 3 (v5 experiment) complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_03_training_validation_v5.sh --full"
fi
