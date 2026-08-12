#!/usr/bin/env bash
# Targeted re-run of ONLY the v3 recon + warmstart-SFT stages, against the recon-collapse fix in
# src/steerable_t2l/trainers/recon.py (warmup->cosine-decay LR instead of flat, gradient-norm
# clipping, and best.pt checkpoint selection by cosine_similarity) -- see
# docs/03_training_validation.md's 2026-08-11 dated update for the diagnosis: recon_v3 (and,
# independently, recon_v4) both learned real signal for several hundred steps then permanently
# collapsed to predicting ~0 after one large, undamped optimizer step re-triggered the same
# dead-gradient fixed point SteerableHyperLoRA._apply_zero_init's docstring warns about for step
# 0 -- and sft_warmstart_v3, warm-started from that collapsed checkpoint at a very low LR, never
# recovered real steering either.
#
# This is NOT the full v3 pipeline (that's run_all_v3.sh). Task-dir building, splits, oracle
# LoRAs, oracle canonicalization, and the from-scratch SFT arm (sft_scratch_v3) are all
# UNTOUCHED by this fix -- trainers/sft.py and every earlier stage are unchanged -- so redoing
# them would just burn GPU time to reproduce identical results. This script only force-retrains
# the two stages that actually changed:
#   1. recon_v3        (scripts/train_recon.py --force)     -- writes latest.pt AND best.pt
#   2. sft_warmstart_v3 (scripts/train_sft.py --force)       -- --init-from recon_v3/best.pt
#   3. ablation report (scripts/run_ablation.py)             -- scratch arm reused as-is
#   4. downstream eval on the fixed warmstart checkpoint, to FRESH output files (does not
#      overwrite outputs/eval/downstream_accuracy{,_full}_v3.json from the pre-fix run, so the
#      two can be compared directly) -- skip with --skip-eval.
#
#   bash run_all_warmupfix_v3.sh                  # lint + tests only (CPU-safe)
#   bash run_all_warmupfix_v3.sh --full            # RUN ON THE B200 NODE. Force-retrains
#                                                   # recon_v3 + sft_warmstart_v3, re-runs the
#                                                   # ablation report, then downstream eval.
#   bash run_all_warmupfix_v3.sh --full --skip-eval   # stop after the ablation report
#
# Prerequisite: run_all_v3.sh --full (or run_03c_training_validation_v3.sh --full) must have
# already completed at least once -- this script errors out early if outputs/oracle_loras_v3,
# outputs/oracle_loras_canon_v3, data/splits_v3.json, or sft_scratch_v3/latest.pt are missing,
# since it does not build any of those itself.
#
# Writes: outputs/checkpoints/recon_v3/{latest,best}.pt (overwritten in place -- the pre-fix
# recon_v3/latest.pt is NOT preserved separately; re-run recon_v3 from data/splits_v3.json etc.
# if you need the old collapsed checkpoint back for comparison), outputs/checkpoints/
# sft_warmstart_v3/latest.pt (overwritten in place), outputs/eval/downstream_accuracy{,_full}_
# warmstart_v3_warmupfix.json (new files, old v3 eval outputs untouched).
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
SKIP_EVAL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --skip-eval) SKIP_EVAL=1 ;;
        *) echo "error: unrecognized argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=== lint"
uv run --no-sync ruff check src tests scripts

echo
echo "=== tests"
uv run --no-sync python -m pytest tests -q

if [[ $FULL -eq 1 ]]; then
    echo
    echo "=== REAL run on B200 -- long-running, run manually and monitor ==="

    TASKS_OUT="${TASKS_OUT:-/home/dg793/text-to-lora/tasks}"
    TRAIN_TASKS_ARR=(textgrad_repro_v3_* gepa_repro_v3_*)
    TARGET_DIR="${TARGET_DIR:-Qwen/Qwen2.5-1.5B-Instruct}"

    for prereq in outputs/oracle_loras_v3 outputs/oracle_loras_canon_v3 data/splits_v3.json \
                  outputs/checkpoints/sft_scratch_v3/latest.pt; do
        if [[ ! -e "$prereq" ]]; then
            echo "error: $prereq not found -- run 'bash run_all_v3.sh --full' first (this script" >&2
            echo "       only redoes recon_v3 + sft_warmstart_v3, it does not build prerequisites)" >&2
            exit 1
        fi
    done

    echo "--- recon warm-start (force-retrain with the LR-decay/clipping/best.pt fix)"
    uv run --no-sync python scripts/train_recon.py --config configs/recon.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --oracle-dir outputs/oracle_loras_v3 --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" \
        --splits data/splits_v3.json --out outputs/checkpoints/recon_v3 --force

    echo
    echo "--- SFT ablation: recon-warm-started (force-retrain, init from the new best.pt)"
    uv run --no-sync accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml --target-dir "$TARGET_DIR" \
        --tasks-root "$TASKS_OUT" --train-tasks "${TRAIN_TASKS_ARR[@]}" --splits data/splits_v3.json \
        --data-config configs/data_v3.yaml --oracle-dir outputs/oracle_loras_v3 \
        --init-from outputs/checkpoints/recon_v3/best.pt --out outputs/checkpoints/sft_warmstart_v3 --force

    echo
    echo "--- ablation report (scratch arm reused as-is -- unaffected by this fix)"
    uv run --no-sync python scripts/run_ablation.py \
        --scratch outputs/checkpoints/sft_scratch_v3/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart_v3/latest.pt

    if [[ $SKIP_EVAL -eq 0 ]]; then
        echo
        echo "--- downstream eval on the fixed warmstart checkpoint (fresh output paths --"
        echo "    outputs/eval/downstream_accuracy{,_full}_v3.json from the pre-fix warmstart"
        echo "    run are left untouched, for direct before/after comparison)"
        HYPERNET_CKPT=outputs/checkpoints/sft_warmstart_v3/latest.pt \
        OUT=outputs/eval/downstream_accuracy_warmstart_v3_warmupfix.json \
        OUT_FULL=outputs/eval/downstream_accuracy_full_warmstart_v3_warmupfix.json \
        bash run_04c_downstream_eval_v3.sh --full
    fi
fi

echo
echo "=== v3 warmup-fix re-run complete"
if [[ $FULL -eq 0 ]]; then
    echo "real run (B200 only, long-running): bash run_all_warmupfix_v3.sh --full"
fi
