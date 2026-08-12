#!/usr/bin/env bash
# The whole v5 experiment, end to end: phase 3 (splits/oracle-reuse/recon/SFT x2/ablation) -> phase
# 4 (downstream accuracy eval). Mirrors run_all_v4.sh's shape but for the description-paraphrase-
# augmentation experiment (run_03_training_validation_v5.sh / run_04_downstream_eval_v5.sh). See
# docs/06_description_augmentation_v5.md for the full design and motivation.
#
# No task-build block here (unlike run_all_v3.sh): v5's task dirs
# (textgrad_repro_v5_*/gepa_repro_v5_*) already exist on disk, built by a prior
# scripts/paraphrase_descs.py run that copied v3's pristine task dirs and appended paraphrased
# descriptions. What's different vs v3: same (question, response) rows, but each task now carries
# up to 8 phrasings of its description instead of exactly 1 -- this is meant to test whether that
# fixes the description-independent-LoRA collapse diagnosed against v3 (see docs/06). Oracle LoRAs
# are reused from v3 rather than retrained (numerically identical, only `descriptions` differs --
# see run_03_training_validation_v5.sh's header for how).
#
# This script and everything it calls is READ-ONLY w.r.t. v3's task dirs, outputs/oracle_loras_v3,
# and outputs/oracle_loras_canon_v3, and writes only to brand-new, disjoint, _v5-suffixed paths
# (data/splits_v5.json, outputs/oracle_loras_v5 + canon (symlink trees), outputs/checkpoints/*_v5,
# outputs/eval/*_v5.json) -- nothing about v1/v2/v3/v4 is read, written, or otherwise affected by
# running this script.
#
#   bash run_all_v5.sh                        # lint + tests only for both phases (CPU-safe)
#   bash run_all_v5.sh --full                  # ... PLUS every real, long-running stage.
#                                               # Hours-long -- RUN ON THE B200 GPU NODE, NOT
#                                               # this CPU node. Needs v3's oracle LoRAs already
#                                               # trained (outputs/oracle_loras_v3,
#                                               # outputs/oracle_loras_canon_v3); recon/SFT
#                                               # training and real generation for downstream eval
#                                               # both need GPU.
#
# Safe to re-run --full after an interruption: every downstream stage script skips work already on
# disk (see run_03_training_validation_v5.sh's and run_04_downstream_eval_v5.sh's own headers).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

FULL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) FULL=1; shift ;;
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

echo
echo "############################################"
echo "### phase 3 (v5 experiment): splits, oracle-LoRA reuse, recon warm-start, SFT (x2), ablation"
echo "############################################"
bash run_03_training_validation_v5.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 4 (v5 experiment): downstream accuracy eval (small Q-holdout + full official"
echo "### test sets, incl. oracle, winning-instruction task scope -- same scope as v3's)"
echo "############################################"
# Needs a trained v5 hypernet checkpoint -- phase 3 writes the warm-started SFT arm to
# outputs/checkpoints/sft_warmstart_v5/latest.pt by default; override via HYPERNET_CKPT if your
# phase-3 run used different --out paths (same env vars run_04_downstream_eval_v5.sh itself reads:
# HYPERNET_CKPT, ORACLE_DIR, OUT, OUT_FULL, GEN_BATCH_SIZE, FORCE).
bash run_04_downstream_eval_v5.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### v5 experiment complete"
echo "############################################"
echo "next: python scripts/compare_downstream_eval.py outputs/eval/downstream_accuracy_full_v3.json outputs/eval/downstream_accuracy_full_v5.json --labels v3 v5"
