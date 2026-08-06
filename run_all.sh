#!/usr/bin/env bash
# The whole pipeline, phases 1-4, end to end. See docs/01-04_*.md.
#
#   bash run_all.sh            # env setup + lint/tests for every phase (CPU-safe, ~1-2 min)
#   bash run_all.sh --full     # ... PLUS every real, long-running stage on a GPU node:
#                                # phase 2's real Qwen2.5 self-check, phase 3's real oracle
#                                # LoRAs -> canonicalize -> recon warm-start -> SFT (scratch +
#                                # warmstart) -> ablation report, and phase 4's real
#                                # downstream accuracy eval against the warm-started SFT
#                                # checkpoint. Hours-long; run this on the B200 node, not the
#                                # CPU login node (phase 3 needs a GPU to make progress at
#                                # all; phase 4 needs one for the pinned flash-attn2 kernel).
#
# This is a thin sequential wrapper around run_01_env.sh/run_02_model.sh/
# run_03_training_validation.sh/run_04_downstream_eval.sh -- it adds no logic of its own
# beyond chaining them and threading --full through the three that accept it. Every
# individual stage is independently safe to re-run (each skips work already on disk; see
# each script's own header), so re-running run_all.sh --full after an interruption resumes
# roughly where it left off rather than redoing everything.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1
FULL_FLAG=()
[[ $FULL -eq 1 ]] && FULL_FLAG=(--full)

echo "############################################"
echo "### phase 1: environment"
echo "############################################"
bash run_01_env.sh

echo
echo "############################################"
echo "### phase 2: model"
echo "############################################"
bash run_02_model.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 3: training & validation"
echo "############################################"
bash run_03_training_validation.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 4: downstream accuracy eval"
echo "############################################"
# Phase 4 needs a trained hypernet checkpoint -- phase 3 writes the warm-started SFT arm to
# outputs/checkpoints/sft_warmstart/latest.pt by default; override via HYPERNET_CKPT if your
# phase-3 run used different --out paths (e.g. HYPERNET_CKPT, ORACLE_DIR, OUT, same as
# run_04_downstream_eval.sh's own env vars).
bash run_04_downstream_eval.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### all phases complete"
echo "############################################"
