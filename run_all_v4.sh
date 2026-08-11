#!/usr/bin/env bash
# The whole v4 experiment, end to end: derive a comprehensive-feedback chain from
# data/textgrad_repro/ (scripts/generate_comprehensive_feedback_v4.py) -> build task dirs from it
# (scripts/build_tasks_from_comprehensive_feedback_v4.py) -> phase 3 (splits/oracle/canon/recon/
# SFT x2/ablation) -> phase 4 (downstream accuracy eval). Mirrors run_all_v3.sh's shape but for
# the comprehensive-feedback-as-T2L-input experiment (run_03_training_validation_v4.sh /
# run_04_downstream_eval_v4.sh). See docs/05_comprehensive_feedback_v4.md for the full design.
#
# What's different vs v3: T2L is conditioned on a generalized "comprehensive feedback" paragraph
# instead of the literal TextGrad-optimized prompt text, derived from the same
# previous-state + this-round's-3-textual-gradients -> new-state recipe (via one Qwen3-14B call
# per accepted round), but framed as reusable guidance rather than an instruction. The LoRA
# target (question -> response) is unchanged. GEPA is out of scope here -- this experiment is
# TextGrad-only, since GEPA's reflective-mutation mechanism doesn't produce the "previous prompt +
# 3 textual feedback" structure this recipe depends on.
#
# This script and everything it calls is READ-ONLY w.r.t. data/textgrad_repro/ and writes only to
# brand-new, disjoint, _v4-suffixed paths (data/textgrad_repro_comprehensive_feedback_v4/,
# tasks/comprehensive_feedback_v4_*, data/splits_v4.json, outputs/*_v4, outputs/checkpoints/*_v4,
# outputs/eval/*_v4.json) -- nothing about the v3 experiment (its data, task dirs, checkpoints, or
# eval outputs) is read, written, or otherwise affected by running this script.
#
#   bash run_all_v4.sh                        # lint + tests only for both phases (CPU-safe)
#   bash run_all_v4.sh --full                  # ... PLUS every real, long-running stage.
#                                               # Hours-long -- RUN ON THE B200 GPU NODE, NOT
#                                               # this CPU node. Nothing in --full is CPU-safe:
#                                               # feedback generation (Qwen3-14B via vLLM),
#                                               # oracle/recon/SFT training, and real generation
#                                               # for downstream eval all need GPU.
#
# Safe to re-run --full after an interruption: every downstream stage script skips work already
# on disk (see run_03_training_validation_v4.sh's and run_04_downstream_eval_v4.sh's own
# headers, and generate_comprehensive_feedback_v4.py's per-source-dir skip check).
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
echo "### phase 3 (v4 experiment): comprehensive-feedback generation + task build, then"
echo "### training & validation"
echo "############################################"
bash run_03_training_validation_v4.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### phase 4 (v4 experiment): downstream accuracy eval (small Q-holdout + full official"
echo "### test sets, incl. oracle, all successful comprehensive-feedback task groups)"
echo "############################################"
# Needs a trained v4 hypernet checkpoint -- phase 3 writes the warm-started SFT arm to
# outputs/checkpoints/sft_warmstart_v4/latest.pt by default; override via HYPERNET_CKPT if your
# phase-3 run used different --out paths (same env vars run_04_downstream_eval_v4.sh itself
# reads: HYPERNET_CKPT, ORACLE_DIR, OUT, OUT_FULL, GEN_BATCH_SIZE, FORCE).
bash run_04_downstream_eval_v4.sh "${FULL_FLAG[@]}"

echo
echo "############################################"
echo "### v4 experiment complete"
echo "############################################"
