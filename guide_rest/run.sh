#!/bin/bash
# Thin env-var wrapper around round_loop.py. Same ephemeral-overlay pattern as
# scripts/textgrad_repro_run.sh / scripts/gepa_repro_run.sh: `uv run --with ...`, never
# synced into pyproject.toml/uv.lock/the persistent .venv. vllm/transformers/kernels pins
# copied verbatim from those scripts (this machine's CUDA 12.8 driver needs vllm==0.11.0,
# which in turn needs transformers<5 -- see textgrad_repro_README.md's "why this is pinned
# much harder" section for the full chain). `peft` resolves inside the same pinned overlay
# (round_loop.py's children -- sampling.py/feedback.py/eval_heldout.py use vllm directly;
# train.py uses peft/transformers -- all need to share one consistent transformers version,
# not two). `math-verify` is guide_rest's one new dependency, for MATH answer-equivalence
# checking (no existing verifier in this repo handles MATH-style answers).
#
# Example smoke test (tiny T/k/pool, gsm8k only, both conditions):
#   ROUNDS=1 K=2 GROW_POOL_SIZE=8 DEV_POOL_SIZE=8 DEV_K=2 HELDOUT_SIZE=8 N=2 EPOCHS=1 TASK=gsm8k ./guide_rest/run.sh
#
# Full run (GROW_POOL_SIZE unset by default -- uses the entire train split minus the dev
# pool each round, matching ReST-EM's own setup: ~7423 questions for gsm8k, ~7450 for math):
#   TASK=gsm8k ./guide_rest/run.sh
#   TASK=math ./guide_rest/run.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TASK="${TASK:-gsm8k}"
CONDITION="${CONDITION:-both}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
ROUNDS="${ROUNDS:-5}"
K="${K:-8}"
GROW_POOL_SIZE="${GROW_POOL_SIZE:-}"   # empty = full train split minus dev pool (ReST-EM's own setup)
DEV_POOL_SIZE="${DEV_POOL_SIZE:-50}"
DEV_K="${DEV_K:-4}"
HELDOUT_SIZE="${HELDOUT_SIZE:-200}"
N="${N:-8}"
MAX_WORDS="${MAX_WORDS:-150}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # measured throughput sweet spot on 1x B200; see train.py's --batch_size help
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SEED="${SEED:-0}"

GROW_POOL_SIZE_ARG=()
if [ -n "$GROW_POOL_SIZE" ]; then
  GROW_POOL_SIZE_ARG=(--grow_pool_size "$GROW_POOL_SIZE")
fi

uv run \
  --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
  --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
  --with peft --with math-verify \
  python guide_rest/round_loop.py \
  --task "$TASK" \
  --condition "$CONDITION" \
  --base_model "$BASE_MODEL" \
  --rounds "$ROUNDS" \
  --k "$K" \
  "${GROW_POOL_SIZE_ARG[@]}" \
  --dev_pool_size "$DEV_POOL_SIZE" \
  --dev_k "$DEV_K" \
  --heldout_size "$HELDOUT_SIZE" \
  --n "$N" \
  --max_words "$MAX_WORDS" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN" \
  --seed "$SEED"

echo "done. results in data/guide_rest/${TASK}/{A,B}/summary.jsonl"
