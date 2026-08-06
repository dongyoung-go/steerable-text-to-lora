#!/bin/bash
# Thin env-var wrapper around scripts/textgrad_repro.py. Ported from
# /home/dg793/text-to-lora/scripts/textgrad_repro_run.sh -- see
# textgrad_repro_README.md for what changed in the port.
#
# Runs via `uv run --with-editable ./textgrad_repro --with ...` -- an
# ephemeral overlay environment layered on top of this repo's own
# resolution, never written into pyproject.toml/uv.lock/the persistent
# .venv. First run resolves/downloads the overlay (~1 min); cached after
# that.
#
# vllm is pinned to 0.11.0 (not just `--extra gen`'s vllm>=0.11) because
# newer vllm releases ship a compiled `_C` extension linked against CUDA 13
# regardless of declared metadata deps -- confirmed this breaks all the way
# down to 0.21.0, not just the newest PyPI release (0.26.0) -- and this
# machine's driver caps out at CUDA 12.8 (`ImportError: libcudart.so.13`
# when `from vllm import LLM` first touches vllm.platforms.cuda). 0.11.0
# (this repo's own declared floor) pulls nvidia-*-cu12 packages instead and
# imports/runs cleanly. The explicit cu128 index + unsafe-best-match is
# still needed for torch itself: without it uv resolves a cu130-tagged
# torch wheel that imports fine but fails at first CUDA call
# (`RuntimeError: The NVIDIA driver ... is too old`).
#
# transformers and kernels are ALSO pinned older here (4.57.1 / 0.10.0)
# purely for this ephemeral overlay -- vllm 0.11.0's tokenizer code needs
# transformers<5 (it calls an attribute transformers 5.x removed), and
# transformers<5 in turn needs huggingface_hub<1.0, which conflicts with
# the persistent venv's kernels>=0.4,<0.16.0 (resolved against
# huggingface_hub>=1.0 during the normal `uv sync --extra attn`). Pinning
# an old-huggingface_hub-compatible kernels release too avoids that clash.
# None of this touches pyproject.toml/uv.lock/the persistent .venv -- the
# project's own transformers>=5.0 floor is untouched for everything else
# in this repo. See textgrad_repro_README.md.
#
# Example:
#   MODEL_DIR=Qwen/Qwen3-14B ENABLE_THINKING=0 MAX_EPOCHS=1 STEPS_PER_EPOCH=2 EVAL_TEST=0 \
#     ./scripts/textgrad_repro_run.sh
#
# TASK selects among the TASKS registry in textgrad_repro.py (default:
# gsm8k). See TEXTGRAD_MULTITASK_PLAN.md.
#   TASK=bbh_object_counting MAX_EPOCHS=1 STEPS_PER_EPOCH=1 EVAL_TEST=0 \
#     ./scripts/textgrad_repro_run.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3-14B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-4}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
OPTIMIZER_MAX_TOKENS="${OPTIMIZER_MAX_TOKENS:-8000}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TASK="${TASK:-gsm8k}"

DATA_DIR="${DATA_DIR:-}"
EVAL_TEST="${EVAL_TEST:-1}"

RUN_VALIDATION_ARG="--run_validation"
if [ "$RUN_VALIDATION" != "1" ]; then
  RUN_VALIDATION_ARG="--no_run_validation"
fi

ENABLE_THINKING_ARG="--enable_thinking"
if [ "$ENABLE_THINKING" != "1" ]; then
  ENABLE_THINKING_ARG="--no_enable_thinking"
fi

EVAL_TEST_ARG=()
if [ "$EVAL_TEST" = "1" ]; then
  EVAL_TEST_ARG=(--eval_test)
fi

DATA_DIR_ARG=()
if [ -n "$DATA_DIR" ]; then
  DATA_DIR_ARG=(--data_dir "$DATA_DIR")
fi

uv run --with-editable ./textgrad_repro \
  --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
  --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
  --with diskcache --with litellm --with graphviz --with gdown --with tenacity --with python-dotenv \
  python scripts/textgrad_repro.py \
  --model_dir "$MODEL_DIR" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN" \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --steps_per_epoch "$STEPS_PER_EPOCH" \
  --optimizer_max_tokens "$OPTIMIZER_MAX_TOKENS" \
  --task "$TASK" \
  "$RUN_VALIDATION_ARG" \
  "$ENABLE_THINKING_ARG" \
  "${DATA_DIR_ARG[@]}" \
  "${EVAL_TEST_ARG[@]}"

echo "done. results in data/textgrad_repro/ (best_prompt.json, iterations.jsonl, forward_outputs.jsonl, gradients.jsonl)"
