#!/bin/bash
# Thin env-var wrapper around scripts/gepa_repro.py, same style as
# scripts/textgrad_repro_run.sh. See gepa_repro_README.md.
#
# Runs via `uv run --with-editable ./textgrad_repro --with-editable ./gepa_repro --with ...`
# -- an ephemeral overlay layered on top of this repo's own resolution,
# never written into pyproject.toml/uv.lock/the persistent .venv. Needs
# BOTH editable installs: gepa_repro.py imports textgrad_repro's TASKS
# registry directly (dataset splits/parsers/task descriptions, same pattern
# scripts/textgrad_baseline_sweep.py already uses), and drives the real
# `gepa` package's optimize_anything()/GEPAConfig/EngineConfig/
# ReflectionConfig. First run resolves/downloads the overlay (~1 min);
# cached after that.
#
# Same vllm==0.11.0/transformers==4.57.1/kernels==0.10.0 pins as
# textgrad_repro_run.sh, for the same reason (this machine's CUDA 12.8
# driver caps out below every vllm release young enough to want
# transformers>=5 -- see textgrad_repro_README.md's "why this is pinned
# much harder" section). litellm/cloudpickle/tqdm added for gepa's own
# `full` extra (gepa's pyproject.toml has zero hard dependencies -- see
# gepa_repro_README.md).
#
# Example smoke tests:
#   TASK=gsm8k MAX_METRIC_CALLS=60 EVAL_TEST=0 ./scripts/gepa_repro_run.sh
#   TASK=aime MAX_METRIC_CALLS=20 ./scripts/gepa_repro_run.sh
#
# TASK selects among textgrad_repro.py's TASKS registry (default: gsm8k) --
# same registry scripts/textgrad_repro_run_all.sh iterates. See
# TEXTGRAD_MULTITASK_PLAN.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TASK="${TASK:-gsm8k}"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3-14B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-3936}"
# Default OFF here (unlike the original text-to-lora gepa_repro_aime.py,
# which defaulted this on) -- see gepa_repro.py's module docstring.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
SEED="${SEED:-42}"
EVAL_TEST="${EVAL_TEST:-1}"
DATA_DIR="${DATA_DIR:-}"
# Opt-in early-stop diagnostic, unset by default -- unlike the paper's own
# protocol, which always runs to MAX_METRIC_CALLS. See gepa_repro.py's
# --no_improvement_patience help text.
NO_IMPROVEMENT_PATIENCE="${NO_IMPROVEMENT_PATIENCE:-}"

THINKING_ARG="--enable_thinking"
if [ "$ENABLE_THINKING" != "1" ]; then
  THINKING_ARG="--no_enable_thinking"
fi

EVAL_TEST_ARG=()
if [ "$EVAL_TEST" = "1" ]; then
  EVAL_TEST_ARG=(--eval_test)
fi

DATA_DIR_ARG=()
if [ -n "$DATA_DIR" ]; then
  DATA_DIR_ARG=(--data_dir "$DATA_DIR")
fi

NO_IMPROVEMENT_PATIENCE_ARG=()
if [ -n "$NO_IMPROVEMENT_PATIENCE" ]; then
  NO_IMPROVEMENT_PATIENCE_ARG=(--no_improvement_patience "$NO_IMPROVEMENT_PATIENCE")
fi

uv run --with-editable ./textgrad_repro --with-editable ./gepa_repro \
  --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
  --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
  --with diskcache --with litellm --with cloudpickle --with tqdm \
  --with graphviz --with gdown --with tenacity --with python-dotenv \
  python scripts/gepa_repro.py \
  --model_dir "$MODEL_DIR" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN" \
  --batch_size "$BATCH_SIZE" \
  --max_metric_calls "$MAX_METRIC_CALLS" \
  --task "$TASK" \
  --seed "$SEED" \
  "$THINKING_ARG" \
  "${DATA_DIR_ARG[@]}" \
  "${EVAL_TEST_ARG[@]}" \
  "${NO_IMPROVEMENT_PATIENCE_ARG[@]}"

echo "done. results in data/gepa_repro/*_${TASK}_gepa-repro/"
