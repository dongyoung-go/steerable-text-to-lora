#!/usr/bin/env bash
# Phase 1 pilot, gated arm: ICRL exactly as published (self_correct_grpo_README.md §1.1).
#
# Runs vendored, byte-for-byte-untouched `icrl.hydra_runner` against the math env, single B200
# GPU, on the prepared MATH pilot data (see scripts/prepare_math_data.py). Must run inside the
# slimerl/slime:latest container (or the build_conda.sh env) on the provisioned GPU node — see
# docs/pilot_setup.md for the full setup sequence (model download + HF->Megatron torch_dist
# conversion) this script assumes has already happened.
#
# This script does not touch /home/dg793/steerable-text-to-lora's own .venv/pyproject.toml/uv.lock
# in any way — it only shells out to the vendored ICRL/slime stack's own Python environment.
set -euo pipefail

SELF_CORRECT_GRPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ICRL_DIR="${SELF_CORRECT_GRPO_DIR}/vendor/ICRL"
REPO_PARENT_DIR="$(dirname -- "${SELF_CORRECT_GRPO_DIR}")"

# Where the pilot's Qwen3-4B-Instruct checkpoint (HF + torch_dist-converted) and the ray/wandb/tb
# run artifacts live. Override these from the environment if your GPU node uses different paths.
: "${ICRL_MODEL_DIR:=${ICRL_DIR}/models}"
: "${PILOT_TRAIN_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/train.jsonl}"
: "${PILOT_EVAL_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/eval.jsonl}"
: "${PILOT_NUM_ROLLOUT:=300}"           # cheap pilot: ~2 orders of magnitude below ICRL's own num_rollout=3000
: "${PILOT_ROLLOUT_BATCH_SIZE:=16}"
: "${PILOT_N_SAMPLES_PER_PROMPT:=8}"

export PYTHONPATH="${ICRL_DIR}:${PYTHONPATH:-}"

cd "${ICRL_DIR}"
python3 -m icrl.hydra_runner \
  --config-dir "${SELF_CORRECT_GRPO_DIR}/hydra_conf" \
  custom=icrl_math_pilot \
  gpu=train_1gpu \
  paths.model_dir="${ICRL_MODEL_DIR}" \
  checkpoint.cli.hf_checkpoint="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507" \
  checkpoint.cli.load="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507-torch-dist" \
  checkpoint.cli.ref_load="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507-torch-dist" \
  rollout.cli.prompt_data="${PILOT_TRAIN_DATA}" \
  rollout.cli.input_key=prompt \
  rollout.cli.label_key=label \
  rollout.cli.metadata_key=metadata \
  rollout.cli.apply_chat_template=false \
  rollout.cli.num_rollout="${PILOT_NUM_ROLLOUT}" \
  rollout.cli.rollout_batch_size="${PILOT_ROLLOUT_BATCH_SIZE}" \
  rollout.cli.n_samples_per_prompt="${PILOT_N_SAMPLES_PER_PROMPT}" \
  eval.cli.eval_prompt_data="[math,${PILOT_EVAL_DATA}]" \
  eval.cli.eval_input_key=prompt \
  eval.cli.eval_label_key=label \
  sglang.cli.rollout_num_gpus_per_engine=1 \
  "gpu.ray_job.runtime_env.env_vars.PYTHONPATH=/root/Megatron-LM/:${ICRL_DIR}:${REPO_PARENT_DIR}" \
  "$@"
