#!/usr/bin/env bash
# Phase 1 pilot, ungated arm: ICRL's own reward formula with the oracle gate simply removed
# (self_correct_grpo_README.md §1.1) — the decisive comparison this pilot exists to produce.
#
# Identical to run_pilot_gated.sh except it invokes self_correct_grpo/icrl_ungated/hydra_runner.py
# instead of vendored icrl.hydra_runner — see icrl_ungated/generate.py for the single-line gating
# diff, and tests/test_icrl_ungated_diff.py / test_icrl_ungated_hydra_runner_diff.py for the
# automated check that nothing else differs from the gated arm's code path.
#
# Must run inside the slimerl/slime:latest container (or build_conda.sh env) on the provisioned
# GPU node — see docs/pilot_setup.md. Does not touch this repo's own .venv/pyproject.toml/uv.lock.
set -euo pipefail

SELF_CORRECT_GRPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ICRL_DIR="${SELF_CORRECT_GRPO_DIR}/vendor/ICRL"
REPO_PARENT_DIR="$(dirname -- "${SELF_CORRECT_GRPO_DIR}")"

: "${ICRL_MODEL_DIR:=${ICRL_DIR}/models}"
: "${PILOT_TRAIN_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/train.jsonl}"
: "${PILOT_EVAL_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/eval.jsonl}"
: "${PILOT_NUM_ROLLOUT:=300}"
: "${PILOT_ROLLOUT_BATCH_SIZE:=16}"
: "${PILOT_N_SAMPLES_PER_PROMPT:=8}"

export PYTHONPATH="${ICRL_DIR}:${REPO_PARENT_DIR}:${PYTHONPATH:-}"

cd "${ICRL_DIR}"
python3 -m self_correct_grpo.icrl_ungated.hydra_runner \
  --config-dir "${SELF_CORRECT_GRPO_DIR}/hydra_conf" \
  custom=icrl_math_pilot \
  custom.custom_config.ungated=true \
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
