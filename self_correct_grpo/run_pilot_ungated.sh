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
: "${MEGATRON_LM_DIR:=/root/Megatron-LM}"

# See run_pilot_gated.sh for why this is needed: system CUDA install conflicts can shadow torch's
# bundled cudart in SGLang-spawned subprocesses, crashing on `import torch`.
_TORCH_LIB_DIR="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
_CUDART_LIB_DIR="$(python3 -c 'import nvidia.cuda_runtime, os; print(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib"))')"
export LD_LIBRARY_PATH="${_TORCH_LIB_DIR}:${_CUDART_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# See run_pilot_gated.sh for why PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is deliberately
# NOT set here -- it breaks the actor<->rollout weight-sync IPC (pidfd_getfd failure) despite being
# PyTorch's own suggested fragmentation-OOM fix. sglang_mem_fraction_static=0.4 is used instead.

export PYTHONPATH="${ICRL_DIR}:${REPO_PARENT_DIR}:${PYTHONPATH:-}"

# See run_pilot_gated.sh for why this is needed: --colocate's default --offload-train/-rollout
# crash/hang natively (torch_memory_saver pause()/resume()) on this node; both disabled via the
# --no-offload-train/--no-offload-rollout negation flags, rendered by naming the hydra keys
# no_offload_train/no_offload_rollout (render_cli_args can't emit `false`).

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
  rollout.cli.metadata_key=metadata \
  +rollout.cli.apply_chat_template=false \
  rollout.cli.num_rollout="${PILOT_NUM_ROLLOUT}" \
  rollout.cli.rollout_batch_size="${PILOT_ROLLOUT_BATCH_SIZE}" \
  rollout.cli.n_samples_per_prompt="${PILOT_N_SAMPLES_PER_PROMPT}" \
  eval.cli.eval_prompt_data="[math,${PILOT_EVAL_DATA}]" \
  sglang.cli.rollout_num_gpus_per_engine=1 \
  logging.cli.use_wandb=false \
  +gpu.resources_cli.no_offload_train=true \
  +gpu.resources_cli.no_offload_rollout=true \
  sglang.cli.sglang_mem_fraction_static=0.3 \
  "gpu.ray_job.runtime_env.env_vars.PYTHONPATH=${MEGATRON_LM_DIR}/:${ICRL_DIR}:${REPO_PARENT_DIR}" \
  "+gpu.ray_job.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  "$@"
