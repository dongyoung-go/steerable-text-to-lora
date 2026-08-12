#!/usr/bin/env bash
# Phase 1 pilot, eval-only harness (self_correct_grpo/docs/pilot_eval_design.md §4-§5, step 3).
#
# Loads one frozen, gated-trained checkpoint (produced by run_pilot_gated.sh) and runs a single
# eval-only pass against held-out MATH500 (data/math_pilot/eval.jsonl), under a chosen
# self-refinement strategy at inference time -- gated (icrl.generate.generate, oracle skips the
# critic round when round 1 is already correct) or ungated (icrl_ungated/generate.py, critic round
# always fires). No training happens: this relies on vendored `vendor/ICRL/train.py`'s built-in
# eval-only special case (`--num-rollout 0` with `--eval-interval` set skips the training loop
# entirely, but still loads the checkpoint, syncs weights to the rollout engine, and runs exactly
# one eval pass -- see train.py's "special case for eval-only").
#
# Usage:
#   PILOT_CKPT_LOAD=<path/to/trained/checkpoint> ./run_pilot_eval_only.sh gated
#   PILOT_CKPT_LOAD=<path/to/trained/checkpoint> ./run_pilot_eval_only.sh ungated
#
# <path/to/trained/checkpoint> is a Megatron torch-dist checkpoint directory written under
# <gated training exp_dir>/checkpoints by the training run this script evaluates (an
# iter_XXXXXXX-style subdirectory tree with a top-level latest_checkpointed_iteration.txt) -- NOT
# the base HF-converted checkpoint used to seed training itself.
#
# run_pilot_gated.sh saves with `no_save_optim=true` (checkpoint-save host-RAM OOM under this
# node's SLURM cgroup memory ceiling -- see that script's comment), so these checkpoints have no
# optimizer state to load; `no_load_optim`/`no_load_rng` below match that (harmless for eval-only
# anyway, since no training/resume happens here).
#
# Must run inside the same environment as run_pilot_gated.sh (icrl-pilot conda env or
# slimerl/slime:latest container) -- see docs/pilot_setup.md.
set -euo pipefail

ARM="${1:?usage: run_pilot_eval_only.sh <gated|ungated>}"
case "${ARM}" in
  gated|ungated) ;;
  *) echo "error: arm must be 'gated' or 'ungated', got '${ARM}'" >&2; exit 1 ;;
esac

SELF_CORRECT_GRPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ICRL_DIR="${SELF_CORRECT_GRPO_DIR}/vendor/ICRL"
REPO_PARENT_DIR="$(dirname -- "${SELF_CORRECT_GRPO_DIR}")"

: "${ICRL_MODEL_DIR:=${ICRL_DIR}/models}"
: "${PILOT_CKPT_LOAD:?set PILOT_CKPT_LOAD to the trained checkpoint directory to evaluate}"
# rollout_manager's Dataset is constructed (and its prompt_data file read) unconditionally in
# create_rollout_manager, even though num_rollout=0 below means it's never actually sampled from
# -- point it at the same train pool the checkpoint was trained on rather than leaving it at
# hydra_runner.py's `_configure_math_datasets` default (data/criticgrpo/train.parquet, which was
# never prepared for this pilot and doesn't exist).
: "${PILOT_TRAIN_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/train.jsonl}"
: "${PILOT_EVAL_DATA:=${SELF_CORRECT_GRPO_DIR}/data/math_pilot/eval.jsonl}"
: "${MEGATRON_LM_DIR:=/root/Megatron-LM}"

# See run_pilot_gated.sh for why this is needed: system CUDA install conflicts can shadow torch's
# bundled cudart in SGLang-spawned subprocesses, crashing on `import torch`.
_TORCH_LIB_DIR="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
_CUDART_LIB_DIR="$(python3 -c 'import nvidia.cuda_runtime, os; print(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib"))')"
export LD_LIBRARY_PATH="${_TORCH_LIB_DIR}:${_CUDART_LIB_DIR}:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${ICRL_DIR}:${REPO_PARENT_DIR}:${PYTHONPATH:-}"

if [[ "${ARM}" == "gated" ]]; then
  RUNNER_MODULE="icrl.hydra_runner"
else
  RUNNER_MODULE="self_correct_grpo.icrl_ungated.hydra_runner"
fi

cd "${ICRL_DIR}"
python3 -m "${RUNNER_MODULE}" \
  --config-dir "${SELF_CORRECT_GRPO_DIR}/hydra_conf" \
  custom=icrl_math_pilot \
  custom.custom_config.ungated="$([[ "${ARM}" == "ungated" ]] && echo true || echo false)" \
  gpu=train_1gpu \
  paths.model_dir="${ICRL_MODEL_DIR}" \
  checkpoint.cli.hf_checkpoint="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507" \
  checkpoint.cli.load="${PILOT_CKPT_LOAD}" \
  checkpoint.cli.ref_load="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507-torch-dist" \
  +checkpoint.cli.no_load_optim=true \
  +checkpoint.cli.no_load_rng=true \
  rollout.cli.prompt_data="${PILOT_TRAIN_DATA}" \
  rollout.cli.num_rollout=0 \
  rollout.cli.input_key=prompt \
  rollout.cli.metadata_key=metadata \
  +rollout.cli.apply_chat_template=false \
  eval.cli.eval_interval=1 \
  eval.cli.eval_prompt_data="[math,${PILOT_EVAL_DATA}]" \
  sglang.cli.rollout_num_gpus_per_engine=1 \
  logging.cli.use_wandb=false \
  +gpu.resources_cli.no_offload_train=true \
  +gpu.resources_cli.no_offload_rollout=true \
  sglang.cli.sglang_mem_fraction_static=0.3 \
  "gpu.ray_job.runtime_env.env_vars.PYTHONPATH=${MEGATRON_LM_DIR}/:${ICRL_DIR}:${REPO_PARENT_DIR}" \
  "+gpu.ray_job.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  "${@:2}"
