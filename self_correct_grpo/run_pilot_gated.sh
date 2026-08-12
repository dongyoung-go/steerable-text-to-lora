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
# Default save_interval (50, from icrl/hydra_conf/checkpoint/base.yaml) means no periodic checkpoint
# ever gets written before this pilot's rollout count wraps up -- override low so a recent
# checkpoint always exists regardless of when/whether training stops early (crash, time limit,
# manual stop). See docs/pilot_eval_design.md step 1.
: "${PILOT_SAVE_INTERVAL:=2}"
# Docker's slimerl/slime:latest image bakes Megatron-LM in at /root/Megatron-LM; the conda-fallback
# build (setup_icrl_pilot_env.sh) installs it under $BASE_DIR instead (default ~/icrl_pilot_build).
: "${MEGATRON_LM_DIR:=/root/Megatron-LM}"

# This node's system CUDA install (ldconfig-registered alongside an older, stale CUDA-12.0 whose
# libcudart.so.12 lacks cudaGetDriverEntryPointByVersion) can shadow torch's own bundled cudart in
# subprocesses SGLang spawns via multiprocessing, crashing with "undefined symbol:
# cudaGetDriverEntryPointByVersion" on `import torch`. Pin LD_LIBRARY_PATH to torch's bundled CUDA
# runtime so every worker resolves the right one regardless of spawn context.
_TORCH_LIB_DIR="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
_CUDART_LIB_DIR="$(python3 -c 'import nvidia.cuda_runtime, os; print(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib"))')"
export LD_LIBRARY_PATH="${_TORCH_LIB_DIR}:${_CUDART_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here, even though it's
# PyTorch's own suggested fix for the step-2 fragmentation OOM this pilot hits (both offload paths
# disabled below means the actor's optimizer state stays GPU-resident, and repeated train
# iterations fragment the allocator). Tried it -- it broke the actor<->rollout weight-sync IPC
# instead: expandable segments are backed by cuMemAddressReserve/cuMemMap virtual memory rather
# than plain cudaMalloc, which torch.multiprocessing's pidfd_getfd-based cross-process tensor
# rebuild (used by update_weights_from_tensor) can't share, failing with
# "RuntimeError: pidfd_getfd: Operation not permitted" on literally the first weight sync every
# time. sglang_mem_fraction_static=0.4 below is the fragmentation mitigation instead.

export PYTHONPATH="${ICRL_DIR}:${PYTHONPATH:-}"

# --colocate forces --offload-train AND --offload-rollout on by default (frees actor/rollout GPU
# mem for each other between phases via torch_memory_saver.pause()/resume()), but that native
# CUDA memory pause/resume is fragile on this node: offload-train's pause() crashed outright
# (silent worker death, no Python traceback -- see git history), and with that disabled,
# offload-rollout's resume() instead hung indefinitely after a transient CUDA_ERROR_OUT_OF_MEMORY
# (its own "may not be an issue, will retry" claim didn't hold -- 5+ min at 0% GPU util, no
# recovery). The pilot's single B200 has ample headroom for actor + rollout engine to both stay
# fully resident throughout (~89GB + ~50GB well under 183GB), so disable both offload paths
# entirely and avoid the pause/resume dance altogether -- render_cli_args can't emit an explicit
# `false` (it only emits a flag when true), but BooleanOptionalAction auto-registers `--no-X` as
# each flag's negation, so naming the hydra keys `no_offload_train`/`no_offload_rollout` renders
# exactly those flags without touching vendored icrl/utils.py's render_cli_args.

# Checkpoint saves ("Storing distributed optimizer sharded state...") OOM-killed the actor process
# twice on this node -- not the GPU-fragmentation issue described above, but the SLURM allocation's
# host-RAM cgroup ceiling (62.5GB; check with `cat /sys/fs/cgroup/system.slice/slurmstepd.scope/
# job_<id>/memory.max`, node itself has ~2TB). Megatron's torch_dist checkpoint save stages the
# full state -- model weights plus Adam's fp32 master copy and two moment buffers, ~4x the 4B-param
# model, 60GB+ -- through host CPU RAM before writing to disk, and that alone blows the 62.5GB
# ceiling even though steady-state training (everything GPU-resident, offload disabled) stays well
# under it. The cgroup OOM killer sends SIGKILL directly (no Python traceback, no core dump --
# `ulimit -c` is 0 here), which is why this looked like a silent crash. Since this pilot only needs
# frozen weights for eval-only inference afterward and never resumes training from these
# checkpoints, skip saving optimizer state entirely -- cuts the save-time RAM spike by the ~48GB
# the fp32 master + 2 moment buffers would otherwise cost. See run_pilot_eval_only.sh's matching
# `no_load_optim` override.
cd "${ICRL_DIR}"
python3 -m icrl.hydra_runner \
  --config-dir "${SELF_CORRECT_GRPO_DIR}/hydra_conf" \
  custom=icrl_math_pilot \
  gpu=train_1gpu \
  paths.model_dir="${ICRL_MODEL_DIR}" \
  checkpoint.cli.hf_checkpoint="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507" \
  checkpoint.cli.load="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507-torch-dist" \
  checkpoint.cli.ref_load="${ICRL_MODEL_DIR}/qwen3-4b-inst-2507-torch-dist" \
  checkpoint.cli.save_interval="${PILOT_SAVE_INTERVAL}" \
  +checkpoint.cli.no_save_optim=true \
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
