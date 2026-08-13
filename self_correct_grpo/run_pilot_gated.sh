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
# Auto-detect rather than requiring the caller to export this every session: use the Docker path
# if it exists, else fall back to the conda-fallback build's default location.
if [[ -z "${MEGATRON_LM_DIR:-}" ]]; then
  if [[ -d /root/Megatron-LM ]]; then
    MEGATRON_LM_DIR=/root/Megatron-LM
  else
    MEGATRON_LM_DIR="${HOME}/icrl_pilot_build/Megatron-LM"
  fi
fi

# Validated (2026-08-12, 12 stable iterations, no crashes): offload both actor and rollout, and
# restore ICRL's own published sglang_mem_fraction_static=0.7 (this pilot had been overriding it
# down to 0.3 defensively before offload was made to work). Override PILOT_OFFLOAD=false to fall
# back to the no-offload/mem_fraction=0.3 config from earlier pilot sessions if this ever
# regresses on a different node.
: "${PILOT_OFFLOAD:=true}"
: "${PILOT_SGLANG_MEM_FRACTION_STATIC:=0.7}"
if [[ "${PILOT_OFFLOAD}" == "true" ]]; then
  PILOT_NO_OFFLOAD_TRAIN=false
  PILOT_NO_OFFLOAD_ROLLOUT=false
else
  PILOT_NO_OFFLOAD_TRAIN=true
  PILOT_NO_OFFLOAD_ROLLOUT=true
fi

# This node's system CUDA install (ldconfig-registered alongside an older, stale CUDA-12.0 whose
# libcudart.so.12 lacks cudaGetDriverEntryPointByVersion) can shadow torch's own bundled cudart in
# subprocesses SGLang spawns via multiprocessing, crashing with "undefined symbol:
# cudaGetDriverEntryPointByVersion" on `import torch`. Pin LD_LIBRARY_PATH to torch's bundled CUDA
# runtime so every worker resolves the right one regardless of spawn context.
_TORCH_LIB_DIR="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
_CUDART_LIB_DIR="$(python3 -c 'import nvidia.cuda_runtime, os; print(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib"))')"
export LD_LIBRARY_PATH="${_TORCH_LIB_DIR}:${_CUDART_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here, even though it's
# PyTorch's own suggested fix for GPU-fragmentation OOMs. Tried it -- it broke the actor<->rollout
# weight-sync IPC instead: expandable segments are backed by cuMemAddressReserve/cuMemMap virtual
# memory rather than plain cudaMalloc, which torch.multiprocessing's pidfd_getfd-based cross-process
# tensor rebuild (used by update_weights_from_tensor) can't share, failing with
# "RuntimeError: pidfd_getfd: Operation not permitted" on literally the first weight sync every
# time.

export PYTHONPATH="${ICRL_DIR}:${PYTHONPATH:-}"

# Offload (actor + rollout both pause/resume via torch_memory_saver, per --colocate's normal
# behavior) plus sglang_mem_fraction_static=0.7 is PILOT_OFFLOAD's default above -- validated
# stable for 12 iterations on 2026-08-12 after earlier sessions had disabled offload entirely due
# to pause()/resume() crashes/hangs on this node with a lower mem_fraction. If offload ever
# regresses again on a different node, rerun with PILOT_OFFLOAD=false to fall back to fully
# GPU-resident actor+rollout (needs ample headroom -- ~89GB + ~50GB fit under this pilot's 183GB
# B200, but leaves no offload safety margin for OOMs elsewhere).
#
# recompute_granularity=full / max_tokens_per_gpu=16384 (set in hydra_conf/gpu/train_1gpu.yaml,
# not overridden here) are load-bearing under the offload+0.7 memory profile above, not overly
# conservative defaults -- both recompute_granularity=selective and max_tokens_per_gpu=32768 were
# smoke-tested on 2026-08-12 and both OOM'd at the 2nd post-resume training step. Do not loosen
# either without re-validating against a resumed run, not just a fresh one (the OOM only shows up
# after a resume's transient allocator overhead).
#
# render_cli_args can't emit an explicit `false` (it only emits a flag when true), but
# BooleanOptionalAction auto-registers `--no-X` as each flag's negation, so naming the hydra keys
# `no_offload_train`/`no_offload_rollout` renders exactly those flags without touching vendored
# icrl/utils.py's render_cli_args.

# A crashed/errored previous ray job can leave a stale ray head bound to port 6379, which blocks
# a fresh `ray start --head` (icrl.hydra_runner does this itself on launch). Force-stop any
# leftover cluster before every launch so this never needs to be diagnosed manually.
ray stop --force >/dev/null 2>&1 || true

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
  +gpu.resources_cli.no_offload_train="${PILOT_NO_OFFLOAD_TRAIN}" \
  +gpu.resources_cli.no_offload_rollout="${PILOT_NO_OFFLOAD_ROLLOUT}" \
  sglang.cli.sglang_mem_fraction_static="${PILOT_SGLANG_MEM_FRACTION_STATIC}" \
  "gpu.ray_job.runtime_env.env_vars.PYTHONPATH=${MEGATRON_LM_DIR}/:${ICRL_DIR}:${REPO_PARENT_DIR}" \
  "+gpu.ray_job.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  "$@"
