#!/bin/bash
# Arm 5 (Trajectory, ON/ON -- Ours): momentum-conditioned rollouts + internalization + calibration.
# README section 4: success criteria are (5) > (4) and (5) > (3).
# Run this directly in a shell that already has the GPU allocated (e.g. inside an salloc session).
#
# Requires OPENAI_API_KEY in the environment (or textual_momentum_grpo/.env, see .env.example) --
# every step calls the frontier model (gpt-5-mini) for the textual gradient / digest / momentum
# update (tmgrpo/trajectory.py). Fails loudly at the first LLM call if unset, not silently.
#
# trainer.test_freq=-1: see run_arm1_floor.sh -- periodic MATH500 validation OOM'd Arm 1's job.
# Evaluate saved checkpoints separately offline instead of relying on mid-run eval, so Arm 5's
# comparison against Arm 1 stays apples-to-apples.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "warning: OPENAI_API_KEY is not set and no .env file found -- tmgrpo.llm_client will fail on the first frontier-model call." >&2
fi

export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# See run_arm1_floor.sh: trainer.logger=[console] alone only reaches Ray's driver-stderr log
# forwarding, invisible in the SLURM-captured .out under sbatch. `file` backend writes unbuffered
# per-step JSONL straight to disk instead. Overwritten (not appended) each run.
mkdir -p logs/metrics
export VERL_FILE_LOGGER_PATH="$(pwd)/logs/metrics/arm5_trajectory_on.jsonl"

# Training data: defaults to open-r1/OpenR1-Math-220k (scripts/prepare_openr1_train.py) --
# Critique-GRPO (our baseline) actually trains on subsets of this, not Hendrycks MATH; MATH left
# Qwen3-8B saturated from step 1 (see configs/base.yaml's data.train_files comment for the full
# rationale). Set TMGRPO_TRAIN_DATA=math to use the legacy MATH pool instead.
TMGRPO_TRAIN_DATA="${TMGRPO_TRAIN_DATA:-openr1}"
case "$TMGRPO_TRAIN_DATA" in
  openr1) TRAIN_FILE=data/train_openr1.parquet ;;
  math) TRAIN_FILE=data/train.parquet ;;
  *)
    echo "error: TMGRPO_TRAIN_DATA must be 'openr1' or 'math', got '${TMGRPO_TRAIN_DATA}'" >&2
    exit 1
    ;;
esac
if [ ! -f "$TRAIN_FILE" ]; then
  echo "error: $TRAIN_FILE not found -- run scripts/prepare_{openr1,math}_train.py then" \
    "scripts/convert_to_verl_parquet.py first (see docs/build_and_run_guide.md)" >&2
  exit 1
fi

.venv-verl/bin/python -m tmgrpo.main_tmgrpo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  data.train_files=${TRAIN_FILE} data.val_files=data/eval/math500.parquet \
  data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=2048 \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path=models/qwen3-8b \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.betas=[0.9,0.98] \
  actor_rollout_ref.actor.optim.weight_decay=0.1 actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.response_length=2048 actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.total_epochs=30 trainer.total_training_steps=300 trainer.val_before_train=false \
  trainer.test_freq=-1 trainer.save_freq=100 trainer.max_actor_ckpt_to_keep=3 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.logger=[console,file] \
  trainer.project_name=tmgrpo trainer.experiment_name=arm5_trajectory_on \
  custom_reward_function.path=tmgrpo/verl_hooks.py custom_reward_function.name=compute_score \
  reward.custom_reward_function.path=tmgrpo/verl_hooks.py reward.custom_reward_function.name=compute_score \
  +custom.conditioning=momentum +custom.internalization=true +custom.calibration=true \
  +custom.calibration_w_max=5.0 +custom.frontier_model=gpt-5-mini
