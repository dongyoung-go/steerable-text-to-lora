#!/bin/bash
# Arm 1 (Floor): stock GRPO, no critique/momentum/internalization.
# Run this directly in a shell that already has the GPU allocated (e.g. inside an salloc session).
#
# trainer.test_freq=-1: periodic MATH500 validation OOM'd the job at step 10, even at 250GB
# RAM (the training loop itself is stable -- validation's much larger batch is what pushed it
# over). Evaluate the saved checkpoint separately offline instead of relying on mid-run eval.
set -euo pipefail
cd "$(dirname "$0")"

export VLLM_ATTENTION_BACKEND=TRITON_ATTN

.venv-verl/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  data.train_files=data/train.parquet data.val_files=data/eval/math500.parquet \
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
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.logger=[console] \
  trainer.project_name=tmgrpo trainer.experiment_name=arm1_floor \
  custom_reward_function.path=tmgrpo/verl_hooks.py custom_reward_function.name=compute_score \
  reward.custom_reward_function.path=tmgrpo/verl_hooks.py reward.custom_reward_function.name=compute_score
