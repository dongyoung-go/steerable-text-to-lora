"""Arm 4/5: RayPPOTrainer subclass wiring momentum conditioning + internalization + calibration
into verl's real training loop.

README (docs/textual_momentum_grpo_README.md) section 3, per step t:
  1. Rollout conditioned on textual momentum M_{t-1} (M_0 = empty)              -> _inject_momentum
  2. Internalization: gradient log-probs computed UNCONDITIONED on M_{t-1}      -> _build_unconditioned_batch
  3. Calibration ratio w_t = pi(y|q,y_<t) / pi(y|q,M_{t-1},y_<t), min(w_t,w_max) -> tmgrpo/calibration.py
  4. Textual gradient from this step's sampled successes/failures               -> tmgrpo/trajectory.py
  5. Trajectory digest update (incremental, LLM-summarized)                     -> tmgrpo/trajectory.py
  6. Textual momentum M_t proposed for the next step                           -> tmgrpo/trajectory.py

verl's `RayPPOTrainer.fit()` has no config-level hook for any of the above (confirmed by reading
the installed verl==0.8.0 source, docs/arms_2_3_4_5_implementation_gap.md) -- steps 2-6 are woven
into a full copy of `fit()`'s body (verl/trainer/ppo/ray_trainer.py) below, with insertions marked
`# --- tmgrpo:` at each seam. Re-diff against the installed verl source if verl is upgraded.
"""

from __future__ import annotations

import copy
import os
import uuid
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    compute_spec_decode_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask

from tmgrpo.calibration import calibration_ratio
from tmgrpo.llm_client import DEFAULT_MODEL, LLMClient
from tmgrpo.trajectory import TrajectoryState
from tmgrpo.verl_hooks import inject_conditioning_context, truncate_head_tail

TRAJECTORY_STATE_FILENAME = "trajectory_state.json"


class TMGrpoTrainer(RayPPOTrainer):
    """Arm 4 (momentum, no internalization) / Arm 5 (momentum + internalization + calibration).

    Toggled entirely via `config.custom.*` (configs/overrides/arm{4,5}_*.yaml):
      custom.conditioning: "momentum" to enable step 1, anything else (or unset) leaves this
        trainer behaving like stock GRPO (equivalent to Arm 1, modulo the unused instance attrs).
      custom.internalization: bool -- steps 2 (unconditioned log-probs replace the gradient target).
      custom.calibration: bool -- step 3 (per-token w_t reweighting of advantages).
      custom.calibration_w_max, custom.frontier_model -- as in configs/base.yaml.
      custom.momentum_update_every_n_steps: int, default 1 -- steps 4-6 (textual gradient -> digest
        -> M_t) only run every Nth step; on skipped steps the previous momentum is reused unchanged.
        README section 6 flagged call cadence ("every RL step vs. every K steps") as an open cost
        question never pinned down; default of 1 preserves the original every-step behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        custom_cfg = self.config.get("custom", {})
        self.tm_conditioning = custom_cfg.get("conditioning", None)
        self.tm_internalization = bool(custom_cfg.get("internalization", False))
        self.tm_calibration = bool(custom_cfg.get("calibration", False))
        self.tm_calibration_w_max = float(custom_cfg.get("calibration_w_max", 5.0))
        self.tm_momentum_update_every_n_steps = int(custom_cfg.get("momentum_update_every_n_steps", 1))
        self.trajectory_state = TrajectoryState()
        self.llm_client = LLMClient(model=custom_cfg.get("frontier_model", DEFAULT_MODEL))

    # ------------------------------------------------------------------
    # tmgrpo-specific helpers
    # ------------------------------------------------------------------

    def _inject_momentum(self, batch: DataProto) -> np.ndarray | None:
        """Step 1: overwrite `batch.non_tensor_batch["raw_prompt"]` in place with the momentum-
        conditioned prompt, IF conditioning is enabled and there is a non-empty M_{t-1} yet (M_0 is
        empty, so step 1 of a fresh run is unconditioned, matching README section 3 step 1).

        Returns the ORIGINAL (unconditioned) raw_prompt array so the caller can reconstruct the
        unconditioned prompt later for internalization/calibration, or None if no injection happened.
        Must be called before `self._get_gen_batch(batch)`, which is the last point `raw_prompt`
        is still a live key on `batch` itself (verl/utils/dataset/rl_dataset.py's RLHFDataset
        returns raw_prompt untokenized; `_get_gen_batch` pops it off `batch` into `gen_batch`).
        """
        if self.tm_conditioning != "momentum" or not self.trajectory_state.momentum:
            return None
        original_raw_prompt = copy.deepcopy(batch.non_tensor_batch["raw_prompt"])
        momentum = self.trajectory_state.momentum
        batch.non_tensor_batch["raw_prompt"] = np.array(
            [inject_conditioning_context(list(p), momentum) for p in batch.non_tensor_batch["raw_prompt"]],
            dtype=object,
        )
        return original_raw_prompt

    def _build_unconditioned_batch(self, batch: DataProto, original_raw_prompt: np.ndarray) -> DataProto:
        """Steps 2-3: reconstruct input_ids/attention_mask/position_ids/prompts under the
        UNCONDITIONED prompt, keeping the already-sampled response tokens (and their padding)
        fixed. Mirrors verl's rollout-side padding convention exactly (confirmed by reading
        verl/experimental/agent_loop/agent_loop.py's `_agent_loop_postprocess`/`_postprocess`):
        prompt left-padded to `rollout.prompt_length`, response right-padded to
        `rollout.response_length`, input_ids = concat(prompt, response),
        attention_mask = concat(prompt_mask, response_mask-derived-attention),
        position_ids = compute_position_id_with_mask(attention_mask) (text-only; this project's
        MATH prompts have no multimodal content, so the vision-aware branch verl uses for Qwen-VL
        is not needed here).
        """
        rollout_cfg = self.config.actor_rollout_ref.rollout
        prompt_length = rollout_cfg.prompt_length
        response_length = rollout_cfg.response_length
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        responses = batch.batch["responses"]
        n = responses.shape[0]

        prompt_id_lists = [
            self.tokenizer.apply_chat_template(
                list(original_raw_prompt[i]),
                add_generation_prompt=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
            for i in range(n)
        ]

        self.tokenizer.padding_side = "left"
        padded_prompt = self.tokenizer.pad(
            {"input_ids": prompt_id_lists},
            padding="max_length",
            max_length=prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids = padded_prompt["input_ids"][:, -prompt_length:].to(responses.device)
        prompt_attention_mask = padded_prompt["attention_mask"][:, -prompt_length:].to(responses.device)

        response_attention_mask = batch.batch["attention_mask"][:, -response_length:]
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompt_ids, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        unconditioned = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompt_ids,
                "responses": responses,
                "response_mask": batch.batch["response_mask"],
            }
        )
        unconditioned.meta_info = dict(batch.meta_info)
        return unconditioned

    def _build_step_summary(self, batch: DataProto, max_samples: int = 5) -> str:
        """Step 4 input: a compact text digest of this step's sampled outcomes, built from real
        batch data (decoded responses + reward-manager correctness), not guessed/synthetic.

        Responses are truncated head+tail (not just head): a plain head cut on a ~600-token math
        solution reliably drops the final boxed answer -- usually the most diagnostic part of the
        response -- before the LLM ever sees it. Budgets are kept modest (well under gpt-5-mini's
        context window, and small enough that 5 examples stay a cheap, bounded-size call) rather
        than raised aggressively.
        """
        acc = batch.non_tensor_batch.get("acc")
        n = batch.batch["responses"].shape[0]
        idx = np.random.default_rng().choice(n, size=min(max_samples, n), replace=False)

        lines = []
        if acc is not None:
            lines.append(f"Step accuracy: {float(np.mean(acc)):.3f} over {n} sampled responses.")
        for i in idx:
            problem = self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True)
            response = self.tokenizer.decode(batch.batch["responses"][i], skip_special_tokens=True)
            correct = bool(acc[i]) if acc is not None else "unknown"
            problem = truncate_head_tail(problem, head=450)
            response = truncate_head_tail(response, head=600, tail=600)
            lines.append(f"--- problem: {problem}\nresponse: {response}\ncorrect: {correct}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # checkpoint persistence: trajectory_state must survive a resumed run,
    # or a resumed Arm 5 run would silently reset to M_0 = "" every restart.
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if self.tm_conditioning != "momentum":
            return
        import json

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        path = os.path.join(local_global_step_folder, TRAJECTORY_STATE_FILENAME)
        with open(path, "w") as f:
            json.dump(
                {
                    "digest": self.trajectory_state.digest,
                    "momentum": self.trajectory_state.momentum,
                    "history": self.trajectory_state.history,
                },
                f,
            )

    def _load_checkpoint(self):
        step = super()._load_checkpoint()
        if self.tm_conditioning != "momentum":
            return step
        import json

        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if global_step_folder is None:
            return step
        path = os.path.join(global_step_folder, TRAJECTORY_STATE_FILENAME)
        if not os.path.exists(path):
            return step
        with open(path) as f:
            state = json.load(f)
        self.trajectory_state.digest = state["digest"]
        self.trajectory_state.momentum = state["momentum"]
        self.trajectory_state.history = state["history"]
        return step

    # ------------------------------------------------------------------
    # fit() -- copied from verl/trainer/ppo/ray_trainer.py's RayPPOTrainer.fit() (verl==0.8.0),
    # with tmgrpo insertions marked `# --- tmgrpo:`. See module docstring for the seam rationale;
    # there is no smaller override point verl exposes for steps 1-6 above.
    # ------------------------------------------------------------------

    def fit(self):
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            from verl.utils.rollout_skip import RolloutSkip

            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # --- tmgrpo: step 1 -- momentum-condition every prompt on M_{t-1} before rollout.
                # Must happen before _get_gen_batch pops raw_prompt off `batch`. Keep the original
                # (unconditioned) raw_prompt around for steps 2-3 below.
                original_raw_prompt = self._inject_momentum(batch)

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # --- tmgrpo: keep the unconditioned raw_prompt aligned with the now-repeated
                    # (rollout.n-fold) batch, for steps 2-3 below.
                    if original_raw_prompt is not None:
                        original_raw_prompt = np.repeat(original_raw_prompt, rollout_n, axis=0)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # --- tmgrpo: steps 2-3 -- unconditioned log-probs (internalization target) and
                    # the calibration ratio w_t. Both need `original_raw_prompt`, i.e. only run when
                    # this step actually conditioned the rollout (M_{t-1} was non-empty).
                    if original_raw_prompt is not None and (self.tm_internalization or self.tm_calibration):
                        with marked_timer("tmgrpo_unconditioned_log_prob", timing_raw, color="blue"):
                            unconditioned_batch = self._build_unconditioned_batch(batch, original_raw_prompt)
                            unconditioned_old_log_prob, _ = self._compute_old_log_prob(unconditioned_batch)
                        logp_unconditioned = unconditioned_old_log_prob.batch["old_log_probs"]
                        logp_conditioned = batch.batch["old_log_probs"]

                        if self.tm_calibration:
                            w_t = calibration_ratio(
                                logp_unconditioned.detach().cpu().numpy(),
                                logp_conditioned.detach().cpu().numpy(),
                                w_max=self.tm_calibration_w_max,
                            )
                            w_t = torch.from_numpy(w_t).to(logp_conditioned.device, logp_conditioned.dtype)
                            valid = batch.batch["response_mask"].bool()
                            metrics.update(
                                {
                                    "tmgrpo/w_t_mean": w_t[valid].mean().item(),
                                    "tmgrpo/w_t_min": w_t[valid].min().item(),
                                    "tmgrpo/w_t_max": w_t[valid].max().item(),
                                }
                            )

                        if self.tm_internalization:
                            batch.batch["old_log_probs"] = logp_unconditioned
                            batch.batch["input_ids"] = unconditioned_batch.batch["input_ids"]
                            batch.batch["attention_mask"] = unconditioned_batch.batch["attention_mask"]
                            batch.batch["position_ids"] = unconditioned_batch.batch["position_ids"]
                            batch.batch["prompts"] = unconditioned_batch.batch["prompts"]

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # --- tmgrpo: step 3 (cont'd) -- reweight per-token advantages by w_t before
                        # the actor update. This is the substitute for apply_calibration's
                        # pg_loss_per_token reweighting (the trainer never sees per-token pg_loss --
                        # verl's actor worker computes it internally), and achieves the same effect:
                        # loss ∝ advantage x log-prob-ratio, so scaling advantages scales the loss.
                        if original_raw_prompt is not None and self.tm_calibration:
                            batch.batch["advantages"] = batch.batch["advantages"] * w_t

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup > self.global_steps:
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # --- tmgrpo: steps 4-6 -- textual gradient -> digest update -> M_t, from
                        # this step's real batch outcomes. Must run BEFORE _save_checkpoint below:
                        # the node this trains on is preemptible, and a checkpoint saved before the
                        # trajectory update would persist the STALE M_{t-1} that conditioned this
                        # step's rollout, not the freshly computed M_t a resumed run actually needs
                        # to condition the next step. Gated by momentum_update_every_n_steps (README
                        # section 6's "every RL step vs. every K steps" cost question); on skipped
                        # steps the previous momentum just stays in effect for the next rollout.
                        if (
                            self.tm_conditioning == "momentum"
                            and self.global_steps % self.tm_momentum_update_every_n_steps == 0
                        ):
                            with marked_timer("tmgrpo_trajectory_step", timing_raw, color="olive"):
                                step_summary = self._build_step_summary(batch)
                                self.trajectory_state.step(self.llm_client, step_summary)
                            metrics["tmgrpo/momentum_len_chars"] = len(self.trajectory_state.momentum)
                            metrics["tmgrpo/llm_call_count"] = self.llm_client.call_count

                        from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

        self._shutdown_dump_executor()
