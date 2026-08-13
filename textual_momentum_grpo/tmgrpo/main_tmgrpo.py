"""Arm 4/5 entrypoint: same Hydra/Ray plumbing as verl.trainer.main_ppo, but constructs
tmgrpo.verl_trainer.TMGrpoTrainer instead of the stock RayPPOTrainer.

`verl.trainer.main_ppo.TaskRunner.run()` builds the trainer in one monolithic Ray-actor method
with no smaller override seam (confirmed via docs/arms_2_3_4_5_implementation_gap.md's reading of
the installed verl source), so `TMGrpoTaskRunner.run()` below is a full copy of that method with
only the trainer-construction lines changed. Re-diff against verl/trainer/main_ppo.py if verl is
upgraded. `run_ppo`'s `task_runner_class` parameter is verl's own supported extension point for
this (verl/trainer/main_ppo.py:52), used here instead of monkeypatching.
"""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer import main_ppo as verl_main_ppo
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from tmgrpo.verl_trainer import TMGrpoTrainer

# run_arm1_floor.sh (and this script's arm5 counterpart) invoke `-m verl.trainer.main_ppo`-style:
# no repo-local --config-path override, just CLI overrides on top of verl's own packaged default
# schema (verl/trainer/config/ppo_trainer.yaml). Point Hydra at that same schema by absolute path,
# since `config_path` in @hydra.main is resolved relative to *this* file's directory, not verl's.
_VERL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_main_ppo.__file__)), "config")


class TMGrpoTaskRunner(TaskRunner):
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"TMGrpoTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # --- tmgrpo: the one line that differs from verl.trainer.main_ppo.TaskRunner.run().
        trainer = TMGrpoTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TMGrpoTaskRunner))


if __name__ == "__main__":
    main()
