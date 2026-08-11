"""Ungated-variant entry point — see `generate.py` in this package for the actual behavioral diff.

This is a byte-for-byte copy of vendored `icrl/hydra_runner.py` with one line changed (marked
`# UNGATED:` below): the `--custom-generate-function-path` handed to slime's `train.py` points at
this package's `generate.generate` instead of upstream's `icrl.generate.generate`. Everything
else — CLI arg assembly, ray job submission, reward/logging function paths — is unchanged, so the
gated and ungated arms run through identical training code and differ only in which rollout
function slime calls per step.

Requires both `vendor/ICRL` and `self_correct_grpo` on `PYTHONPATH` (so `icrl.*` and
`self_correct_grpo.icrl_ungated.*` both resolve). Run as
`python -m self_correct_grpo.icrl_ungated.hydra_runner custom=icrl_math_pilot ...` from a
directory where hydra's `config_path="hydra_conf"` resolution still finds the vendored
`icrl/hydra_conf` tree (see `../run_pilot_ungated.sh`).
"""

import json
import os

import hydra
from icrl.utils import dump_resolved_custom_config, load_model_args, render_cli_args, run_command
from omegaconf import DictConfig, OmegaConf, open_dict


def _configure_math_datasets(cfg: DictConfig) -> None:
    if cfg.custom.env_name != "math":
        return

    train_default = f"{cfg.paths.data_dir}/agentgym/train/{cfg.custom.env_name}_train.jsonl"
    test_default = f"{cfg.paths.data_dir}/agentgym/test/{cfg.custom.env_name}_test.jsonl"
    math_train = f"{cfg.paths.data_dir}/criticgrpo/train.parquet"
    math_test = f"{cfg.paths.data_dir}/criticgrpo/test.parquet"

    with open_dict(cfg):
        if cfg.rollout.cli.get("prompt_data") in (None, train_default):
            cfg.rollout.cli.prompt_data = math_train
        if cfg.rollout.cli.get("input_key") == "task_id":
            cfg.rollout.cli.input_key = "prompt"
        if cfg.rollout.cli.get("label_key") is None:
            cfg.rollout.cli.label_key = "label"
        if cfg.rollout.cli.get("metadata_key") is None:
            cfg.rollout.cli.metadata_key = "metadata"

        eval_prompt_data = list(cfg.eval.cli.get("eval_prompt_data") or [])
        if not eval_prompt_data or eval_prompt_data == [cfg.custom.env_name, test_default]:
            cfg.eval.cli.eval_prompt_data = [cfg.custom.env_name, math_test]
        if cfg.eval.cli.get("eval_input_key") is None:
            cfg.eval.cli.eval_input_key = "prompt"
        if cfg.eval.cli.get("eval_label_key") is None:
            cfg.eval.cli.eval_label_key = "label"


@hydra.main(version_base="1.3", config_path="../vendor/ICRL/icrl/hydra_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _configure_math_datasets(cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, cfg.paths.exp_dir + "/config_resoved.yaml")

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.runtime.env_vars.items()})

    ray_start_cmd = ["ray", "start"] + render_cli_args(cfg.gpu.ray_start)
    run_command(ray_start_cmd, env=env)

    custom_config_path = dump_resolved_custom_config(cfg)
    model_args = load_model_args(cfg, env=env)

    train_args: list[str] = []
    train_args.extend(render_cli_args(cfg.gpu.resources_cli))
    train_args.extend(model_args)
    train_args.extend(render_cli_args(cfg.checkpoint.cli))
    train_args.extend(render_cli_args(cfg.rollout.cli))
    train_args.extend(render_cli_args(cfg.optimizer.cli))
    train_args.extend(render_cli_args(cfg.algo.cli))
    train_args.extend(render_cli_args(cfg.logging.cli))
    train_args.extend(render_cli_args(cfg.gpu.perf_cli))
    train_args.extend(render_cli_args(cfg.eval.cli))
    train_args.extend(render_cli_args(cfg.sglang.cli))
    train_args.extend(render_cli_args(cfg.misc.cli))
    train_args.extend(["--custom-config-path", custom_config_path])
    # UNGATED: upstream always points here at `icrl.generate.generate` (the oracle-gated
    # rollout). This is the one behavioral change in this file — route to the ungated rollout
    # function in `self_correct_grpo/icrl_ungated/generate.py` instead.
    train_args.extend(["--custom-generate-function-path", "self_correct_grpo.icrl_ungated.generate.generate"])
    train_args.extend(["--custom-rollout-log-function-path", "icrl.logging_utils.log_rollout_data"])
    train_args.extend(["--custom-eval-rollout-log-function-path", "icrl.logging_utils.log_eval_rollout_data"])
    if cfg.custom.reward_process is not None:
        train_args.extend(["--custom-reward-post-process-path", cfg.custom.reward_process])

    submit_cmd = [
        "ray",
        "job",
        "submit",
        f"--address={cfg.gpu.ray_job.address}",
        f"--runtime-env-json={json.dumps(OmegaConf.to_container(cfg.gpu.ray_job.runtime_env, resolve=True), separators=(',', ':'))}",
        "--",
        str(cfg.runtime.python_executable),
        str(cfg.runtime.train_entrypoint),
        *train_args,
    ]
    run_command(submit_cmd, env=env)


if __name__ == "__main__":
    main()
