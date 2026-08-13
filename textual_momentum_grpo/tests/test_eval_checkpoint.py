import json
from pathlib import Path

import pytest

from scripts.eval_checkpoint import (
    build_chat_prompt,
    build_report,
    default_merged_dir_name,
    find_fsdp_actor_dir,
    is_merged_hf_dir,
    load_eval_set,
    resolve_hf_model_dir,
    score_eval_set,
)


def test_load_eval_set_parses_prompt_label_rows(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "2+2?"}], "label": "4"},
        {"prompt": [{"role": "user", "content": "3+3?"}], "label": "6"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    loaded = load_eval_set(path)

    assert loaded == rows


def test_load_eval_set_skips_blank_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    row = {"prompt": [{"role": "user", "content": "x"}], "label": "1"}
    path.write_text(f"\n{json.dumps(row)}\n\n")

    assert load_eval_set(path) == [row]


def test_is_merged_hf_dir_true_with_config_and_safetensors(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")

    assert is_merged_hf_dir(tmp_path) is True


def test_is_merged_hf_dir_false_without_config(tmp_path):
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")

    assert is_merged_hf_dir(tmp_path) is False


def test_is_merged_hf_dir_false_without_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    assert is_merged_hf_dir(tmp_path) is False


def test_find_fsdp_actor_dir_when_path_is_the_actor_dir(tmp_path):
    (tmp_path / "model_world_size_1_rank_0.pt").write_bytes(b"")

    assert find_fsdp_actor_dir(tmp_path) == tmp_path


def test_find_fsdp_actor_dir_when_path_is_the_global_step_parent(tmp_path):
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(b"")

    assert find_fsdp_actor_dir(tmp_path) == actor_dir


def test_find_fsdp_actor_dir_none_when_no_shards_present(tmp_path):
    assert find_fsdp_actor_dir(tmp_path) is None


def test_default_merged_dir_name_extracts_global_step():
    checkpoint = Path("checkpoints/tmgrpo/arm1_floor/global_step_300/actor")
    assert default_merged_dir_name(checkpoint, "arm1_floor") == "arm1_floor_global_step_300"


def test_default_merged_dir_name_falls_back_when_no_step_in_path(tmp_path):
    assert default_merged_dir_name(tmp_path / "some_dir", "arm1_floor") == "arm1_floor_unknown_step"


def test_resolve_hf_model_dir_returns_checkpoint_directly_if_already_merged(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")

    result = resolve_hf_model_dir(tmp_path, "arm1_floor", hf_cache_dir=tmp_path / "cache")

    assert result == tmp_path


def test_resolve_hf_model_dir_merges_fsdp_checkpoint(tmp_path):
    checkpoint = tmp_path / "global_step_100" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_world_size_1_rank_0.pt").write_bytes(b"")
    hf_cache_dir = tmp_path / "cache"

    calls = []

    def fake_merge(actor_dir, target_dir):
        calls.append((actor_dir, target_dir))
        (target_dir / "config.json").write_text("{}")
        (target_dir / "model.safetensors").write_bytes(b"")

    result = resolve_hf_model_dir(
        checkpoint, "arm1_floor", hf_cache_dir=hf_cache_dir, merge_fn=fake_merge
    )

    assert result == hf_cache_dir / "arm1_floor_global_step_100"
    assert calls == [(checkpoint, hf_cache_dir / "arm1_floor_global_step_100")]


def test_resolve_hf_model_dir_reuses_cached_merge_without_calling_merge_fn(tmp_path):
    checkpoint = tmp_path / "global_step_100" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_world_size_1_rank_0.pt").write_bytes(b"")
    hf_cache_dir = tmp_path / "cache"
    cached = hf_cache_dir / "arm1_floor_global_step_100"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    (cached / "model.safetensors").write_bytes(b"")

    def fail_merge(actor_dir, target_dir):
        raise AssertionError("should not be called when a valid cached merge exists")

    result = resolve_hf_model_dir(
        checkpoint, "arm1_floor", hf_cache_dir=hf_cache_dir, merge_fn=fail_merge
    )

    assert result == cached


def test_resolve_hf_model_dir_force_remerge_ignores_cache(tmp_path):
    checkpoint = tmp_path / "global_step_100" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_world_size_1_rank_0.pt").write_bytes(b"")
    hf_cache_dir = tmp_path / "cache"
    cached = hf_cache_dir / "arm1_floor_global_step_100"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    (cached / "model.safetensors").write_bytes(b"")

    calls = []
    result = resolve_hf_model_dir(
        checkpoint,
        "arm1_floor",
        hf_cache_dir=hf_cache_dir,
        force_remerge=True,
        merge_fn=lambda a, t: calls.append((a, t)),
    )

    assert result == cached
    assert calls == [(checkpoint, cached)]


def test_resolve_hf_model_dir_raises_when_neither_merged_nor_fsdp(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_hf_model_dir(tmp_path, "arm1_floor", hf_cache_dir=tmp_path / "cache")


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<thinking={enable_thinking}>{messages[-1]['content']}"


def test_build_chat_prompt_passes_through_settings():
    prompt = build_chat_prompt(_FakeTokenizer(), [{"role": "user", "content": "2+2?"}], enable_thinking=False)
    assert prompt == "<thinking=False>2+2?"


def test_score_eval_set_single_sample_accuracy():
    rows = [{"label": "4"}, {"label": "6"}]
    completions = [["\\boxed{4}"], ["\\boxed{5}"]]

    def check(pred, label):
        return pred == f"\\boxed{{{label}}}"

    result = score_eval_set(rows, completions, check_answer_fn=check)

    assert result == {"n_questions": 2, "n_samples_per_question": 1, "accuracy": 0.5}


def test_score_eval_set_multi_sample_reports_pass_at_n():
    rows = [{"label": "4"}]
    completions = [["\\boxed{4}", "\\boxed{5}", "\\boxed{5}"]]

    def check(pred, label):
        return pred == f"\\boxed{{{label}}}"

    result = score_eval_set(rows, completions, check_answer_fn=check)

    assert result["n_samples_per_question"] == 3
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["pass_at_3"] == 1.0


def test_score_eval_set_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        score_eval_set([{"label": "4"}], [], check_answer_fn=lambda pred, label: True)


def test_build_report_aggregates_unweighted_mean_accuracy():
    eval_set_results = {
        "math500": {"n_questions": 500, "n_samples_per_question": 1, "accuracy": 0.8},
        "aime24": {"n_questions": 30, "n_samples_per_question": 1, "accuracy": 0.2},
    }

    report = build_report(
        "arm1_floor", "checkpoints/.../actor", "checkpoints_hf/arm1_floor_step300", eval_set_results
    )

    assert report["arm_name"] == "arm1_floor"
    assert report["eval_sets"] == eval_set_results
    assert report["overall_accuracy"] == pytest.approx(0.5)
    assert "timestamp" in report
