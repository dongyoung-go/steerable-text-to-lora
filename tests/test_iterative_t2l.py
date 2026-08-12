"""Inference-time iterative TextGrad-style refinement loop. See
docs/07_iterative_t2l_application_v3.md."""

from __future__ import annotations

import json
import random

import pytest
import yaml

from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.eval_accuracy import GenerationConfig
from steerable_t2l.iterative_t2l import (
    run_iterative_t2l,
    split_feedback_and_score_rows,
)


def _make_task_with_embedded_gold(root, name, n_rows=8):
    """A task whose rows carry a bare embedded 'gold_answer' (the v2/v3 shape) rather than the
    legacy external-join shape ``two_tasks`` in ``test_eval_accuracy.py`` uses -- this loop only
    supports the embedded-gold path (see ``iterative_t2l._resolve_parser``)."""
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"what is {i}+{i}?", "gold_answer": str(2 * i)}) + "\n")
    metadata = {
        "descriptions": [f"{name} instruction"],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "gold_answer",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)


@pytest.fixture
def one_task(tmp_path):
    _make_task_with_embedded_gold(tmp_path, "ta", n_rows=8)
    tasks = discover_tasks(tmp_path, ["t*"])
    splits = Splits(q_frac=1.0, d_holdout={"ta": []}, t_holdout=[], seed=0)
    return tasks, splits


# -- split_feedback_and_score_rows -------------------------------------------------------


def test_split_feedback_and_score_rows_disjoint_and_sized():
    rows = [{"i": i} for i in range(10)]
    feedback_rows, score_rows = split_feedback_and_score_rows(rows, 3, 4, random.Random(0))
    assert len(feedback_rows) == 3
    assert len(score_rows) == 4
    feedback_ids = {r["i"] for r in feedback_rows}
    score_ids = {r["i"] for r in score_rows}
    assert feedback_ids.isdisjoint(score_ids)


def test_split_feedback_and_score_rows_raises_when_not_enough_rows():
    rows = [{"i": i} for i in range(5)]
    with pytest.raises(ValueError, match="need at least"):
        split_feedback_and_score_rows(rows, 3, 4, random.Random(0))


def test_split_feedback_and_score_rows_deterministic_given_seed():
    rows = [{"i": i} for i in range(10)]
    a = split_feedback_and_score_rows(rows, 3, 3, random.Random(0))
    b = split_feedback_and_score_rows(rows, 3, 3, random.Random(0))
    assert a == b


# -- run_iterative_t2l (tiny CPU fixtures, fake critique/rewrite) -----------------------


def _fake_critique_rewrite_fns():
    """Deterministic stand-ins for the real Qwen3-14B calls -- no vLLM engine needed. Each
    rewrite just appends a counter so every round's text is distinct and traceable."""
    calls = {"critique": 0, "rewrite": 0}

    def critique_fn(current_text, examples):
        calls["critique"] += 1
        assert examples, "critique_fn should always see at least one example"
        return f"critique #{calls['critique']} of: {current_text[:20]}"

    def rewrite_fn(current_text, feedback_text):
        calls["rewrite"] += 1
        return f"{current_text} [rev{calls['rewrite']}]"

    return critique_fn, rewrite_fn, calls


def test_run_iterative_t2l_round_count_and_shape(one_task, tokenizer, target_model_for_tokenizer, spec, hypernet):
    tasks, splits = one_task
    critique_fn, rewrite_fn, calls = _fake_critique_rewrite_fns()

    report = run_iterative_t2l(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
        feedback_llm=None, feedback_tokenizer=None,
        n_rounds=2, feedback_n=2, score_n=2,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
        critique_fn=critique_fn, rewrite_fn=rewrite_fn,
    )

    assert report["task"] == "ta"
    assert report["n_rounds"] == 2
    assert len(report["rounds"]) == 3  # round 0, 1, 2
    assert calls == {"critique": 2, "rewrite": 2}  # one critique+rewrite per round *except* the last


def test_round_zero_has_no_lora_and_instruction_in_context(
    one_task, tokenizer, target_model_for_tokenizer, spec, hypernet
):
    tasks, splits = one_task
    critique_fn, rewrite_fn, _ = _fake_critique_rewrite_fns()

    report = run_iterative_t2l(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
        feedback_llm=None, feedback_tokenizer=None,
        n_rounds=2, feedback_n=2, score_n=2,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
        critique_fn=critique_fn, rewrite_fn=rewrite_fn,
    )

    round0 = report["rounds"][0]
    assert round0["used_lora"] is False
    assert round0["instruction_in_context"] is True

    for round_entry in report["rounds"][1:]:
        assert round_entry["used_lora"] is True
        assert round_entry["instruction_in_context"] is False


def test_lora_replaces_not_stacks_across_rounds(
    one_task, tokenizer, target_model_for_tokenizer, spec, hypernet, monkeypatch
):
    """Every generate_texts call opens at most one lora_hooks context, and no two are ever
    concurrently open -- confirms rounds replace the previous LoRA rather than stacking it."""
    import steerable_t2l.eval_accuracy as eval_accuracy_module

    active = {"count": 0, "max_seen": 0}
    orig_lora_hooks = eval_accuracy_module.lora_hooks

    from contextlib import contextmanager

    @contextmanager
    def counting_lora_hooks(*args, **kwargs):
        active["count"] += 1
        active["max_seen"] = max(active["max_seen"], active["count"])
        try:
            with orig_lora_hooks(*args, **kwargs):
                yield
        finally:
            active["count"] -= 1

    monkeypatch.setattr(eval_accuracy_module, "lora_hooks", counting_lora_hooks)

    tasks, splits = one_task
    critique_fn, rewrite_fn, _ = _fake_critique_rewrite_fns()

    run_iterative_t2l(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
        feedback_llm=None, feedback_tokenizer=None,
        n_rounds=2, feedback_n=2, score_n=2,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
        critique_fn=critique_fn, rewrite_fn=rewrite_fn,
    )

    assert active["max_seen"] == 1  # never more than one LoRA active at once
    assert active["count"] == 0  # every context was cleanly closed


def test_feedback_and_score_pools_fixed_across_rounds(
    one_task, tokenizer, target_model_for_tokenizer, spec, hypernet
):
    tasks, splits = one_task
    critique_fn, rewrite_fn, _ = _fake_critique_rewrite_fns()

    report = run_iterative_t2l(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
        feedback_llm=None, feedback_tokenizer=None,
        n_rounds=2, feedback_n=2, score_n=2,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
        critique_fn=critique_fn, rewrite_fn=rewrite_fn,
    )

    fb_questions_per_round = [
        [r["question"] for r in round_entry["feedback_pool_rows"]] for round_entry in report["rounds"]
    ]
    score_questions_per_round = [
        [r["question"] for r in round_entry["held_out_rows"]] for round_entry in report["rounds"]
    ]
    assert all(qs == fb_questions_per_round[0] for qs in fb_questions_per_round)
    assert all(qs == score_questions_per_round[0] for qs in score_questions_per_round)
    assert set(fb_questions_per_round[0]).isdisjoint(score_questions_per_round[0])


def test_report_round_trips_through_json(one_task, tokenizer, target_model_for_tokenizer, spec, hypernet):
    tasks, splits = one_task
    critique_fn, rewrite_fn, _ = _fake_critique_rewrite_fns()

    report = run_iterative_t2l(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
        feedback_llm=None, feedback_tokenizer=None,
        n_rounds=1, feedback_n=2, score_n=2,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
        critique_fn=critique_fn, rewrite_fn=rewrite_fn,
    )

    round_tripped = json.loads(json.dumps(report))
    assert round_tripped == report


def test_run_iterative_t2l_raises_without_enough_rows(one_task, tokenizer, target_model_for_tokenizer, spec, hypernet):
    tasks, splits = one_task
    critique_fn, rewrite_fn, _ = _fake_critique_rewrite_fns()

    with pytest.raises(ValueError, match="need at least"):
        run_iterative_t2l(
            hypernet, target_model_for_tokenizer, tokenizer, spec, tasks[0], splits, tasks,
            feedback_llm=None, feedback_tokenizer=None,
            n_rounds=1, feedback_n=10, score_n=10,  # only 8 rows exist
            gen_config=GenerationConfig(max_new_tokens=3, batch_size=2),
            critique_fn=critique_fn, rewrite_fn=rewrite_fn,
        )
