"""Generation-based downstream accuracy eval. See docs/04_downstream_eval.md."""

from __future__ import annotations

import json

import pytest
import yaml

from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import make_splits
from steerable_t2l.eval_accuracy import (
    GenerationConfig,
    build_prompted_prompt,
    classify_answer_parser,
    compute_comparisons,
    condition_desc,
    parse_exact_answer,
    parse_integer_answer,
    parse_mcq_letter_answer,
    run_downstream_eval,
    score_condition,
)
from steerable_t2l.hooks import build_sites


def _make_task(root, name, n_rows=6, n_desc=2, prefix=True, best_desc_idx=None):
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            q = f"Question: what is {i}+{i}?" if prefix else f"what is {i}+{i}?"
            f.write(json.dumps({"question": q, "response": f"Answer: {2 * i}"}) + "\n")
    metadata = {
        "descriptions": [f"{name} instruction {i}" for i in range(n_desc)],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    if best_desc_idx is not None:
        metadata["best_description_index"] = best_desc_idx
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)
    return {f"what is {i}+{i}?": f"#### {2 * i}" for i in range(n_rows)}


@pytest.fixture
def two_tasks(tmp_path):
    gold_a = _make_task(tmp_path, "ta", n_desc=2)
    gold_b = _make_task(tmp_path, "tb", n_desc=1)
    gold_index = {**gold_a, **gold_b}
    return discover_tasks(tmp_path, ["t*"]), gold_index


# -- pure functions -------------------------------------------------------------------


def test_parse_integer_answer_last_numeric_token():
    assert parse_integer_answer("reasoning...\nAnswer: 42") == 42


def test_parse_integer_answer_none_on_no_digits():
    assert parse_integer_answer("no digits here") is None


def test_parse_integer_answer_does_not_default_to_zero():
    # the vendored textgrad_repro.big_bench_hard.parse_integer_answer would return 0 here --
    # this module must not, since 0 could be a genuine gold answer (docs/04 §6's hazard).
    assert parse_integer_answer("") is None


def test_parse_mcq_letter_answer_prefers_answer_marker_over_parens():
    assert parse_mcq_letter_answer("(A) is wrong, actually Answer: C") == "C"


def test_parse_mcq_letter_answer_falls_back_to_parens_then_bare_letter():
    assert parse_mcq_letter_answer("I think it's (E)") == "E"
    assert parse_mcq_letter_answer("the answer is D") == "D"
    assert parse_mcq_letter_answer("no letter here at all 123") is None


def test_parse_exact_answer_strips_answer_marker_and_normalizes_case():
    assert parse_exact_answer("blah blah\nAnswer: Yes.") == "yes"
    assert parse_exact_answer("  Invalid  ") == "invalid"
    assert parse_exact_answer("") is None


def test_classify_answer_parser_integer():
    assert classify_answer_parser(["2200", "-5", "0"]) == "integer"


def test_classify_answer_parser_mcq_letter():
    assert classify_answer_parser(["C", "(E)", "A"]) == "mcq_letter"


def test_classify_answer_parser_exact_for_anything_else():
    assert classify_answer_parser(["Yes", "No"]) == "exact"
    assert classify_answer_parser(["invalid", "valid"]) == "exact"
    assert classify_answer_parser(["] ) )"]) == "exact"


def test_classify_answer_parser_survives_a_single_outlier():
    # Real case: BBH movie_recommendation's official test JSON has one row whose target is
    # "Monsters, Inc" (a full movie title) among 99 clean single-letter targets -- a strict
    # all() would collapse the whole task to "exact" over one data-quality outlier, silently
    # breaking parsing for every other row too. Majority vote (>=90%) must still pick the
    # dominant shape.
    mostly_letters = [f"({chr(65 + i % 4)})" for i in range(99)] + ["Monsters, Inc"]
    assert classify_answer_parser(mostly_letters) == "mcq_letter"

    mostly_integers = [str(i) for i in range(99)] + ["not a number"]
    assert classify_answer_parser(mostly_integers) == "integer"

    # But a genuine near-even mix must still fall back to "exact", not get forced into
    # whichever shape happens to have a bare majority.
    assert classify_answer_parser(["A", "B", "Yes", "No", "maybe", "C"]) == "exact"
    assert classify_answer_parser([]) == "exact"


def test_score_condition_uses_embedded_gold_answer_and_classified_parser(
    two_tasks, tokenizer, target_model_for_tokenizer, spec
):
    # A task whose rows carry an embedded "gold_answer" (mcq-letter shaped) must be scored
    # via the classified mcq_letter parser against that field, never the legacy GSM8K
    # integer/external-join path -- regardless of what's in gold_index.
    import random

    tasks, gold_index = two_tasks
    task = tasks[0]
    rows = [{"question": "what is 0+0?", "response": "Answer: A", "gold_answer": "A"}]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    result = score_condition(
        None, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "base",
        rows, gold_index, None, random.Random(0), GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert result["n"] == 1


def test_condition_desc_base_and_oracle_need_none(two_tasks):
    import random

    tasks, _ = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.34, seed=0)
    task = tasks[0]
    rng = random.Random(0)
    assert condition_desc(task, splits, tasks, "base", rng) is None
    assert condition_desc(task, splits, tasks, "oracle", rng) is None


def test_condition_desc_gibberish_is_first_entry(two_tasks):
    import random

    from steerable_t2l.validation import GIBBERISH_DESCS

    tasks, _ = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.34, seed=0)
    task = tasks[0]
    assert condition_desc(task, splits, tasks, "t2l_gibberish_desc", random.Random(0)) == GIBBERISH_DESCS[0]


def test_condition_desc_t2l_train_and_prompted_share_the_same_desc(two_tasks):
    import random

    tasks, _ = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.34, seed=0)
    task = tasks[0]
    train_desc = condition_desc(task, splits, tasks, "t2l_train_desc", random.Random(0))
    prompted_desc = condition_desc(task, splits, tasks, "prompted", random.Random(0))
    assert train_desc == prompted_desc
    assert train_desc in task.metadata.descriptions


def test_condition_desc_prefers_best_description_index_over_pool_zero(tmp_path):
    """A task whose best-textgrad-iteration description isn't descriptions[0] must have
    prompted/t2l_train_desc use that best one, not silently fall back to the pool's first
    (usually unoptimized-seed) entry -- see docs/04's condition_desc fix."""
    import random

    _make_task(tmp_path, "tc", n_desc=3, best_desc_idx=2)
    tasks = discover_tasks(tmp_path, ["t*"])
    task = tasks[0]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.0, seed=0)

    train_desc = condition_desc(task, splits, tasks, "t2l_train_desc", random.Random(0))
    prompted_desc = condition_desc(task, splits, tasks, "prompted", random.Random(0))

    assert train_desc == "tc instruction 2"
    assert prompted_desc == "tc instruction 2"


def test_condition_desc_falls_back_to_pool_zero_when_best_is_held_out(tmp_path):
    """If the recorded best description is itself D-held-out (excluded from the train_descs
    pool), condition_desc must not return a description outside the pool -- fall back to the
    pool's first entry instead, same as when best_description_index is unset entirely."""
    import random

    from steerable_t2l.data.splits import Splits

    _make_task(tmp_path, "td", n_desc=3, best_desc_idx=2)
    tasks = discover_tasks(tmp_path, ["t*"])
    task = tasks[0]
    splits = Splits(q_frac=0.0, d_holdout={task.name: [2]}, t_holdout=[], seed=0)

    train_desc = condition_desc(task, splits, tasks, "t2l_train_desc", random.Random(0))

    assert train_desc == "td instruction 0"


def test_condition_desc_legacy_task_without_best_index_uses_pool_zero(two_tasks):
    """Tasks built before best_description_index existed (best_description_index is None)
    keep the old pool[0] behavior -- backward compatible, not a hard requirement."""
    import random

    tasks, _ = two_tasks
    task = tasks[0]
    assert task.metadata.best_description_index is None
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.34, seed=0)
    desc = condition_desc(task, splits, tasks, "t2l_train_desc", random.Random(0))
    assert desc == task.metadata.descriptions[0]


def test_build_prompted_prompt_contains_desc_and_question(tokenizer):
    text = build_prompted_prompt(tokenizer, "Think step by step.", "what is 2+2?")
    assert "Think step by step." in text
    assert "what is 2+2?" in text


# -- generation-backed scoring (tiny CPU fixtures) -------------------------------------


def test_score_condition_base(two_tasks, tokenizer, target_model_for_tokenizer, spec):
    import random

    from steerable_t2l.eval_accuracy import eval_rows_for_task

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task = tasks[0]
    rows = eval_rows_for_task(task, splits)
    assert rows

    result = score_condition(
        None, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "base",
        rows, gold_index, None, random.Random(0), GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert isinstance(result, dict)
    assert result["n"] == len(rows)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_score_condition_t2l_train_desc(two_tasks, tokenizer, target_model_for_tokenizer, spec, hypernet):
    import random

    from steerable_t2l.eval_accuracy import eval_rows_for_task

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task = tasks[0]
    rows = eval_rows_for_task(task, splits)

    result = score_condition(
        hypernet, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "t2l_train_desc",
        rows, gold_index, None, random.Random(0), GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert isinstance(result, dict)
    assert result["n"] == len(rows)


def test_score_condition_t2l_needs_hypernet(two_tasks, tokenizer, target_model_for_tokenizer, spec):
    import random

    from steerable_t2l.eval_accuracy import eval_rows_for_task

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task = tasks[0]
    rows = eval_rows_for_task(task, splits)

    result = score_condition(
        None, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "t2l_train_desc",
        rows, gold_index, None, random.Random(0), GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert result == "n/a"


def test_score_condition_oracle_via_synthetic_adapter(two_tasks, tokenizer, target_model_for_tokenizer, spec, tmp_path):
    import random

    from peft import get_peft_model

    from steerable_t2l.eval_accuracy import eval_rows_for_task

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task = tasks[0]
    rows = eval_rows_for_task(task, splits)

    peft_model = get_peft_model(target_model_for_tokenizer, spec.to_lora_config())
    oracle_dir = tmp_path / "oracle" / task.name
    oracle_dir.mkdir(parents=True)
    peft_model.save_pretrained(str(oracle_dir))

    result = score_condition(
        None, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "oracle",
        rows, gold_index, str(tmp_path / "oracle"), random.Random(0), GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert isinstance(result, dict)
    assert result["n"] == len(rows)


def test_score_condition_oracle_na_when_missing(two_tasks, tokenizer, target_model_for_tokenizer, spec, tmp_path):
    import random

    from steerable_t2l.eval_accuracy import eval_rows_for_task

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task = tasks[0]
    rows = eval_rows_for_task(task, splits)

    result = score_condition(
        None, target_model_for_tokenizer, tokenizer, spec, task, splits, tasks, "oracle",
        rows, gold_index, str(tmp_path / "does_not_exist"), random.Random(0),
        GenerationConfig(max_new_tokens=3, batch_size=2),
    )
    assert result == "n/a"


# -- aggregation ------------------------------------------------------------------------


def test_compute_comparisons():
    per_task = {
        "t0": {
            "base": {"accuracy": 0.1, "n": 10, "n_correct": 1},
            "prompted": {"accuracy": 0.3, "n": 10, "n_correct": 3},
            "oracle": {"accuracy": 0.8, "n": 10, "n_correct": 8},
            "t2l_train_desc": {"accuracy": 0.5, "n": 10, "n_correct": 5},
        },
    }
    result = compute_comparisons(per_task)
    assert result["per_task"]["t0"]["t2l_train_desc_minus_base"] == pytest.approx(0.4)
    assert result["per_task"]["t0"]["t2l_train_desc_minus_prompted"] == pytest.approx(0.2)
    assert result["per_task"]["t0"]["t2l_train_desc_over_oracle"] == pytest.approx(0.625)
    assert result["macro"]["t2l_train_desc_minus_base"] == pytest.approx(0.4)


# -- end-to-end + progress callbacks -----------------------------------------------------


def test_run_downstream_eval_fires_progress_callbacks(two_tasks, tokenizer, target_model_for_tokenizer, spec, hypernet):
    from steerable_t2l.eval_accuracy import CONDITIONS

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)

    starts: list[tuple[str, str, int]] = []
    dones: list[tuple[str, str]] = []

    def on_start(task_name, condition, n_rows):
        starts.append((task_name, condition, n_rows))

    def on_done(task_name, condition, result_entry):
        dones.append((task_name, condition))
        assert isinstance(result_entry, dict) or result_entry == "n/a"

    result = run_downstream_eval(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks, splits,
        oracle_dir=None, gen_config=GenerationConfig(max_new_tokens=3, batch_size=2), seed=0,
        gold_index=gold_index, on_condition_start=on_start, on_condition_done=on_done,
    )

    expected_pairs = {(t.name, c) for t in tasks for c in CONDITIONS}
    assert {(s[0], s[1]) for s in starts} == expected_pairs
    assert set(dones) == expected_pairs
    assert set(result["per_task"]) == {t.name for t in tasks}
    assert set(result["overall"]) == set(CONDITIONS)


def test_run_downstream_eval_resumes_from_existing_and_skips_generation(two_tasks, tokenizer, target_model_for_tokenizer, spec, hypernet):
    """A pair already present in ``existing`` must be reused verbatim, not re-scored --
    the whole point of resuming after an interrupted real run is to not redo expensive
    generation. Verified here by handing it a sentinel result no real ``score_condition``
    call could produce, and checking it comes back unchanged."""
    from steerable_t2l.eval_accuracy import CONDITIONS

    tasks, gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)
    task0 = tasks[0]

    sentinel = {"accuracy": 0.4242, "n": 999, "n_correct": 999}
    existing = {task0.name: {"base": sentinel}}

    starts: list[tuple[str, str]] = []

    def on_start(task_name, condition, n_rows):
        starts.append((task_name, condition))

    result = run_downstream_eval(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks, splits,
        oracle_dir=None, gen_config=GenerationConfig(max_new_tokens=3, batch_size=2), seed=0,
        gold_index=gold_index, on_condition_start=on_start, existing=existing,
    )

    assert result["per_task"][task0.name]["base"] == sentinel
    assert (task0.name, "base") not in starts  # never re-generated
    # every other pair still ran for real
    all_pairs = {(t.name, c) for t in tasks for c in CONDITIONS}
    assert set(starts) == all_pairs - {(task0.name, "base")}


def test_run_downstream_eval_rows_for_task_overrides_default_q_holdout(
    two_tasks, tokenizer, target_model_for_tokenizer, spec, hypernet
):
    """``rows_for_task`` (the full-official-test-set hook) must fully replace
    ``eval_rows_for_task``'s Q-holdout rows, not merely supplement them -- scoring against a
    bigger/disjoint official test set only makes sense if every condition sees exactly those
    rows and nothing from the small held-out split."""
    from steerable_t2l.eval_accuracy import CONDITIONS

    tasks, _gold_index = two_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.5, seed=0)

    external_rows = {
        t.name: [{"question": f"external q{i}", "response": "x", "gold_answer": "0"} for i in range(3)]
        for t in tasks
    }
    seen_n_rows: dict[str, int] = {}

    def on_start(task_name, condition, n_rows):
        seen_n_rows[task_name] = n_rows

    result = run_downstream_eval(
        hypernet, target_model_for_tokenizer, tokenizer, spec, tasks, splits,
        oracle_dir=None, conditions=CONDITIONS,
        gen_config=GenerationConfig(max_new_tokens=3, batch_size=2), seed=0,
        on_condition_start=on_start,
        rows_for_task=lambda task: external_rows[task.name],
    )

    for task in tasks:
        assert seen_n_rows[task.name] == 3
        for condition in CONDITIONS:
            entry = result["per_task"][task.name][condition]
            if isinstance(entry, dict):
                assert entry["n"] == 3


# -- hook coverage across a multi-token generate() call (docs/04 §4's ⚠️) ---------------


def test_lora_hooks_stay_attached_across_multi_token_generate(monkeypatch, tokenizer, target_model_for_tokenizer, spec, hypernet):
    import steerable_t2l.hooks as hooks_mod
    from steerable_t2l.eval_accuracy import _expand_per_module

    calls = []
    orig_make_hook = hooks_mod._make_hook

    def counting_make_hook(A, B, scaling, dropout):
        hook = orig_make_hook(A, B, scaling, dropout)

        def wrapped(module, args, output):
            calls.append(1)
            return hook(module, args, output)

        return wrapped

    monkeypatch.setattr(hooks_mod, "_make_hook", counting_make_hook)

    per_module = hypernet.generate_for_batch(["solve carefully"])
    per_module_fixed = {m: (A[0], B[0]) for m, (A, B) in per_module.items()}
    sites = build_sites(spec, _expand_per_module(per_module_fixed, 1))
    n_sites = len(sites)
    assert n_sites > 0

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "what is 2+2?"}], tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)

    with hooks_mod.lora_hooks(target_model_for_tokenizer, sites, spec.scaling):
        target_model_for_tokenizer.generate(
            **enc, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id
        )

    # One firing per site for the prefill call alone would be exactly `n_sites`; more than
    # that proves the hooks were still attached for at least one incremental decode step too.
    assert len(calls) > n_sites
