"""``steerable_t2l.data.external_testsets``. See docs/04_downstream_eval.md's "full official
test set" follow-up. Only the pure/dispatch logic is tested here without network access;
the three real loaders (``load_gsm8k_test_rows``/``load_aqua_test_rows``/``load_bbh_test_rows``)
are exercised indirectly via ``test_eval_accuracy.py``'s ``rows_for_task`` override test, which
injects a fake loader instead of hitting the network.
"""

from __future__ import annotations

import pytest

from steerable_t2l.data.external_testsets import (
    _bbh_subtask_name,
    _format_aqua_option,
    load_external_test_rows,
)


def test_bbh_subtask_name_strips_task_prefix():
    assert _bbh_subtask_name("textgrad_repro_v2_bbh_causal_judgement") == "causal_judgement"
    assert _bbh_subtask_name("bbh_dyck_languages") == "dyck_languages"


def test_bbh_subtask_name_strips_v3_instruction_suffix():
    # build_tasks_from_{textgrad,gepa}_repro_v3.py name per-instruction task dirs
    # "..._bbh_<subtask>_d<K>" -- the "_d<K>" tail must not leak into the BBH URL slug.
    assert _bbh_subtask_name("textgrad_repro_v3_bbh_causal_judgement_d0") == "causal_judgement"
    assert _bbh_subtask_name("gepa_repro_v3_bbh_word_sorting_d12") == "word_sorting"


def test_bbh_subtask_name_rejects_non_bbh_task():
    with pytest.raises(ValueError):
        _bbh_subtask_name("textgrad_repro_v2_aqua")


def test_format_aqua_option_adds_space_after_letter():
    assert _format_aqua_option("A)21") == "A) 21"
    assert _format_aqua_option("no-paren-option") == "no-paren-option"


def test_load_external_test_rows_unknown_domain_raises_loudly():
    with pytest.raises(KeyError):
        load_external_test_rows("some_untracked_domain", "textgrad_repro_v2_some_untracked_domain")


def test_load_external_test_rows_dispatches_other_domain_by_task_name_suffix(monkeypatch):
    # build_tasks_from_textgrad_repro_v2.py's domain_for() tags all of mmlu_all/gpqa_main/
    # commonsenseqa/strategyqa/trec/multiarith as domain "other" -- dispatch must fall back to
    # matching the task-name suffix against _OTHER_DOMAIN_LOADERS rather than KeyError on
    # domain alone. Monkeypatch the loader itself so this stays network-free.
    import steerable_t2l.data.external_testsets as ext

    called = []

    def fake_loader(task_name):
        called.append(task_name)
        return [{"question": "q", "response": "", "gold_answer": "LOC"}]

    monkeypatch.setitem(ext._OTHER_DOMAIN_LOADERS, "trec", fake_loader)

    rows = load_external_test_rows("other", "textgrad_repro_v2_trec")

    assert called == ["textgrad_repro_v2_trec"]
    assert rows[0]["gold_answer"] == "LOC"


def test_load_external_test_rows_dispatches_other_domain_with_v3_instruction_suffix(monkeypatch):
    # Same as above, but with a v3-style "..._d<K>" task dir name -- the suffix match must
    # strip it before comparing, not require an exact "_trec" tail.
    import steerable_t2l.data.external_testsets as ext

    called = []

    def fake_loader(task_name):
        called.append(task_name)
        return [{"question": "q", "response": "", "gold_answer": "LOC"}]

    monkeypatch.setitem(ext._OTHER_DOMAIN_LOADERS, "trec", fake_loader)

    rows = load_external_test_rows("other", "textgrad_repro_v3_trec_d3")

    # the loader still receives the full, un-stripped task name (needed downstream elsewhere)
    assert called == ["textgrad_repro_v3_trec_d3"]
    assert rows[0]["gold_answer"] == "LOC"
