"""Gold-answer join, pure-function slice (no network). See docs/04_downstream_eval.md §6."""

from __future__ import annotations

import pytest

from steerable_t2l.data.gold_answers import gold_answer, strip_question_prefix


def test_strip_question_prefix_removes_prefix():
    assert strip_question_prefix("Question: what is 2+2?") == "what is 2+2?"


def test_strip_question_prefix_passthrough_without_prefix():
    assert strip_question_prefix("what is 2+2?") == "what is 2+2?"


def test_gold_answer_joins_with_or_without_prefix():
    index = {"what is 2+2?": "#### 4"}
    assert gold_answer(index, "Question: what is 2+2?") == "#### 4"
    assert gold_answer(index, "what is 2+2?") == "#### 4"


def test_gold_answer_raises_loudly_on_miss():
    index = {"what is 2+2?": "#### 4"}
    with pytest.raises(KeyError):
        gold_answer(index, "Question: unrelated question?")
