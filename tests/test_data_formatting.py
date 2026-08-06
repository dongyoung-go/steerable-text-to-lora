"""Pair-encoding label masking and truncation reporting. See docs/03_training_validation.md §1."""

from __future__ import annotations

from steerable_t2l.data.formatting import build_cache_key, format_example, tokenize_pair
from steerable_t2l.data.metadata import TaskMetadata


def _metadata(**overrides):
    kwargs = dict(
        descriptions=("do the thing",),
        ds_kwargs={"path": "json", "data_files": "x.jsonl", "split": "train"},
        response_field="response",
        user_prompt_template="{question}",
    )
    kwargs.update(overrides)
    return TaskMetadata(**kwargs)


def test_format_example_uses_arbitrary_row_fields(tokenizer):
    metadata = _metadata(user_prompt_template="{passage} -- {query}")
    row = {"passage": "P", "query": "Q", "response": "R"}
    prompt_text, response_text = format_example(row, metadata, tokenizer)
    assert "P" in prompt_text and "Q" in prompt_text
    assert response_text == "R"


def test_format_example_applies_assistant_prefill(tokenizer):
    metadata = _metadata(assistant_prefill="Answer: ")
    row = {"question": "2+2?", "response": "4"}
    _, response_text = format_example(row, metadata, tokenizer)
    assert response_text == "Answer: 4"


def test_format_example_omits_system_role(tokenizer):
    # system_message is always "" (enforced by TaskMetadata); format_example must build the
    # chat-template call with a user-only message list, never an empty system turn.
    metadata = _metadata()
    row = {"question": "2+2?", "response": "4"}
    prompt_text, _ = format_example(row, metadata, tokenizer)

    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": metadata.user_prompt_template.format(**row)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt_text == user_only


def test_tokenize_pair_masks_prompt_labels(tokenizer):
    metadata = _metadata()
    row = {"question": "2+2?", "response": "4"}
    prompt_text, response_text = format_example(row, metadata, tokenizer)
    tok = tokenize_pair(tokenizer, prompt_text, response_text, inp_max_len=512)

    assert tok.input_ids.shape == tok.labels.shape
    assert not tok.response_truncated

    # Some labels are real tokens (the response), the rest are masked.
    assert (tok.labels != -100).any()
    assert (tok.labels == -100).any()

    # Decoding only the non-masked labels recovers exactly the response text region.
    real_ids = tok.labels[tok.labels != -100].tolist()
    decoded = tokenizer.decode(real_ids)
    assert "4" in decoded


def test_tokenize_pair_reports_response_truncation(tokenizer):
    metadata = _metadata()
    long_response = " ".join(["word"] * 500)
    row = {"question": "2+2?", "response": long_response}
    prompt_text, response_text = format_example(row, metadata, tokenizer)

    tok_full = tokenize_pair(tokenizer, prompt_text, response_text, inp_max_len=4096)
    assert not tok_full.response_truncated

    tok_cut = tokenize_pair(tokenizer, prompt_text, response_text, inp_max_len=8)
    assert tok_cut.response_truncated
    assert tok_cut.input_ids.numel() == 8


def test_build_cache_key_changes_with_inp_max_len(tokenizer):
    metadata = _metadata()
    k1 = build_cache_key(metadata, tokenizer.name_or_path, 512, {})
    k2 = build_cache_key(metadata, tokenizer.name_or_path, 1024, {})
    assert k1 != k2
    assert build_cache_key(metadata, tokenizer.name_or_path, 512, {}) == k1
