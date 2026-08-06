"""Shared fixtures. Everything is tiny, CPU-only, fp32, and downloads no model weights."""

from __future__ import annotations

import pytest
import torch

from steerable_t2l import testing as fixtures


@pytest.fixture(scope="session", autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="session")
def tokenizer():
    tok = fixtures.get_tokenizer()
    if tok is None:
        pytest.skip("no Qwen2.5 tokenizer available (needs the HF cache or network)")
    return tok


@pytest.fixture
def target_config():
    return fixtures.tiny_target_config()


@pytest.fixture
def spec(target_config):
    return fixtures.tiny_spec(target_config)


@pytest.fixture
def target_model(target_config):
    torch.manual_seed(1234)
    return fixtures.tiny_target_model(target_config)


@pytest.fixture
def target_model_for_tokenizer(target_config, tokenizer):
    """A tiny target model sized to the real tokenizer's vocabulary -- for tests that run
    real chat-templated/tokenized text through the target, as opposed to synthetic randint
    token ids (which `target_model` is paired with)."""
    torch.manual_seed(1234)
    return fixtures.tiny_target_model_for_tokenizer(tokenizer, target_config)


@pytest.fixture
def hypernet(spec, tokenizer):
    torch.manual_seed(1234)
    return fixtures.tiny_hypernet(spec, tokenizer)


DESCS = [
    "Solve the problem step by step and verify every arithmetic operation.",
    "Answer with only the final number. Do not show any reasoning.",
    "Decompose the problem into sub-steps, then state the answer.",
]
