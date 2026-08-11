import pytest

from tmgrpo.verl_hooks import compute_score, inject_conditioning_context, recompute_unconditioned_logprobs


def test_compute_score_matches_verl_contract_signature():
    assert compute_score("math", "\\boxed{4}", "4") == 1.0
    assert compute_score("math", "\\boxed{4}", "5", extra_info={}) == 0.0


def test_inject_conditioning_context_appends_user_turn():
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    result = inject_conditioning_context(prompt, "Try being more careful.")
    assert len(result) == 2
    assert result[0] == prompt[0]
    assert result[1]["role"] == "user"
    assert "Try being more careful." in result[1]["content"]


def test_inject_conditioning_context_no_op_on_empty_context():
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    assert inject_conditioning_context(prompt, "") == prompt


def test_recompute_unconditioned_logprobs_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        recompute_unconditioned_logprobs()
