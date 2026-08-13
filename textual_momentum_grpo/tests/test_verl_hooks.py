from tmgrpo.verl_hooks import compute_score, inject_conditioning_context, truncate_head_tail


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


def test_truncate_head_tail_no_op_under_budget():
    assert truncate_head_tail("short text", head=600, tail=600) == "short text"


def test_truncate_head_tail_keeps_both_ends():
    text = "A" * 1000 + "MIDDLE" + "B" * 1000 + "\\boxed{42}"
    result = truncate_head_tail(text, head=600, tail=600)
    assert result.startswith("A" * 600)
    assert result.endswith("\\boxed{42}")
    assert "MIDDLE" not in result
    assert "…[truncated]…" in result


def test_truncate_head_tail_head_only_when_tail_zero():
    text = "A" * 1000
    result = truncate_head_tail(text, head=450, tail=0)
    assert result == "A" * 450 + " …[truncated]"
