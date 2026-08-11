from tmgrpo.critique import RolloutSample, generate_critique, pool_original_and_refined


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def complete(self, system_prompt, user_prompt, temperature=1.0):
        self.calls.append((system_prompt, user_prompt))
        return self._response


def test_generate_critique_includes_problem_and_response():
    client = FakeClient("looks correct")
    result = generate_critique(client, problem="What is 2+2?", response="4")
    assert result == "looks correct"
    user_prompt = client.calls[0][1]
    assert "What is 2+2?" in user_prompt
    assert "4" in user_prompt


def test_pool_original_and_refined_concatenates_both_groups():
    original = [RolloutSample(response="a", reward=0.0), RolloutSample(response="b", reward=1.0)]
    refined = [RolloutSample(response="a2", reward=1.0, conditioning_context="critique text")]

    pooled = pool_original_and_refined(original, refined)

    assert pooled == [*original, *refined]
    assert len(pooled) == 3


def test_pool_original_and_refined_preserves_conditioning_context():
    refined = [RolloutSample(response="x", reward=1.0, conditioning_context="be more careful")]
    pooled = pool_original_and_refined([], refined)
    assert pooled[0].conditioning_context == "be more careful"
