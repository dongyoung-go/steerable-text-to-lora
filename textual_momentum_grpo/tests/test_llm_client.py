from types import SimpleNamespace

from tmgrpo.llm_client import DEFAULT_MODEL, LLMClient


class FakeOpenAI:
    """Mimics the slice of the openai.OpenAI client surface LLMClient.complete() touches."""

    def __init__(self, content="a response", prompt_tokens=10, completion_tokens=5):
        self._content = content
        self._usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=self._usage)


def test_default_model_is_gpt5_mini():
    assert DEFAULT_MODEL == "gpt-5-mini"


def test_complete_returns_response_text_and_uses_configured_model():
    fake = FakeOpenAI(content="4")
    client = LLMClient(model="gpt-5-mini", client=fake)

    result = client.complete("system prompt", "user prompt")

    assert result == "4"
    assert fake.last_kwargs["model"] == "gpt-5-mini"
    assert fake.last_kwargs["messages"][0] == {"role": "system", "content": "system prompt"}
    assert fake.last_kwargs["messages"][1] == {"role": "user", "content": "user prompt"}


def test_complete_tracks_call_count_and_token_usage():
    fake = FakeOpenAI(prompt_tokens=7, completion_tokens=3)
    client = LLMClient(client=fake)

    client.complete("s", "u")
    client.complete("s", "u")

    summary = client.usage_summary()
    assert summary["call_count"] == 2
    assert summary["total_prompt_tokens"] == 14
    assert summary["total_completion_tokens"] == 6


def test_complete_handles_missing_usage_gracefully():
    fake = FakeOpenAI()
    fake._usage = None
    client = LLMClient(client=fake)

    client.complete("s", "u")

    assert client.usage_summary()["total_prompt_tokens"] == 0
