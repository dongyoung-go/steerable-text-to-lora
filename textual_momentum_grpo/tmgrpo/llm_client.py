"""Thin OpenAI wrapper for every frontier-model call in this project.

Used by critique.py (arm 2/3 same-iteration critique), trajectory.py (arm 4/5 textual gradient,
trajectory-digest update, momentum generation). Model defaults to gpt-5-mini per an explicit
user decision -- do not change this default without asking again.

API key is read from the OPENAI_API_KEY environment variable (never hardcoded, never logged). A
real shell-exported OPENAI_API_KEY always wins; `textual_momentum_grpo/.env` (gitignored, see
`.env.example` for the expected format) is only consulted to fill in what's not already set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# override=False: real environment variables (e.g. exported by the shell or the GPU node's job
# launcher) take precedence over anything in .env, so .env is purely a local-dev convenience.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

DEFAULT_MODEL = "gpt-5-mini"


@dataclass
class LLMClient:
    """Wraps an OpenAI client + model choice + call accounting.

    `call_count` and cumulative token usage are tracked here because README section 6 asks that
    frontier-model cost be reported alongside performance gains, not treated as a footnote.
    """

    model: str = DEFAULT_MODEL
    client: OpenAI = field(default_factory=lambda: OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
        """One chat completion; returns the response text and updates call/token counters."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        self.call_count += 1
        usage = response.usage
        if usage is not None:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
        return response.choices[0].message.content or ""

    def usage_summary(self) -> dict:
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
