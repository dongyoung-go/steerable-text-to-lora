"""Online critique + rewrite calls for the iterative-T2L application experiment. See
``docs/07_iterative_t2l_application_v3.md``.

Wraps a single vLLM Qwen3-14B engine with two call shapes, mirroring TextGrad's own backward
(textual gradient) + ``optimizer.step()`` (prompt rewrite) decomposition
(``scripts/textgrad_repro.py``) -- but as direct prompt/response calls rather than by driving the
``textgrad`` library itself. The library's ``tg.BlackboxLLM`` assumes the text being optimized is
literally injected into the solver's own forward pass every round; that's false here from round 1
onward, where only T2L ever sees the current text and the target model only ever sees the bare
question (``data/formatting.py::format_example``'s invariant, unchanged by this experiment).

``critique()`` plays TextGrad's per-example textual-gradient role: given the current
instruction/feedback text and a batch of ``(question, response, gold)`` triples, produce a
critique of what's wrong and how to fix it.

``rewrite()`` plays TextGrad's ``optimizer.step()`` role, in one of two modes:
  - ``mode="prompt"`` (v3-shaped, the pilot default): rewrite the instruction text directly.
  - ``mode="comprehensive_feedback"`` (v4-shaped, not run elsewhere yet either -- see docs/05):
    reuses that document's exact merge-prompt template, so this loop can later point at a
    v4-shaped checkpoint with no template drift between the two experiments.
"""

from __future__ import annotations

CRITIQUE_PROMPT = """You are critiquing a language model's attempts at solving problems of a \
particular kind, so its instructions can be improved. Here is the current instruction it was \
given (or guidance it has been steered by):

{current_text}

Here are {n} examples of the model's recent attempts, each with the question, the model's \
response, and the correct answer:

{examples}

Write a concise critique of what is going wrong across these examples and how the instruction \
or guidance should change to fix it. Focus on generalizable mistakes (e.g. wrong reasoning \
strategy, wrong answer format, missed constraints), not on restating individual answers. Output \
only the critique, nothing else."""

REWRITE_PROMPT_PROMPT_MODE = """You are improving a system prompt for a language model, based on \
feedback about its recent mistakes.

Current prompt:
{current_text}

Feedback on recent mistakes:
{feedback}

Rewrite the prompt to address this feedback while keeping it a clear, self-contained instruction \
for solving problems of this kind. Preserve any correct guidance the current prompt already \
contains. Output only the rewritten prompt, nothing else."""

# Verbatim from scripts/generate_comprehensive_feedback_v4.py's MERGE_PROMPT (docs/05), with
# "New feedback from this round" now sourced from this loop's single critique call rather than
# TextGrad's 3 per-round textual gradients -- see this module's docstring for why.
REWRITE_PROMPT_COMPREHENSIVE_FEEDBACK_MODE = """You are maintaining a running, generalized set of \
guidance notes for how to correctly solve problems of this kind. The notes must generalize beyond \
any single question -- do not reference specific numbers, names, or exact wording from the \
examples below; state the underlying principle or strategy instead.

Previous guidance notes: {current_text}

New feedback from this round (a critique of what went wrong and how to fix it):
{feedback}

Merge the previous guidance notes (if any) with the new feedback into a single, self-contained, \
generalized paragraph of guidance for solving problems of this kind. Remove redundant points. \
Resolve contradictions in favor of the more recent feedback. Do not write a system prompt or \
instruction to a model -- write guidance/feedback notes only. Keep it concise: no more than \
{max_words} words. Output only the merged paragraph, nothing else."""

REWRITE_MODES = ("prompt", "comprehensive_feedback")


def load_vllm_engine(model_dir: str, gpu_memory_utilization: float, max_model_len: int, seed: int):
    """Same load pattern as ``generate_comprehensive_feedback_v4.py::load_vllm_engine`` -- kept
    as a separate copy (not imported from there) since that script is a batch offline tool run
    once per source dir, while this one drives a single engine instance reused across many small
    online calls inside ``iterative_t2l.run_iterative_t2l``'s round loop."""
    from vllm import LLM

    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    return llm, llm.get_tokenizer()


def _build_chat_prompt(tokenizer, user_content: str) -> str:
    # thinking off: text analysis over feedback, not the hard task itself -- same rationale
    # generate_comprehensive_feedback_v4.py's build_chat_prompt gives.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_one(llm, tokenizer, user_content: str, temperature: float, max_tokens: int) -> str:
    from vllm import SamplingParams

    prompt = _build_chat_prompt(tokenizer, user_content)
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def build_critique_content(current_text: str, examples: list[dict]) -> str:
    """``examples``: list of ``{"question", "response", "gold"}``."""
    blocks = [
        f"- Question: {ex['question']}\n  Model response: {ex['response']}\n  "
        f"Correct answer: {ex['gold']}"
        for ex in examples
    ]
    return CRITIQUE_PROMPT.format(
        current_text=current_text or "(none yet)", n=len(examples), examples="\n".join(blocks)
    )


def critique(
    llm,
    tokenizer,
    current_text: str,
    examples: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """One critique call over a batch of ``(question, response, gold)`` triples."""
    content = build_critique_content(current_text, examples)
    return _generate_one(llm, tokenizer, content, temperature, max_tokens)


def build_rewrite_content(current_text: str, feedback: str, *, mode: str, max_words: int) -> str:
    if mode == "prompt":
        return REWRITE_PROMPT_PROMPT_MODE.format(
            current_text=current_text or "(none yet)", feedback=feedback
        )
    if mode == "comprehensive_feedback":
        return REWRITE_PROMPT_COMPREHENSIVE_FEEDBACK_MODE.format(
            current_text=current_text or "", feedback=feedback, max_words=max_words
        )
    raise ValueError(f"unknown rewrite mode {mode!r} -- expected one of {REWRITE_MODES}")


def rewrite(
    llm,
    tokenizer,
    current_text: str,
    feedback: str,
    *,
    mode: str = "prompt",
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_words: int = 150,
) -> str:
    """One rewrite call producing the next round's T2L input text."""
    content = build_rewrite_content(current_text, feedback, mode=mode, max_words=max_words)
    return _generate_one(llm, tokenizer, content, temperature, max_tokens)
