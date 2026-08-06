"""Shared vLLM plumbing for `scripts/gepa_repro.py`.

Ported from `/home/dg793/text-to-lora/scripts/gepa_repro_common.py` -- see
`gepa_repro_README.md`'s "What changed vs. the original" section for the
port deltas. GEPA needs one local vLLM engine to play two roles it itself
distinguishes: the *task* LM (solves the actual problem, scored by our
evaluator) and the *reflection* LM (critiques a minibatch of scored
rollouts and proposes an improved candidate). Unlike `textgrad`, GEPA has
no built-in local-inference engine -- its `reflection_lm` config field
accepts either a litellm model-name string or a plain callable satisfying
`gepa`'s `LanguageModel` protocol (`def __call__(self, prompt: str | list[dict])
-> str`, confirmed by reading `gepa_repro/src/gepa/proposer/
reflective_mutation/base.py` and `gepa_repro/src/gepa/gepa_launcher.py::
make_litellm_lm`) -- so we never need litellm, an HTTP server, or two
GPU-resident engines: one shared `vllm.LLM` plays both roles, the same
"one engine, multiple roles" philosophy `textgrad_repro.py`'s `ChatVLLM`
uses.

Unlike the original, `extract_final_int_answer` (which depended on
`fishfarm.tasks.language_restricted_math.extract_answer_number`, a
dependency this repo doesn't have) is dropped: `scripts/gepa_repro.py`
scores every task with `textgrad_repro.ANSWER_PARSERS` instead (the same
registry `scripts/textgrad_repro.py` uses), so no separate answer-
extraction helper is needed here.
"""

from vllm import LLM, SamplingParams


def load_vllm_engine(model_dir, gpu_memory_utilization=0.85, max_model_len=16384, seed=42):
    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def build_chat_prompt(tokenizer, system_prompt, user_prompt, enable_thinking=False):
    conversation = []
    if system_prompt:
        conversation = [{"role": "system", "content": system_prompt}]
    conversation += [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def batched_generate(
    llm, tokenizer, system_prompt, user_prompts, temperature=0.6, max_tokens=1024, top_p=0.95, top_k=20,
    enable_thinking=False,
):
    """One batched `vllm.LLM.generate(list_of_prompts, ...)` call for a whole
    minibatch/valset/test stage -- the same "batched call, not a thread pool
    against one in-process engine" deviation `textgrad_repro.py` documents.
    GEPA's `batch_evaluator` hook (all pairs of one evaluation stage arrive
    in a single call) is what makes this possible without fighting GEPA's
    own default per-pair thread pool.

    Sampling defaults (temperature=0.6, top_p=0.95, top_k=20) match the GEPA
    paper's Qwen3-8B settings (Appendix E.2: "we use a decoding temperature
    of 0.6, top-p of 0.95, and top-k of 20 for training as well as
    inference"), applied uniformly regardless of `enable_thinking` -- unlike
    an earlier version of this function, which silently fell back to greedy
    decoding (temperature=0.0, top_p=1.0) whenever thinking was off."""
    if not user_prompts:
        return []
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, n=1)
    chat_strs = [build_chat_prompt(tokenizer, system_prompt, p, enable_thinking=enable_thinking) for p in user_prompts]
    outputs = llm.generate(chat_strs, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


class VLLMLanguageModel:
    """Adapts the shared vLLM engine to GEPA's `LanguageModel` protocol
    (`def __call__(self, prompt: str | list[dict]) -> str`) so it can be
    passed directly as `ReflectionConfig(reflection_lm=...)`. GEPA calls this
    one prompt at a time (one reflective-mutation step per candidate
    proposal, not batched across candidates), so a single-item
    `llm.generate([...], ...)` call per invocation is the correct
    granularity here (unlike the task-solving path, which GEPA does let us
    batch via `batch_evaluator`). Reflection always runs with thinking off
    (see module docstring / gepa_repro_README.md) -- it's text-analysis
    over feedback, not itself the hard task, so there's no reason to pay
    for a `<think>` block here regardless of the solver's `--enable_thinking`
    setting."""

    def __init__(self, llm, tokenizer, system_prompt="", temperature=0.7, max_tokens=4096, top_p=0.95):
        self.llm = llm
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.num_calls = 0

    def __call__(self, prompt):
        self.num_calls += 1
        if isinstance(prompt, str):
            conversation = []
            if self.system_prompt:
                conversation = [{"role": "system", "content": self.system_prompt}]
            conversation += [{"role": "user", "content": prompt}]
        else:
            conversation = list(prompt)
        chat_str = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, n=1)
        outputs = self.llm.generate([chat_str], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text
