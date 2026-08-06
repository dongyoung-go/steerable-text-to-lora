"""Faithful reproduction of TextGrad's own prompt-optimization algorithm,
using the actual `textgrad` library (cloned into `textgrad_repro/`,
installed as an ephemeral `uv run --with-editable` overlay -- see
`scripts/textgrad_repro_run.sh` / `textgrad_repro_README.md`), not a
reimplementation of it. Originally GSM8K-only (hence this file's history --
see below); `--task` now selects among ~29 tasks, see the `TASKS` registry.

Ported from `/home/dg793/text-to-lora/scripts/textgrad_repro_gsm8k.py`
(same experiment, same textgrad primitives, same JSONL schema) onto this
repo's newer, unpinned environment (`transformers>=5.0`, `torch>=2.9`,
`vllm>=0.11` via the `gen` extra) instead of text-to-lora's hard pins
(`transformers==4.51.1`, `torch==2.7.0`, `vllm==0.9.2`) -- see
textgrad_repro_README.md's "Why this lives here" section for what forced
the port. The algorithm, logging schema, and both disclosed monkeypatches
(add_generation_prompt=True, concise BACKWARD_SYSTEM_PROMPT) are unchanged.

One addition not present in the original: `--enable_thinking` /
`--no_enable_thinking` (default: thinking on, matching Qwen3's own
default), threaded through every chat-template call so reasoning models
(Qwen3 family) can be run either way -- see `build_chat_prompt` and
`_patch_chat_vllm_engine` below.

Task generalization: `--task` selects among GSM8K, all BIG-Bench Hard tasks
with a non-empty 250-row pool, MMLU/GPQA/AIME (extensions -- TextGrad's own
paper only reports prompt optimization on Object Counting/Word
Sorting/GSM8K; MMLU/GPQA are solution-optimization benchmarks upstream, and
AIME isn't in TextGrad at all), and five additional HF-backed tasks
(MultiArith/AQuA/CommonsenseQA/StrategyQA/TREC) for literature comparability
with OPRO/EvoPrompt/Promptbreeder. See `TASKS` below and
`TEXTGRAD_MULTITASK_PLAN.md` for the full registry and the reasoning behind
each task's parser/split/description choices.

This script never calls upstream's `load_task()`. For every BBH task except
`object_counting`, upstream's own `load_task()` would use
`MultiFieldTokenParsedEvaluation`, an LLM-as-judge eval. We deliberately do
not use that path for any task: all tasks here score with a deterministic
`ANSWER_PARSERS` entry (integer / mcq_letter / exact) applied symmetrically
to prediction and gold via `StringBasedFunction`, matching what upstream
already does for `object_counting` and GSM8K. This is a disclosed deviation
from upstream's default behavior for the other BBH tasks, chosen because
judge variance would confound an already-small 100-example val split.

MMLU/GPQA (`textgrad_repro/textgrad/tasks/mmlu.py` and `gpqa.py`) expose no
`get_task_description()`, only raw HF splits with no train/val/test concept
of their own (GPQA has none at all; MMLU's `dev`/`validation`/`test` don't
line up with our 50/100/N convention) -- `_load_mmlu_all`/`_load_gpqa_main`
below wrap them with our own slicing and an explicit registry
`task_description`. AIME reuses no existing script in this repo: unlike
`/home/dg793/text-to-lora`, which has `scripts/gepa_repro_aime.py` with a
`load_aime_splits()` helper to import, this repo has no `gepa_repro/` port
at all, so the same split logic (`AI-MO/aimo-validation-aime` shuffled seed
0, halved into train/val; `MathArena/aime_2025` held out as test) is
reimplemented inline in `_load_aime_splits` below rather than imported.
`multiarith`/`aqua`/`commonsenseqa`/`strategyqa`/`trec` are plain HF-backed
loaders returning a shared `_RowsDataset` wrapper (duck-typed to the same
`__len__`/`__getitem__`/`get_task_description()` shape as `GSM8K_DSPy` and
`BigBenchHard`, which is all `tg.tasks.DataLoader` and this script's own
`dataset_to_rows()` need -- notably *not* upstream's abstract
`textgrad.tasks.base.Dataset`, which buys nothing here since nothing in the
optimizer loop does an `isinstance` check against it).

Deliberately not implemented: TruthfulQA (its generation-setting metric
needs a fine-tuned judge model this script has no path for), HotPotQA
(needs a retriever + multi-module pipeline, outside what a single
system-prompt `Variable` can express), LeetCode-Hard (a 3-tuple,
sandboxed-execution, solution-optimization benchmark in TextGrad's own
design, not prompt optimization), IFBench (needs a new constraint-verifier
dependency and a new non-answer-correctness metric type). See
`TEXTGRAD_MULTITASK_PLAN.md` section 6d.

GPQA (`Idavidrein/gpqa`) is a **gated** HF dataset -- the run needs
`HF_TOKEN` set to an account that has accepted its license, or loading it
raises a 401. (Confirmed set in this environment already.)

Example:
    uv run --with-editable ./textgrad_repro \\
        --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \\
        --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \\
        --with diskcache --with litellm --with graphviz --with gdown --with tenacity --with python-dotenv \\
        python scripts/textgrad_repro.py --model_dir Qwen/Qwen3-14B --no_enable_thinking --eval_test

Note on the pins above: newer vllm releases (confirmed through at least
0.21.0, and PyPI's latest 0.26.0) ship a compiled `_C` extension linked
against CUDA 13 regardless of declared metadata deps, which fails to
import on a machine whose driver caps out at CUDA 12.8
(`ImportError: libcudart.so.13`). vllm==0.11.0 (this repo's own declared
floor for the `gen` extra) pulls cu12 native packages instead and works,
but its tokenizer code needs transformers<5 (calls an attribute removed
in transformers 5.x); transformers<5 in turn needs huggingface_hub<1.0,
which conflicts with the persistent venv's kernels>=0.4,<0.16.0 (resolved
against huggingface_hub>=1.0 during the normal `uv sync --extra attn`) --
hence the additional kernels==0.10.0 pin, which is happy with an older
huggingface_hub. None of this touches the persistent project dependencies
(transformers>=5.0 stays the floor for everything else in this repo); it
is entirely contained in this script's ephemeral `--with` overlay. The
explicit cu128 index + unsafe-best-match is still needed for torch itself:
without it uv resolves a cu130-tagged torch wheel that imports fine but
raises `RuntimeError: The NVIDIA driver ... is too old` on first CUDA
call. See textgrad_repro_README.md.
"""

import argparse
import json
import os
import random
import re

import numpy as np
import textgrad as tg
from textgrad.autograd import llm_ops, string_based_ops
from textgrad.autograd.string_based_ops import StringBasedFunction
from textgrad.engine.vllm import ChatVLLM
from textgrad.tasks.big_bench_hard import BigBenchHard
from textgrad.tasks.gsm8k import GSM8K_DSPy
from vllm import SamplingParams

METHOD_NAME = "textgrad-repro"


def _parse_mcq_letter(text):
    """Handles the three surface forms actually produced across tasks: BBH's
    parenthesized `(A)`, MMLU/GPQA's `Answer: A` (upstream's own
    `eval_string_based` regex, which we extend to A-E for AQuA/CommonsenseQA's
    five options instead of upstream's hardcoded A-D), and a bare trailing
    letter. Takes the *last* match of whichever form is present, matching
    parse_integer_answer's own last-token convention."""
    matches = re.findall(r"(?i)Answer\s*:\s*([A-Za-z])", text)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\(([A-Za-z])\)", text)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-Za-z])\b", text.strip())
    if matches:
        return matches[-1].upper()
    return None


def _parse_exact(text):
    text = text.strip()
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    text = text.strip().lower().rstrip(".").strip()
    return text if text else None


def _parse_integer(text):
    matched = [token for token in text.strip().split() if any(c.isdigit() for c in token)]
    if not matched:
        return None
    digits = "".join(c for c in matched[-1].split(".")[0] if c.isdigit())
    if not digits:
        return None
    # A degenerate/repetitive generation can produce a "number" thousands of
    # digits long, which trips CPython's int-string conversion limit
    # (sys.set_int_max_str_digits, default 4300) and crashes the whole run.
    # Treat it as an unparseable (i.e. wrong) answer instead of raising.
    if len(digits) > 4300:
        return None
    return int(digits)


ANSWER_PARSERS = {
    "integer": _parse_integer,
    "mcq_letter": _parse_mcq_letter,
    "exact": _parse_exact,
}


# Fixed seed for deterministic pool slicing/sampling (GPQA's shuffle, MMLU's
# test sample, AQuA's train sample). Deliberately NOT args.seed: --seed is
# meant to vary across repeat runs of the *same* task for averaging, but the
# splits themselves must stay identical across those repeats or "the same
# task" would silently mean different data each time.
_SPLIT_SAMPLE_SEED = 20260803

# Matches upstream's own load_math_dataset seed (see docstring) -- kept as
# a separate constant so it's clear this one is inherited, not chosen fresh.
_AIME_SPLIT_SEED = 0


class _RowsDataset:
    """Duck-typed textgrad Dataset: __len__/__getitem__ returning
    (question_prompt, answer) 2-tuples and get_task_description(), the only
    interface tg.tasks.DataLoader and this script's dataset_to_rows() need.
    Backs every task built from a plain Python list of rows rather than one
    of upstream's own Dataset subclasses (GSM8K_DSPy, BigBenchHard, MMLU,
    GPQA) -- see the module docstring."""

    def __init__(self, rows, task_description):
        self._rows = rows
        self._task_description = task_description

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def get_task_description(self):
        return self._task_description


def _reasoning_task_description(value_desc):
    """Builds a BBH-style task description with an explicit VALUE meaning,
    for every task whose gold answer isn't a bare number -- BBH's own
    default ("VALUE is a numerical value") is wrong for MCQ/word-list/
    yes-no/etc. answers and, left in place, would mislead the seed prompt
    that TextGrad optimizes against. See TEXTGRAD_MULTITASK_PLAN.md section
    5/6a."""
    return (
        "You will answer a reasoning question. Think step by step. The last line of "
        f"your response should be of the following format: 'Answer: $VALUE' where VALUE {value_desc}."
    )


def _load_mmlu_all(split):
    from textgrad.tasks.mmlu import MMLU

    hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
    dataset = MMLU(subset="all", split=hf_split)
    rows = [dataset[i] for i in range(len(dataset))]
    if split == "train":
        rows = rows[:50]
    elif split == "val":
        rows = rows[:100]
    else:
        rows = random.Random(_SPLIT_SAMPLE_SEED).sample(rows, min(300, len(rows)))
    return _RowsDataset(rows, task_description=None)


def _load_gpqa_main(split):
    from textgrad.tasks.gpqa import GPQA

    dataset = GPQA(subset="gpqa_main")
    indices = list(range(len(dataset)))
    random.Random(_SPLIT_SAMPLE_SEED).shuffle(indices)
    start, stop = {"train": (0, 50), "val": (50, 150), "test": (150, len(indices))}[split]
    rows = [dataset[i] for i in indices[start:stop]]
    return _RowsDataset(rows, task_description=None)


_aime_cache = {}


def _load_aime_splits():
    """Same split as upstream's own `examples/aime_math/utils.py::load_math_dataset`
    (and this repo's sibling script family's `gepa_repro_aime.py:load_aime_splits`
    in /home/dg793/text-to-lora, which this repo has no port of -- see module
    docstring): AI-MO/aimo-validation-aime shuffled seed 0, split in half into
    train/val; MathArena/aime_2025 (full) held out as test."""
    from datasets import load_dataset

    train_val_ds = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    train_val_rows = [(item["problem"], int(item["answer"])) for item in train_val_ds]
    random.Random(_AIME_SPLIT_SEED).shuffle(train_val_rows)

    test_ds = load_dataset("MathArena/aime_2025", "default", split="train")
    test_rows = [(item["problem"], int(item["answer"])) for item in test_ds]

    half = len(train_val_rows) // 2
    return {"train": train_val_rows[:half], "val": train_val_rows[half:], "test": test_rows}


def _load_aime(split):
    if not _aime_cache:
        _aime_cache.update(_load_aime_splits())
    return _RowsDataset(_aime_cache[split], task_description=None)


def _load_multiarith(split):
    from datasets import load_dataset

    if split == "test":
        rows = [
            (r["question"], str(r["final_ans"])) for r in load_dataset("ChilleD/MultiArith", split="test")
        ]
    else:
        all_rows = [
            (r["question"], str(r["final_ans"])) for r in load_dataset("ChilleD/MultiArith", split="train")
        ]
        rows = all_rows[:50] if split == "train" else all_rows[50:150]
    return _RowsDataset(rows, task_description=None)


def _format_aqua_option(option):
    # Raw options are e.g. "A)21" -- render as "A) 21" prompt lines.
    if ")" in option:
        letter, rest = option.split(")", 1)
        return f"{letter}) {rest.strip()}"
    return option


def _aqua_row(row):
    options_str = "\n".join(_format_aqua_option(o) for o in row["options"])
    return f"{row['question']}\n{options_str}", row["correct"]


def _load_aqua(split):
    from datasets import load_dataset

    if split == "train":
        dataset = load_dataset("deepmind/aqua_rat", "raw", split="train")
        indices = random.Random(_SPLIT_SAMPLE_SEED).sample(range(len(dataset)), 50)
        rows = [_aqua_row(dataset[i]) for i in indices]
    elif split == "val":
        rows = [_aqua_row(r) for r in load_dataset("deepmind/aqua_rat", "raw", split="validation")][:100]
    else:
        rows = [_aqua_row(r) for r in load_dataset("deepmind/aqua_rat", "raw", split="test")]
    return _RowsDataset(rows, task_description=None)


def _commonsenseqa_row(row):
    choices_str = "\n".join(
        f"{label}) {text}"
        for label, text in zip(row["choices"]["label"], row["choices"]["text"], strict=True)
    )
    return f"{row['question']}\n{choices_str}", row["answerKey"]


def _load_commonsenseqa(split):
    from datasets import load_dataset

    if split == "train":
        rows = [_commonsenseqa_row(r) for r in load_dataset("tau/commonsense_qa", split="train")][:50]
    else:
        all_rows = [_commonsenseqa_row(r) for r in load_dataset("tau/commonsense_qa", split="validation")]
        rows = all_rows[:100] if split == "val" else all_rows[100:400]
    return _RowsDataset(rows, task_description=None)


def _load_strategyqa(split):
    from datasets import load_dataset

    def row(r):
        return r["question"], ("True" if r["answer"] else "False")

    if split == "test":
        rows = [row(r) for r in load_dataset("ChilleD/StrategyQA", split="test")][:300]
    else:
        all_rows = [row(r) for r in load_dataset("ChilleD/StrategyQA", split="train")]
        rows = all_rows[:50] if split == "train" else all_rows[50:150]
    return _RowsDataset(rows, task_description=None)


def _load_trec(split):
    from datasets import load_dataset

    def row(r):
        return r["text"], r["label_coarse_original"]

    if split == "test":
        rows = [row(r) for r in load_dataset("SetFit/TREC-QC", split="test")]
    else:
        all_rows = [row(r) for r in load_dataset("SetFit/TREC-QC", split="train")]
        rows = all_rows[:50] if split == "train" else all_rows[50:150]
    return _RowsDataset(rows, task_description=None)


def _make_equality_fn(parse):
    """Builds a `string_based_equality_fn`-shaped scorer (same signature/return
    type as textgrad.tasks.big_bench_hard.string_based_equality_fn) around one
    of the ANSWER_PARSERS entries, so the training-loss signal uses the same
    parser as eval_split()'s accuracy, symmetrically on prediction and gold,
    with a None/None non-match (see the module docstring's parser trap)."""

    def equality_fn(prediction, ground_truth_answer):
        predicted = parse(str(prediction.value))
        gold = parse(str(ground_truth_answer.value))
        return int(predicted is not None and predicted == gold)

    return equality_fn


# (task_key) -> how to build splits, describe the task, and score answers.
# See TEXTGRAD_MULTITASK_PLAN.md sections 3, 6a-6c for the reasoning behind
# each entry's parser/task_description choice.
TASKS = {
    "gsm8k": dict(
        loader=lambda split: GSM8K_DSPy(split=split),
        parser="integer",
        role_noun="GSM8K math word problem task",
        eval_purpose=(
            "Checks whether the predicted final numeric answer matches the gold answer "
            "for a GSM8K grade-school math word problem."
        ),
        task_description=None,
    ),
}

# BBH tasks with a non-empty 250-row pool (excludes e.g. penguins_in_a_table,
# 146 rows -> empty test split; see TEXTGRAD_MULTITASK_PLAN.md section 8).
# (task suffix, BBH task_name, parser, VALUE description or None -> use
# BBH's own numeric-answer default description).
_BBH_TASKS = [
    ("object_counting", "object_counting", "integer", None),
    ("word_sorting", "word_sorting", "exact", "is the sorted list of words"),
    ("multistep_arithmetic_two", "multistep_arithmetic_two", "integer", None),
    (
        "dyck_languages",
        "dyck_languages",
        "exact",
        "is the sequence of closing brackets that completes the expression",
    ),
    ("navigate", "navigate", "exact", "is Yes or No"),
    ("boolean_expressions", "boolean_expressions", "exact", "is True or False"),
    ("causal_judgement", "causal_judgement", "exact", "is Yes or No"),
    ("formal_fallacies", "formal_fallacies", "exact", "is valid or invalid"),
    ("sports_understanding", "sports_understanding", "exact", "is yes or no"),
    ("web_of_lies", "web_of_lies", "exact", "is Yes or No"),
    ("date_understanding", "date_understanding", "mcq_letter", "is the letter of the correct option"),
    ("temporal_sequences", "temporal_sequences", "mcq_letter", "is the letter of the correct option"),
    (
        "logical_deduction_seven_objects",
        "logical_deduction_seven_objects",
        "mcq_letter",
        "is the letter of the correct option",
    ),
    (
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_seven_objects",
        "mcq_letter",
        "is the letter of the correct option",
    ),
    ("geometric_shapes", "geometric_shapes", "mcq_letter", "is the letter of the correct option"),
    (
        "salient_translation_error_detection",
        "salient_translation_error_detection",
        "mcq_letter",
        "is the letter of the correct option",
    ),
    ("hyperbaton", "hyperbaton", "mcq_letter", "is the letter of the correct option"),
    (
        "movie_recommendation",
        "movie_recommendation",
        "mcq_letter",
        "is the letter of the correct option",
    ),  # OPRO-reported
    ("ruin_names", "ruin_names", "mcq_letter", "is the letter of the correct option"),  # OPRO-reported
    (
        "snarks",
        "snarks",
        "mcq_letter",
        "is the letter of the correct option",
    ),  # OPRO-reported; only 178 examples
]

for _suffix, _bbh_name, _parser, _value_desc in _BBH_TASKS:

    def _make_bbh_loader(name):
        return lambda split: BigBenchHard(name, split=split)

    TASKS[f"bbh_{_suffix}"] = dict(
        loader=_make_bbh_loader(_bbh_name),
        parser=_parser,
        role_noun=f"BIG-Bench Hard {_suffix.replace('_', ' ')} task",
        eval_purpose=(
            "Checks whether the predicted answer matches the gold answer for a BIG-Bench "
            f"Hard {_suffix.replace('_', ' ')} question."
        ),
        task_description=_reasoning_task_description(_value_desc) if _value_desc else None,
    )

# Extensions beyond the paper's own prompt-optimization benchmarks (Object
# Counting / Word Sorting / GSM8K) -- see module docstring.
TASKS["mmlu_all"] = dict(
    loader=_load_mmlu_all,
    parser="mcq_letter",
    role_noun="MMLU multiple-choice question-answering task",
    eval_purpose=(
        "Checks whether the predicted answer letter matches the gold answer letter for an "
        "MMLU multiple-choice question."
    ),
    task_description=_reasoning_task_description("is the letter of the correct option"),
)
TASKS["gpqa_main"] = dict(
    loader=_load_gpqa_main,
    parser="mcq_letter",
    role_noun="GPQA graduate-level multiple-choice science question task",
    eval_purpose=(
        "Checks whether the predicted answer letter matches the gold answer letter for a "
        "GPQA graduate-level science question."
    ),
    task_description=_reasoning_task_description("is the letter of the correct option"),
)
TASKS["aime"] = dict(
    loader=_load_aime,
    parser="integer",
    role_noun="AIME competition math task",
    eval_purpose=(
        "Checks whether the predicted final integer answer matches the gold answer for an "
        "AIME competition math problem."
    ),
    task_description=(
        "Solve the competition math problem carefully. Think step by step. The last line "
        "of your response should be of the following format: 'Answer: $VALUE' where VALUE "
        "is the integer answer (0-999)."
    ),
    # See TEXTGRAD_MULTITASK_PLAN.md section 8: a thinking model's reasoning
    # plus a competition-math solution blows past the 2000-token default and
    # truncates before the answer line, scoring 0 and looking like a
    # capability failure rather than a budget problem.
    max_tokens=16000,
    min_max_model_len=32768,
    # TextGrad-only (gepa_repro.py never reads this key -- it has its own
    # separate --reflection_max_tokens for the analogous rewrite call).
    # Matches max_tokens above rather than the --optimizer_max_tokens
    # default of 8000: the optimizer.step() rewrite call concatenates the
    # current prompt + 3 gradients (themselves generated at this task's
    # bumped forward budget) and, with --enable_thinking's TextGrad-side
    # default of True, needs headroom for a <think> block before reaching
    # the closing tag -- see _OptimizerEngineProxy's docstring for the
    # (simpler-task) incident this same failure mode already caused once.
    optimizer_max_tokens=16000,
)

# New Dataset loaders (~15-25 lines each above, via the shared _RowsDataset
# wrapper) for literature comparability with OPRO/EvoPrompt/Promptbreeder.
TASKS["multiarith"] = dict(
    loader=_load_multiarith,
    parser="integer",
    role_noun="MultiArith arithmetic word problem task",
    eval_purpose=(
        "Checks whether the predicted final numeric answer matches the gold answer for a "
        "MultiArith arithmetic word problem."
    ),
    task_description=_reasoning_task_description("is a numerical value"),
)
TASKS["aqua"] = dict(
    loader=_load_aqua,
    parser="mcq_letter",
    role_noun="AQuA-RAT algebraic word problem task",
    eval_purpose=(
        "Checks whether the predicted answer letter matches the gold answer letter for an "
        "AQuA-RAT algebraic word problem."
    ),
    task_description=_reasoning_task_description("is the letter of the correct option"),
)
TASKS["commonsenseqa"] = dict(
    loader=_load_commonsenseqa,
    parser="mcq_letter",
    role_noun="CommonsenseQA multiple-choice question task",
    eval_purpose=(
        "Checks whether the predicted answer letter matches the gold answer letter for a "
        "CommonsenseQA question."
    ),
    task_description=_reasoning_task_description("is the letter of the correct option"),
)
TASKS["strategyqa"] = dict(
    loader=_load_strategyqa,
    parser="exact",
    role_noun="StrategyQA multi-hop yes/no reasoning task",
    eval_purpose=(
        "Checks whether the predicted True/False answer matches the gold answer for a StrategyQA question."
    ),
    task_description=_reasoning_task_description("is True or False"),
)
TASKS["trec"] = dict(
    loader=_load_trec,
    parser="exact",
    role_noun="TREC question classification task",
    eval_purpose=(
        "Checks whether the predicted coarse category label matches the gold label for a TREC question."
    ),
    task_description=_reasoning_task_description(
        "is the coarse category label (one of ABBR, DESC, ENTY, HUM, LOC, NUM)"
    ),
)

CONCISE_GRADIENT_INSTRUCTION = (
    "\n\nKeep your feedback concise: at most 3-4 sentences (roughly 100 words) "
    "that identify the single most impactful failure mode and the single most "
    "promising fix. Do not enumerate multiple alternative suggestions, do not "
    "use headers, bullet lists, or a summary section -- write plain prose only."
)


def _patch_chat_vllm_engine(enable_thinking, default_max_tokens=2000):
    """Monkeypatch ChatVLLM.generate to pass add_generation_prompt=True (and,
    new in this port, enable_thinking=<flag>).

    Upstream's own generate() omits add_generation_prompt (see
    textgrad_repro/textgrad/engine/vllm.py), which the original text-to-lora
    repro originally matched byte-for-byte -- a full run there crashed at
    iteration 2 because the model failed to recover the missing generation-
    prompt cue under the optimizer step's large concatenated context and
    returned empty output. Patched here (not in the vendored clone) so
    textgrad_repro/ stays an untouched reference copy.

    `default_max_tokens` overrides the 2000-token default for tasks whose
    registry entry sets `max_tokens` (currently just `aime` -- see the
    TASKS registry and TEXTGRAD_MULTITASK_PLAN.md section 8): this is what
    `tg.BlackboxLLM`'s forward pass actually calls (via `LLMCall.forward`,
    which never threads a `max_tokens` kwarg through), so it's the only
    place a per-task forward budget can be applied without patching
    upstream's own call sites.
    """

    def generate(self, prompt, system_prompt=None, temperature=0, max_tokens=default_max_tokens, top_p=0.99):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        # Cache key must include enable_thinking: it's not part of
        # sys_prompt_arg/prompt, so without this prefix, replaying an
        # identical (system_prompt, prompt) pair under a different
        # enable_thinking setting silently returns a stale response
        # generated under the other setting (bit us across the
        # thinking-on -> thinking-off migration -- a handful of stale
        # thinking-enabled completions leaked into "thinking-disabled"
        # reruns that shared ~/.cache/textgrad/ with the old runs).
        cache_key = f"[enable_thinking={enable_thinking}]" + sys_prompt_arg + prompt
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none
        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]
        conversation += [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1)
        response = self.client.generate([chat_str], sampling_params)
        response = response[0].outputs[0].text
        self._save_cache(cache_key, response)
        return response

    ChatVLLM.generate = generate


def _patch_backward_system_prompt():
    """Append a conciseness constraint to BACKWARD_SYSTEM_PROMPT.

    Upstream's prompt (textgrad_repro/textgrad/autograd/llm_backward_prompts.py)
    has no length guidance; reasoning models can produce multi-thousand-
    character, multi-section critique essays per example. Concatenating a
    handful of these into one optimizer-step prompt is what pushed context
    large enough to trigger an empty-response crash in the original
    text-to-lora run. `BACKWARD_SYSTEM_PROMPT` is imported by value into each
    op module, so each module's copy is patched individually.
    """
    concise_prompt = llm_ops.BACKWARD_SYSTEM_PROMPT + CONCISE_GRADIENT_INSTRUCTION
    llm_ops.BACKWARD_SYSTEM_PROMPT = concise_prompt
    string_based_ops.BACKWARD_SYSTEM_PROMPT = concise_prompt


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_dir_name(model_dir, task_key):
    return f"{slugify(model_dir)}_{task_key}_{METHOD_NAME}"


def dataset_to_rows(dataset):
    rows = []
    for i in range(len(dataset)):
        question_prompt, answer = dataset[i]
        rows.append({"question_prompt": question_prompt, "answer": answer})
    return rows


def _json_default(obj):
    """json.dumps default= hook: BigBenchHard is pandas-backed, so integer
    gold answers (object_counting, multistep_arithmetic_two, ...) come back
    as numpy scalars, not native Python types -- json.dumps rejects those
    outright. Covers every JSON write site in this script (append_jsonl and
    the two manual train/val set dumps) so no per-task-loader workaround is
    needed."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def build_chat_prompt(tokenizer, system_prompt, user_prompt, enable_thinking):
    """Matches the (now-patched) textgrad.engine.vllm.ChatVLLM.generate's
    template construction -- add_generation_prompt=True, enable_thinking=
    <flag>, see _patch_chat_vllm_engine -- so batched calls here produce
    prompts identical to their own single-item generate()."""
    conversation = []
    if system_prompt:
        conversation = [{"role": "system", "content": system_prompt}]
    conversation += [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def batched_generate(
    engine, system_prompt, user_prompts, enable_thinking, temperature=0, max_tokens=2000, top_p=0.99
):
    if not user_prompts:
        return []
    chat_strs = [build_chat_prompt(engine.tokenizer, system_prompt, p, enable_thinking) for p in user_prompts]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1)
    outputs = engine.client.generate(chat_strs, sampling_params)
    return [o.outputs[0].text for o in outputs]


class _OptimizerEngineProxy:
    """Wraps the shared engine so `TextualGradientDescent.step()` gets a
    larger `max_tokens` budget than upstream's hardcoded generate() default
    of 2000 (textgrad_repro/textgrad/engine/vllm.py). The optimizer prompt
    concatenates every sampled example's full conversation + gradient text
    (batch_size=3 -> up to ~15K chars) into one rewrite call, and a
    reasoning model's default-on <think> block eats into that budget before
    any content is written -- with only 2000 tokens this can exhaust the
    budget before the model reaches the closing `<IMPROVED_VARIABLE>` tag,
    making the response unparseable (see the text-to-lora original's
    incident writeup)."""

    def __init__(self, engine, max_tokens):
        self._engine = engine
        self._max_tokens = max_tokens

    def __call__(self, prompt, **kwargs):
        kwargs.setdefault("max_tokens", self._max_tokens)
        return self._engine.generate(prompt, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)


def eval_split(
    engine,
    system_prompt_value,
    rows,
    iteration,
    split_name,
    forward_outputs_path,
    enable_thinking,
    parse,
    max_tokens=2000,
):
    responses = batched_generate(
        engine,
        system_prompt_value,
        [r["question_prompt"] for r in rows],
        enable_thinking,
        max_tokens=max_tokens,
    )
    out_rows = []
    n_correct = 0
    for r, response in zip(rows, responses, strict=True):
        predicted = parse(response)
        gold = parse(str(r["answer"]))
        correct = predicted is not None and predicted == gold
        n_correct += int(correct)
        out_rows.append(
            {
                "iteration": iteration,
                "split": split_name,
                "prompt": system_prompt_value,
                "question": r["question_prompt"],
                "gold_answer": r["answer"],
                "model_response": response,
                "predicted_answer": predicted,
                "correct": correct,
            }
        )
    append_jsonl(forward_outputs_path, out_rows)
    accuracy = n_correct / len(rows) if rows else 0.0
    return accuracy, out_rows


def main(args):
    set_seed(args.seed)

    spec = TASKS[args.task]
    parse = ANSWER_PARSERS[spec["parser"]]
    forward_max_tokens = spec.get("max_tokens", 2000)
    _patch_chat_vllm_engine(args.enable_thinking, default_max_tokens=forward_max_tokens)
    _patch_backward_system_prompt()

    max_model_len = max(args.max_model_len, spec.get("min_max_model_len") or 0)
    if max_model_len != args.max_model_len:
        print(
            f"bumping max_model_len {args.max_model_len} -> {max_model_len} for task {args.task!r} "
            "(see TASKS registry's min_max_model_len)"
        )

    data_dir = args.data_dir or os.path.join("data/textgrad_repro", run_dir_name(args.model_dir, args.task))
    os.makedirs(data_dir, exist_ok=True)
    print(f"writing artifacts to {data_dir}")
    train_set_path = os.path.join(data_dir, "train_set.jsonl")
    val_set_path = os.path.join(data_dir, "val_set.jsonl")
    forward_outputs_path = os.path.join(data_dir, "forward_outputs.jsonl")
    gradients_path = os.path.join(data_dir, "gradients.jsonl")
    iterations_path = os.path.join(data_dir, "iterations.jsonl")
    best_prompt_path = os.path.join(data_dir, "best_prompt.json")
    for path in (forward_outputs_path, gradients_path, iterations_path):
        open(path, "w").close()

    engine = ChatVLLM(
        args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=args.seed,
    )
    tg.set_backward_engine(engine, override=True)

    train_set = spec["loader"]("train")
    val_set = spec["loader"]("val")
    test_set = spec["loader"]("test")
    print(f"train/val/test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    assert len(test_set) > 0, (
        f"empty test split for task {args.task!r} -- see plan section 8 (short BBH tasks)"
    )

    train_rows_dump = dataset_to_rows(train_set)
    val_rows = dataset_to_rows(val_set)
    with open(train_set_path, "w") as f:
        for row in train_rows_dump:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(val_set_path, "w") as f:
        for row in val_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")

    baseline_prompt = spec["task_description"] or train_set.get_task_description()
    system_prompt = tg.Variable(
        baseline_prompt,
        requires_grad=True,
        role_description=(
            f"structured system prompt to a language model that specifies the behavior "
            f"and strategies for the {spec['role_noun']}"
        ),
    )
    model = tg.BlackboxLLM(engine, system_prompt)
    optimizer_max_tokens = spec.get("optimizer_max_tokens", args.optimizer_max_tokens)
    optimizer_engine = _OptimizerEngineProxy(engine, optimizer_max_tokens)
    optimizer = tg.TextualGradientDescent(engine=optimizer_engine, parameters=[system_prompt])
    eval_fn = StringBasedFunction(_make_equality_fn(parse), function_purpose=spec["eval_purpose"])

    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_iter = iter(train_loader)

    baseline_accuracy, _ = eval_split(
        engine,
        system_prompt.value,
        val_rows,
        -1,
        "val",
        forward_outputs_path,
        args.enable_thinking,
        parse,
        max_tokens=forward_max_tokens,
    )
    print(f"baseline (pre-training) val_accuracy={baseline_accuracy:.4f}")
    best = {"prompt": system_prompt.value, "val_accuracy": baseline_accuracy, "iteration": -1}
    previous_val_accuracy = baseline_accuracy

    total_iterations = args.max_epochs * args.steps_per_epoch
    for iteration in range(total_iterations):
        prompt_before = system_prompt.value

        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        optimizer.zero_grad()
        losses, batch_examples, train_forward_rows = [], [], []
        for x_text, y_text in zip(batch_x, batch_y, strict=True):
            x = tg.Variable(x_text, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(str(y_text), requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            losses.append(eval_output_variable)
            batch_examples.append((x_text, y_text, response))
            predicted = parse(response.value)
            gold = parse(str(y_text))
            train_forward_rows.append(
                {
                    "iteration": iteration,
                    "split": "train",
                    "prompt": prompt_before,
                    "question": x_text,
                    "gold_answer": y_text,
                    "model_response": response.value,
                    "predicted_answer": predicted,
                    "correct": predicted is not None and predicted == gold,
                }
            )
        append_jsonl(forward_outputs_path, train_forward_rows)

        total_loss = tg.sum(losses)
        total_loss.backward()

        # Pair each accumulated gradient on system_prompt back to its source
        # example: system_prompt.gradients is an unordered set, but each
        # gradient's stored context string contains the exact question_prompt
        # text of the example that produced it (distinct per example in a
        # shuffled batch), so substring matching recovers the pairing.
        gradient_rows = []
        remaining_gradients = set(system_prompt.gradients)
        for x_text, y_text, response in batch_examples:
            matched = None
            for g in remaining_gradients:
                context = system_prompt.gradients_context.get(g)
                if context and x_text in context.get("context", ""):
                    matched = g
                    break
            if matched is None:
                continue
            remaining_gradients.discard(matched)
            gradient_rows.append(
                {
                    "iteration": iteration,
                    "question": x_text,
                    "model_response": response.value,
                    "gold_answer": y_text,
                    "textual_gradient": matched.value,
                }
            )
        append_jsonl(gradients_path, gradient_rows)

        optimizer.step()
        updated_prompt = system_prompt.value

        val_accuracy, _ = eval_split(
            engine,
            updated_prompt,
            val_rows,
            iteration,
            "val",
            forward_outputs_path,
            args.enable_thinking,
            parse,
            max_tokens=forward_max_tokens,
        )

        reverted = False
        if args.run_validation and val_accuracy < previous_val_accuracy:
            system_prompt.set_value(prompt_before)
            reverted = True
            val_accuracy = previous_val_accuracy
        final_prompt = system_prompt.value
        previous_val_accuracy = val_accuracy

        print(
            f"iteration {iteration}: val_accuracy={val_accuracy:.4f}"
            f"{' (reverted)' if reverted else ''} n_gradients={len(gradient_rows)}"
        )

        if val_accuracy >= best["val_accuracy"]:
            best = {"prompt": final_prompt, "val_accuracy": val_accuracy, "iteration": iteration}

        append_jsonl(
            iterations_path,
            [
                {
                    "iteration": iteration,
                    "prompt": prompt_before,
                    "val_accuracy": val_accuracy,
                    "n_correct": round(val_accuracy * len(val_rows)),
                    "n_total": len(val_rows),
                    "reverted": reverted,
                    "n_sampled_for_gradient": len(batch_examples),
                    "textual_gradients": [g["textual_gradient"] for g in gradient_rows],
                    "updated_prompt": final_prompt,
                }
            ],
        )

    print(f"best prompt (iteration {best['iteration']}, val_accuracy={best['val_accuracy']:.4f}):")
    print(best["prompt"])

    result = dict(best)
    result["task"] = args.task
    result["baseline_val_accuracy"] = baseline_accuracy
    if args.eval_test:
        test_rows = dataset_to_rows(test_set)
        print(f"evaluating best prompt on full {args.task} test split ({len(test_rows)} questions)")
        test_eval_path = os.path.join(data_dir, "test_eval.jsonl")
        open(test_eval_path, "w").close()
        test_accuracy, _ = eval_split(
            engine,
            best["prompt"],
            test_rows,
            -1,
            "test",
            test_eval_path,
            args.enable_thinking,
            parse,
            max_tokens=forward_max_tokens,
        )
        print(f"test_accuracy={test_accuracy:.4f} ({round(test_accuracy * len(test_rows))}/{len(test_rows)})")
        result["test_accuracy"] = test_accuracy

        # Un-optimized (pre-training) prompt on the same test split -- the
        # only way to tell whether TextGrad's val-set gains actually
        # transfer to held-out test, rather than just reading test_accuracy
        # above in isolation. Previously only done for gsm8k, via a manual
        # one-off run never wired into this script; folded in here so every
        # task's best_prompt.json carries both numbers from one run.
        print(f"evaluating baseline (pre-training) prompt on full {args.task} test split ({len(test_rows)} questions)")
        baseline_test_eval_path = os.path.join(data_dir, "baseline_test_eval.jsonl")
        open(baseline_test_eval_path, "w").close()
        baseline_test_accuracy, _ = eval_split(
            engine,
            baseline_prompt,
            test_rows,
            -1,
            "test",
            baseline_test_eval_path,
            args.enable_thinking,
            parse,
            max_tokens=forward_max_tokens,
        )
        print(
            f"baseline_test_accuracy={baseline_test_accuracy:.4f} "
            f"({round(baseline_test_accuracy * len(test_rows))}/{len(test_rows)})"
        )
        result["baseline_test_accuracy"] = baseline_test_accuracy

        baseline_test_accuracy_path = os.path.join(data_dir, "baseline_test_accuracy.json")
        with open(baseline_test_accuracy_path, "w") as f:
            json.dump(
                {
                    "prompt": baseline_prompt,
                    "test_accuracy": baseline_test_accuracy,
                    "n_correct": round(baseline_test_accuracy * len(test_rows)),
                    "n_total": len(test_rows),
                },
                f,
                indent=2,
                default=_json_default,
            )
        print(f"wrote {baseline_test_accuracy_path}")

    with open(best_prompt_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"wrote {best_prompt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="Qwen/Qwen3-14B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="needs headroom beyond a single Q&A: optimizer.step() packs every sampled example's full "
        "conversation + gradient text into one prompt, which easily exceeds 4096",
    )
    parser.add_argument("--batch_size", type=int, default=3, help="matches upstream's default")
    parser.add_argument(
        "--optimizer_max_tokens",
        type=int,
        default=8000,
        help="max_tokens for optimizer.step()'s rewrite call only (upstream's ChatVLLM.generate default of "
        "2000 is too small for batch_size=3 gradients + a thinking model's <think> block -- see "
        "_OptimizerEngineProxy)",
    )
    parser.add_argument("--max_epochs", type=int, default=3, help="matches upstream's default")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4,
        help="upstream hardcodes this via `if steps == 3: break`; default reproduces the same 12 total steps",
    )
    parser.add_argument(
        "--run_validation",
        dest="run_validation",
        action="store_true",
        default=True,
        help="revert to the previous prompt if val accuracy regresses (upstream's own run_validation_revert; "
        "upstream's own script default is False, we default True -- see textgrad_repro_README.md)",
    )
    parser.add_argument("--no_run_validation", dest="run_validation", action="store_false")
    parser.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="passed to apply_chat_template for every role (solve/backward/optimizer); matches Qwen3's own "
        "chat-template default. Not present in the original text-to-lora script (which never varied this).",
    )
    parser.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="gsm8k",
        help="which task's splits/prompt/parser to use -- see TASKS registry and TEXTGRAD_MULTITASK_PLAN.md",
    )
    parser.add_argument(
        "--data_dir", default=None, help="default: data/textgrad_repro/{model_dir}_{task}_textgrad-repro/"
    )
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
