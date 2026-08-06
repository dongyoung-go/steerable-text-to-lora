"""Generation-based downstream accuracy eval. See ``docs/04_downstream_eval.md``.

Six conditions, all scored via real greedy generation (never teacher forcing) over the exact
same Q-axis held-out rows ``validation.py`` scores by loss:

    base                 no LoRA, no steering instruction anywhere
    prompted             no LoRA, steering instruction injected into the *target's own*
                          prompt (system turn) -- a genuinely separate code path, since
                          ``data.formatting.format_example`` hard-enforces an empty
                          ``system_message`` (docs/03 §1) and must not be relaxed to build this
    oracle                task's own trained LoRA (canonicalized), no prompt instruction
    t2l_train_desc        hypernetwork-generated LoRA from the task's own training description
    t2l_other_task_desc   hypernetwork-generated LoRA from another task's description (control)
    t2l_gibberish_desc    hypernetwork-generated LoRA from a gibberish string (control)

Unlike ``validation.py`` (which redraws a description independently per row, matching how
training samples them), every condition here generates **one LoRA per (task, condition)** from
one fixed description and applies it to every held-out row -- "the LoRA" this document's
accuracy numbers describe is a single concrete artifact, not a per-row-resampled average.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from steerable_t2l.data.formatting import format_example
from steerable_t2l.data.gold_answers import gold_answer, load_gold_index
from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import Splits, resolve_q_holdout
from steerable_t2l.hooks import build_sites, lora_hooks
from steerable_t2l.hypernet import SteerableHyperLoRA
from steerable_t2l.oracle.canonicalize import load_and_canonicalize_oracle
from steerable_t2l.target_spec import TargetSpec
from steerable_t2l.validation import GIBBERISH_DESCS
from steerable_t2l.validation import build_condition_descs as _validation_pool

CONDITIONS = (
    "base",
    "prompted",
    "oracle",
    "t2l_train_desc",
    "t2l_other_task_desc",
    "t2l_gibberish_desc",
)

NA = "n/a"

# condition -> the validation.py condition name whose description-pool logic it reuses
# (docs/03 §4's train_descs/other_task_descs pools, unchanged here -- only *which* pool
# entry gets used differs: validation samples per-row, this module fixes one per task).
_POOL_CONDITION = {
    "prompted": "train_descs",
    "t2l_train_desc": "train_descs",
    "t2l_other_task_desc": "other_task_descs",
}


def parse_integer_answer(text: str) -> int | None:
    """The last digit-bearing token in ``text``, as an int -- or ``None`` on any parse failure.

    Adapted from ``scripts/textgrad_repro.py``'s own ``_parse_integer``, not the
    vendored ``textgrad.tasks.big_bench_hard.parse_integer_answer``: that one defaults to
    ``0`` on parse failure, which would spuriously "match" a genuinely-zero gold answer
    against a response that failed to produce any answer at all (docs/04 §6's flagged
    hazard). Returning ``None`` and requiring both sides to parse before comparing avoids it.
    """
    tokens = [tok for tok in text.strip().split() if any(c.isdigit() for c in tok)]
    if not tokens:
        return None
    digits = "".join(c for c in tokens[-1].split(".")[0] if c.isdigit())
    return int(digits) if digits else None


def parse_mcq_letter_answer(text: str) -> str | None:
    """The predicted multiple-choice letter (``A``-``Z``), or ``None`` on parse failure.

    Ported verbatim from ``scripts/textgrad_repro.py``'s ``_parse_mcq_letter`` (the
    sibling text-to-lora repo's textgrad reproduction, which is where the raw v2 dataset's
    ``forward_outputs.jsonl``/``correct`` labels were originally computed -- reusing its exact
    parser keeps this eval's scoring consistent with how "correct" was defined upstream).
    Tries, in order: ``Answer: X``, a parenthesized ``(X)``, then a bare standalone letter --
    each time taking the *last* match, matching ``parse_integer_answer``'s last-token convention.
    """
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


def parse_exact_answer(text: str) -> str | None:
    """Normalized free-text answer (lowercased, trailing period stripped, text after a
    trailing ``Answer:`` marker preferred if present), or ``None`` if empty.

    Ported verbatim from ``scripts/textgrad_repro.py``'s ``_parse_exact`` -- used for every
    gold-answer shape that's neither a bare integer nor a single MCQ letter: yes/no,
    true/false, valid/invalid, bracket sequences, sorted word lists, category labels, etc.
    """
    text = text.strip()
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    text = text.strip().lower().rstrip(".").strip()
    return text if text else None


ANSWER_PARSERS: dict[str, Callable[[str], object | None]] = {
    "integer": parse_integer_answer,
    "mcq_letter": parse_mcq_letter_answer,
    "exact": parse_exact_answer,
}

_INTEGER_RE = re.compile(r"^-?\d+$")
_MCQ_LETTER_RE = re.compile(r"^\(?[A-Za-z]\)?$")


def classify_answer_parser(gold_values: list[str]) -> str:
    """Which of ``ANSWER_PARSERS`` fits a task's gold answers, inferred from their raw shape
    -- no hardcoded per-task/per-domain table, so this keeps working unchanged as more
    domains get added to ``data/textgrad_repro/`` beyond today's 10 (gsm8k, aqua, 8x bbh_*),
    e.g. the wider task set already reproduced in ``scripts/textgrad_repro.py``/
    ``GEPA_REPRO_RESULTS.md`` (mmlu_all, gpqa_main, commonsenseqa, strategyqa, trec, aime,
    more bbh_* subtasks, ...). Only valid on a task's own *embedded* gold answers (already
    the bare final-answer form, e.g. ``"C"``, ``"(E)"``, ``"invalid"``, ``"2200"``) -- never
    call this on the legacy externally-joined GSM8K gold strings (full CoT solutions ending
    in ``"#### N"``), which don't fit any of these three shapes and must stay hardcoded to
    ``"integer"`` (``parse_integer_answer`` already finds the trailing number regardless).
    """
    # Majority vote (>=90%), not a strict all() -- a single data-quality outlier must not
    # collapse a whole task's parser choice. Confirmed real case: BBH movie_recommendation's
    # official test JSON has one row whose target is "Monsters, Inc" (a full movie title --
    # upstream split that title across two lettered options, (A) "Monsters" / (B) "Inc", and
    # kept the untouched title as the target) among 99 clean single-letter targets; requiring
    # literally every value to match previously forced the "exact" parser onto all 100 rows,
    # which can't match any free-text response and silently produced a false 0% across every
    # condition. The one outlier row still can't be parsed correctly under majority voting --
    # it just costs at most one wrong answer instead of the whole task.
    non_empty = [g.strip() for g in gold_values if g and g.strip()]
    if not non_empty:
        return "exact"
    n = len(non_empty)
    integer_frac = sum(1 for g in non_empty if _INTEGER_RE.match(g)) / n
    mcq_frac = sum(1 for g in non_empty if _MCQ_LETTER_RE.match(g)) / n
    if integer_frac >= 0.9:
        return "integer"
    if mcq_frac >= 0.9:
        return "mcq_letter"
    return "exact"


def condition_desc(
    task: Task,
    splits: Splits,
    all_tasks: list[Task],
    condition: str,
    rng: random.Random,
) -> str | None:
    """The single fixed steering description for ``(task, condition)``, or ``None`` if the
    condition needs no description (``base``/``oracle``) or none is available (e.g. no other
    non-held-out task exists for ``t2l_other_task_desc``).

    ``prompted``/``t2l_train_desc`` prefer ``task.metadata.best_description_index`` -- the
    textgrad-optimized instruction that actually produced this task's SFT training responses
    -- over the pool's first entry. ``descriptions[0]`` is just whichever prompt textgrad tried
    *first* (usually its unoptimized seed instruction, since descriptions are collected in
    first-appearance order, not accuracy order); scoring "the best available steering
    description" (docs/04 §1/§2's stated intent) against that seed prompt understates both
    conditions. Falls back to the pool's first entry when no best index is recorded (legacy
    tasks built before this field existed) or the best description is itself D-held-out.
    """
    if condition in ("base", "oracle"):
        return None
    if condition == "t2l_gibberish_desc":
        return GIBBERISH_DESCS[0]
    pool = _validation_pool(task, splits, all_tasks, _POOL_CONDITION[condition], rng)
    if not pool:
        return None
    if condition in ("prompted", "t2l_train_desc"):
        best_idx = task.metadata.best_description_index
        if best_idx is not None:
            best_desc = task.metadata.descriptions[best_idx]
            if best_desc in pool:
                return best_desc
    return pool[0]


def build_prompted_prompt(tokenizer, desc: str, user_content: str) -> str:
    """The ``prompted`` condition's prompt: ``desc`` as the system turn, falling back to
    prepending it to the user turn if the tokenizer's chat template has no system role (some
    templates silently drop an unsupported role instead of raising -- checked for explicitly,
    not just wrapped in a bare try/except). See docs/04 §5."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": desc}, {"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if desc.strip() and desc.strip() not in text:
        merged = f"{desc}\n\n{user_content}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}], tokenize=False, add_generation_prompt=True
        )
    return text


def _expand_per_module(
    per_module: dict[str, tuple[torch.Tensor, torch.Tensor]], bs: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``{module: (A [n_layers,r,in], B [n_layers,out,r])}`` -> the same, broadcast to a
    batch dimension of size ``bs`` (every row in the batch gets the identical LoRA)."""
    return {m: (A.unsqueeze(0).expand(bs, -1, -1, -1), B.unsqueeze(0).expand(bs, -1, -1, -1)) for m, (A, B) in per_module.items()}


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2560
    batch_size: int = 8


def generate_texts(
    target,
    tokenizer,
    prompts: list[str],
    spec: TargetSpec,
    per_module_fixed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    gen_config: GenerationConfig,
    device,
) -> list[str]:
    """Greedy-decode ``prompts`` (already chat-templated, generation-prompt included) in
    chunks of ``gen_config.batch_size``.

    ``per_module_fixed`` is un-batched (``[n_layers, ...]``, as ``oracle``/``t2l_*``
    conditions produce) or ``None`` for no LoRA at all (``base``/``prompted``). The same
    ``lora_hooks`` context wraps the *entire* ``model.generate`` call -- not one forward --
    so the injected LoRA is live for every incremental decode step, not just the prefill
    (docs/04 §4's ⚠️; verified by ``tests/test_eval_accuracy.py``'s hook-coverage check).
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), gen_config.batch_size):
            batch_prompts = prompts[start : start + gen_config.batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            bs = enc["input_ids"].shape[0]
            gen_kwargs = dict(
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            with torch.no_grad():
                if per_module_fixed is None:
                    out = target.generate(**enc, **gen_kwargs)
                else:
                    sites = build_sites(spec, _expand_per_module(per_module_fixed, bs))
                    with lora_hooks(target, sites, spec.scaling):
                        out = target.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = orig_padding_side
    return outputs


def eval_rows_for_task(task: Task, splits: Splits) -> list[dict]:
    """The Q-axis held-out rows of ``task``, in the same order/indices
    ``validation.run_validation`` scores -- via ``data.splits.resolve_q_holdout`` against the
    raw dataset directly (no tokenization needed here, unlike ``PerTaskDescDataset``, since
    accuracy scoring works from real text, not tokenized labels)."""
    import datasets as hf_datasets

    raw_rows = list(hf_datasets.load_dataset(**task.metadata.ds_kwargs))
    q_idx = resolve_q_holdout(splits, task.name, len(raw_rows))
    return [raw_rows[i] for i in q_idx]


def score_condition(
    hypernet: SteerableHyperLoRA | None,
    target,
    tokenizer,
    spec: TargetSpec,
    task: Task,
    splits: Splits,
    all_tasks: list[Task],
    condition: str,
    rows: list[dict],
    gold_index: dict[str, str],
    oracle_dir: str | Path | None,
    rng: random.Random,
    gen_config: GenerationConfig,
) -> dict | str:
    """Score one ``(task, condition)`` pair. Returns
    ``{"accuracy": float, "n": int, "n_correct": int}`` or ``"n/a"``."""
    device = next(target.parameters()).device

    per_module_fixed = None
    if condition == "oracle":
        if oracle_dir is None:
            return NA
        adapter_dir = Path(oracle_dir) / task.name
        if not adapter_dir.exists():
            return NA
        per_module_fixed = load_and_canonicalize_oracle(str(adapter_dir), spec, device=device)

    desc = condition_desc(task, splits, all_tasks, condition, rng)
    if condition in ("prompted", "t2l_train_desc", "t2l_other_task_desc", "t2l_gibberish_desc") and desc is None:
        return NA
    if condition.startswith("t2l_"):
        if hypernet is None:
            return NA
        per_module = hypernet.generate_for_batch([desc])
        per_module_fixed = {m: (A[0], B[0]) for m, (A, B) in per_module.items()}

    # Rows built by build_tasks_from_textgrad_repro_v2.py (or any future domain built the same
    # way) carry their own bare gold answer ("C", "(E)", "invalid", "2200", ...) straight from
    # forward_outputs.jsonl -- no external dataset join needed, and its raw shape is what
    # classify_answer_parser reads to pick integer/mcq_letter/exact. Legacy GSM8K tasks built
    # without that field fall back to the external HF gsm8k join (full CoT solutions ending in
    # "#### N") and always use the integer parser, matching this function's prior behavior
    # exactly -- classify_answer_parser must never see that shape (see its docstring).
    has_embedded_gold = all("gold_answer" in row for row in rows)
    if has_embedded_gold:
        gold_strs = [row["gold_answer"] for row in rows]
        parser_name = classify_answer_parser(gold_strs)
    else:
        gold_strs = [gold_answer(gold_index, row["question"]) for row in rows]
        parser_name = "integer"
    parse = ANSWER_PARSERS[parser_name]
    golds = [parse(g) for g in gold_strs]

    prompts: list[str] = []
    for row in rows:
        if condition == "prompted":
            user_content = task.metadata.user_prompt_template.format(**row)
            prompts.append(build_prompted_prompt(tokenizer, desc, user_content))
        else:
            prompt_text, _ = format_example(row, task.metadata, tokenizer)
            prompts.append(prompt_text)

    responses = generate_texts(target, tokenizer, prompts, spec, per_module_fixed, gen_config, device)

    n_correct = 0
    for response, gold in zip(responses, golds, strict=True):
        predicted = parse(response)
        n_correct += int(predicted is not None and gold is not None and predicted == gold)
    n = len(rows)
    return {"accuracy": (n_correct / n) if n else float("nan"), "n": n, "n_correct": n_correct}


def _macro_average(per_task: dict[str, dict], condition: str) -> float | str:
    """Unweighted mean of per-task accuracy -- deliberately *not* row-weighted (docs/04 §7:
    one large task must not dominate the headline number the way pooling rows would)."""
    values = [conds[condition]["accuracy"] for conds in per_task.values() if isinstance(conds.get(condition), dict)]
    return sum(values) / len(values) if values else NA


def _task_comparisons(conds: dict[str, dict | str]) -> dict[str, float] | str:
    t2l = conds.get("t2l_train_desc")
    base = conds.get("base")
    prompted = conds.get("prompted")
    oracle = conds.get("oracle")
    out: dict[str, float] = {}
    if isinstance(t2l, dict) and isinstance(base, dict):
        out["t2l_train_desc_minus_base"] = t2l["accuracy"] - base["accuracy"]
    if isinstance(t2l, dict) and isinstance(prompted, dict):
        out["t2l_train_desc_minus_prompted"] = t2l["accuracy"] - prompted["accuracy"]
    if isinstance(t2l, dict) and isinstance(oracle, dict) and oracle["accuracy"]:
        out["t2l_train_desc_over_oracle"] = t2l["accuracy"] / oracle["accuracy"]
    return out if out else NA


def compute_comparisons(per_task: dict[str, dict]) -> dict:
    """The three comparisons docs/04 §7 asks for, per-task and macro-averaged:
    ``t2l_train_desc - base``, ``t2l_train_desc - prompted``, ``t2l_train_desc / oracle``."""
    per_task_out = {name: _task_comparisons(conds) for name, conds in per_task.items()}
    keys = ("t2l_train_desc_minus_base", "t2l_train_desc_minus_prompted", "t2l_train_desc_over_oracle")
    macro = {}
    for key in keys:
        values = [entry[key] for entry in per_task_out.values() if isinstance(entry, dict) and key in entry]
        macro[key] = sum(values) / len(values) if values else NA
    return {"per_task": per_task_out, "macro": macro}


def run_downstream_eval(
    hypernet: SteerableHyperLoRA | None,
    target,
    tokenizer,
    spec: TargetSpec,
    tasks: list[Task],
    splits: Splits,
    *,
    oracle_dir: str | Path | None = None,
    gen_config: GenerationConfig | None = None,
    seed: int = 0,
    conditions: tuple[str, ...] = CONDITIONS,
    gold_index: dict[str, str] | None = None,
    on_condition_start: Callable[[str, str, int], None] | None = None,
    on_condition_done: Callable[[str, str, dict | str], None] | None = None,
    existing: dict[str, dict[str, dict | str]] | None = None,
    rows_for_task: Callable[[Task], list[dict]] | None = None,
) -> dict:
    """Iterate (task x condition) over every non-T-held-out task's Q-axis held-out rows.

    T-held-out tasks are out of scope here (unlike ``validation.run_validation``'s
    ``unseen_task_descs`` condition): docs/04's condition table has no zero-shot-task
    analogue, only the six conditions above, all scored against tasks the hypernetwork
    trained on.

    ``gold_index`` defaults to :func:`load_gold_index` (a real HF Hub / cache read); tests
    inject a small fake index directly to stay network-free.

    ``on_condition_start``/``on_condition_done`` are optional progress hooks -- each
    ``(task, condition)`` pair can run real greedy generation over dozens of rows at up to
    ``gen_config.max_new_tokens`` tokens each, easily minutes per pair; a caller with no
    per-pair signal has no way to distinguish "still working" from "hung". Called with
    ``(task_name, condition, n_rows)`` before and ``(task_name, condition, result)`` after.

    ``existing`` is a previous (possibly partial) ``per_task`` result -- e.g. loaded from an
    interrupted run's output JSON. Any ``(task, condition)`` pair already present there is
    reused verbatim instead of re-run (real generation is expensive; resuming should not
    redo completed work). Still fires both callbacks for a reused pair, so a caller logging
    progress sees it either way. Note: skipping a pair means ``rng`` isn't consumed for it,
    so ``t2l_other_task_desc``'s "which other task" choice for *later* tasks/conditions can
    differ from what a from-scratch run with the same ``seed`` would pick -- resuming trades
    exact reproducibility of that one choice for not re-running finished work.

    ``rows_for_task`` overrides the default row source (:func:`eval_rows_for_task`, the small
    Q-axis held-out split). Pass e.g. ``data.external_testsets.load_external_test_rows`` bound
    per-task to score against a domain's full official test set instead -- see
    ``scripts/eval_downstream_accuracy_full.py``. Only conditions that don't need a per-task
    trained artifact make sense with a swapped row source; ``oracle`` still scores against
    whatever rows this returns, so callers wanting a *bigger, cleaner, less noisy* number for
    the other five conditions should also drop ``"oracle"`` from ``conditions`` (the oracle
    LoRA was never trained to generalize past its own tiny training pool).
    """
    gen_config = gen_config or GenerationConfig()
    rng = random.Random(seed)
    t_holdout = set(splits.t_holdout)
    trained_tasks = [t for t in tasks if t.name not in t_holdout]
    existing = existing or {}
    rows_for_task = rows_for_task or (lambda task: eval_rows_for_task(task, splits))

    per_task: dict[str, dict[str, dict | str]] = {}
    for task in trained_tasks:
        rows = rows_for_task(task)
        if not rows:
            continue
        # load_gold_index() is a real HF Hub/cache read (network) -- only pay for it the first
        # time a task actually lacks the embedded "gold_answer" field build_tasks_from_
        # textgrad_repro_v2.py (and any future domain built the same way) writes; tasks that
        # all have it (v2+) never need the legacy GSM8K join at all.
        if gold_index is None and not all("gold_answer" in row for row in rows):
            gold_index = load_gold_index()
        per_task[task.name] = {}
        existing_conditions = existing.get(task.name, {})
        for condition in conditions:
            if condition in existing_conditions:
                result = existing_conditions[condition]
            else:
                if on_condition_start is not None:
                    on_condition_start(task.name, condition, len(rows))
                result = score_condition(
                    hypernet, target, tokenizer, spec, task, splits, tasks, condition,
                    rows, gold_index, oracle_dir, rng, gen_config,
                )
            per_task[task.name][condition] = result
            if on_condition_done is not None:
                on_condition_done(task.name, condition, result)

    overall = {condition: _macro_average(per_task, condition) for condition in conditions}
    comparisons = compute_comparisons(per_task)

    return {"per_task": per_task, "overall": overall, "comparisons": comparisons}
