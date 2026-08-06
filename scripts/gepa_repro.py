"""GEPA (https://github.com/gepa-ai/gepa, arXiv:2507.19457) reproduction,
using the real `gepa` package (cloned into `gepa_repro/`, installed as an
ephemeral `uv run --with-editable` overlay -- see `scripts/gepa_repro_run.sh`
/ `gepa_repro_README.md`), driving our own local Qwen3 vLLM engine for both
the task-solving and reflective-mutation roles. Not a reimplementation of
GEPA's optimizer: `optimize_anything()`, Pareto-based candidate
selection/acceptance, and the reflective-mutation prompt template are all
`gepa`'s own code.

Ported from `/home/dg793/text-to-lora/scripts/gepa_repro_gsm8k.py` +
`gepa_repro_aime.py` (two separate, near-duplicate scripts there) onto this
repo's `scripts/textgrad_repro.py` task registry: rather than one script per
task, `--task` selects among the same ~29-entry `TASKS` registry
`textgrad_repro.py` already built (GSM8K, 20 BBH tasks, MMLU, GPQA, AIME,
MultiArith, AQuA, CommonsenseQA, StrategyQA, TREC -- see
`TEXTGRAD_MULTITASK_PLAN.md`), imported directly (`from textgrad_repro import
...`, same pattern `scripts/textgrad_baseline_sweep.py` already uses) rather
than reimplemented, so the two optimizers (`textgrad_repro.py`'s TextGrad
loop and this GEPA loop) stay comparable on identical splits/prompts/parsers
for every task, and a new task added to one registry covers both scripts.
This also means every task's seed prompt is `TASKS[task]["task_description"]
or dataset.get_task_description()` -- the same value `textgrad_repro.py`
optimizes from -- rather than the original gsm8k/aime scripts' own
hand-written `SEED_PROMPT` strings.

Three changes from the original text-to-lora scripts, beyond the task
generalization above:
    - **Model**: default `--model_dir` is `Qwen/Qwen3-14B`, not
      `Qwen/Qwen3-32B` -- matching `textgrad_repro.py`'s own default and this
      repo's completed GSM8K regression baseline
      (`data/textgrad_repro/qwen-qwen3-14b_gsm8k_textgrad-repro/`), not
      text-to-lora's model choice.
    - **Thinking mode default flipped off**: `--enable_thinking` defaults to
      `False` here (`--enable_thinking` to turn it on), the opposite of the
      original `gepa_repro_aime.py`'s `default=True`. Applies uniformly to
      every task, not just AIME -- keeps GEPA runs directly comparable to
      `textgrad_repro.py --no_enable_thinking` runs on the same task, and
      keeps the reflection LM (always thinking-off, see
      `gepa_repro_common.VLLMLanguageModel`) and the solver on the same
      decoding regime by default.
    - **Environment**: runs inside the same `--with-editable ./textgrad_repro
      --with-editable ./gepa_repro ...` overlay `scripts/gepa_repro_run.sh`
      bakes in (this repo's `vllm==0.11.0`/`transformers==4.57.1`/
      `kernels==0.10.0` pins -- see `textgrad_repro_README.md`'s "why this is
      pinned much harder" section -- not text-to-lora's
      `transformers==4.51.1`/`vllm==0.9.2`).

Also dropped: the original `gepa_repro_gsm8k.py`'s
`hyper_llm_modulator.steering.textgrad_verifiers.verify_gsm8k_answer` (a
this-repo-doesn't-have dependency) -- GSM8K here scores with
`textgrad_repro.ANSWER_PARSERS["integer"]`, the same deterministic
last-numeric-token parser `textgrad_repro.py` itself uses for `gsm8k`, so
scoring stays apples-to-apples between the two optimizers rather than
apples-to-oranges between two different GSM8K graders.

What's unchanged from the original (see `gepa_repro_README.md` for the full
list, including the "known upstream quirks" and "known gaps" sections):
one shared in-process vLLM engine for solver + reflection roles (no
litellm/HTTP server); `batch_evaluator` instead of GEPA's default per-pair
thread pool; `test_set` scored manually (not via
`optimize_anything(test_set=...)`, which rejects the legacy `GEPAConfig`
object this script uses) rather than the new `OptimizeAnythingConfig` API;
the same JSONL output schema (`train_set.jsonl`, `val_set.jsonl`,
`forward_outputs.jsonl`, `gradients.jsonl`, `iterations.jsonl`,
`best_prompt.json`, `test_eval.jsonl`), reconstructed post-hoc from
`GEPAResult.candidates`/`.parents`/`.val_aggregate_scores`/
`.discovery_eval_counts`/`.val_subscores` the same way.

Example:
    uv run --with-editable ./textgrad_repro --with-editable ./gepa_repro \\
        --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \\
        --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \\
        --with diskcache --with litellm --with cloudpickle --with tqdm \\
        --with graphviz --with gdown --with tenacity --with python-dotenv \\
        python scripts/gepa_repro.py --model_dir Qwen/Qwen3-14B --task gsm8k --eval_test
"""

import argparse
import json
import os
import random

import numpy as np
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig
from gepa.optimize_anything import optimize_anything
from gepa.utils.stop_condition import NoImprovementStopper
from gepa_repro_common import VLLMLanguageModel, batched_generate, load_vllm_engine

from textgrad_repro import ANSWER_PARSERS, TASKS, _json_default, dataset_to_rows, slugify

METHOD_NAME = "gepa-repro"


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def run_dir_name(model_dir, task_key):
    return f"{slugify(model_dir)}_{task_key}_{METHOD_NAME}"


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def rows_with_ids(rows, prefix):
    """Adds GEPA's own recognized `"id"` field (see `oa/eval_server.py`'s
    `_resolve_id`) to `textgrad_repro.dataset_to_rows()`'s plain
    `{"question_prompt", "answer"}` dicts, so val/train subscores stay
    stably keyed across candidates -- same convention the original
    `gepa_repro_gsm8k.py`/`gepa_repro_aime.py` used for their own
    hand-built row dicts."""
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        row["id"] = f"{prefix}_{i}"
        out.append(row)
    return out


def main(args):
    set_seed(args.seed)

    spec = TASKS[args.task]
    parse = ANSWER_PARSERS[spec["parser"]]
    forward_max_tokens = spec.get("max_tokens") or args.max_tokens
    max_model_len = max(args.max_model_len, spec.get("min_max_model_len") or 0)
    if max_model_len != args.max_model_len:
        print(
            f"bumping max_model_len {args.max_model_len} -> {max_model_len} for task {args.task!r} "
            "(see textgrad_repro.TASKS registry's min_max_model_len)"
        )

    data_dir = args.data_dir or os.path.join("data/gepa_repro", run_dir_name(args.model_dir, args.task))
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

    llm, tokenizer = load_vllm_engine(
        args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=args.seed,
    )
    reflection_lm = VLLMLanguageModel(llm, tokenizer, temperature=0.7, max_tokens=args.reflection_max_tokens)

    train_set = spec["loader"]("train")
    val_set = spec["loader"]("val")
    test_set = spec["loader"]("test")
    print(f"train/val/test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    assert len(test_set) > 0, f"empty test split for task {args.task!r}"

    train_rows = rows_with_ids(dataset_to_rows(train_set), "train")
    val_rows = rows_with_ids(dataset_to_rows(val_set), "val")
    test_rows = rows_with_ids(dataset_to_rows(test_set), "test")
    with open(train_set_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(val_set_path, "w") as f:
        for row in val_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")

    seed_prompt = spec["task_description"] or train_set.get_task_description()

    call_counter = {"n": 0}

    def batch_evaluate(pairs):
        """One batched vLLM call per GEPA evaluation stage (minibatch, val,
        or test pass all arrive here in a single list, per GEPA's own
        `batch_evaluator` contract -- see `gepa_repro/src/gepa/oa/eval_server.py`).
        Also our hook point for `forward_outputs.jsonl`/`gradients.jsonl`
        logging, since GEPA owns the outer optimization loop, not us."""
        call_counter["n"] += 1
        iteration = call_counter["n"]
        candidates = [c for c, _ex in pairs]
        examples = [ex for _c, ex in pairs]
        # A GEPA evaluation stage always scores one shared candidate against
        # many examples (never many candidates at once), so this holds.
        candidate_text = candidates[0]
        responses = batched_generate(
            llm,
            tokenizer,
            candidate_text,
            [ex["question_prompt"] for ex in examples],
            max_tokens=forward_max_tokens,
            enable_thinking=args.enable_thinking,
        )
        results = []
        forward_rows = []
        gradient_rows = []
        for ex, response in zip(examples, responses, strict=True):
            predicted = parse(response)
            gold = parse(str(ex["answer"]))
            correct = predicted is not None and predicted == gold
            score = 1.0 if correct else 0.0
            if predicted is None:
                feedback = (
                    f"Could not parse a valid answer from the response. The gold answer is "
                    f"'{ex['answer']}'. Explain what went wrong and how the approach should change."
                )
            elif correct:
                feedback = f"Correct. The gold answer is '{ex['answer']}'."
            else:
                feedback = (
                    f"Incorrect. The gold answer is '{ex['answer']}'. "
                    "Explain what went wrong and how the approach should change."
                )
            side_info = {
                "id": ex["id"],
                "question": ex["question_prompt"],
                "response": response,
                "predicted_answer": predicted,
                "gold_answer": ex["answer"],
                "feedback": feedback,
            }
            results.append((score, side_info))
            forward_rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate_text,
                    "question": ex["question_prompt"],
                    "gold_answer": ex["answer"],
                    "model_response": response,
                    "predicted_answer": predicted,
                    "correct": correct,
                }
            )
            gradient_rows.append(
                {
                    "iteration": iteration,
                    "question": ex["question_prompt"],
                    "model_response": response,
                    "gold_answer": ex["answer"],
                    "textual_feedback": feedback,
                }
            )
        append_jsonl(forward_outputs_path, forward_rows)
        append_jsonl(gradients_path, gradient_rows)
        return results

    # Opt-in early-stop diagnostic: off by default so plain runs still match
    # the paper's protocol (every benchmark run there stops purely by
    # rollout-budget exhaustion -- see gepa_repro_README.md's hyperparameter
    # discussion). When set, this doesn't create headroom on a near-ceiling
    # task/model pairing -- it just stops burning budget once GEPA's val-set
    # best score has gone `no_improvement_patience` iterations without
    # improving, and flags the run as converged-early in best_prompt.json so
    # gepa_repro_run_all.sh's summary table can surface likely-ceiling tasks.
    stop_callbacks = None
    no_improvement_stopper = None
    if args.no_improvement_patience is not None:
        no_improvement_stopper = NoImprovementStopper(args.no_improvement_patience)
        stop_callbacks = no_improvement_stopper

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(data_dir, "gepa_run"),
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            parallel=False,
            seed=args.seed,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=args.batch_size,
        ),
        # `stop_callbacks` lives on GEPAConfig itself, not EngineConfig --
        # confirmed against gepa_repro/src/gepa/gepa_launcher.py (EngineConfig
        # has no such field; GEPAConfig.stop_callbacks: StopperProtocol |
        # Sequence[StopperProtocol] | None).
        stop_callbacks=stop_callbacks,
    )

    result = optimize_anything(
        seed_candidate=seed_prompt,
        batch_evaluator=batch_evaluate,
        dataset=train_rows,
        valset=val_rows,
        config=gepa_config,
    )

    # Reconstruct iterations.jsonl from the GEPAResult's own candidate
    # lineage/scores (candidates/parents/val_aggregate_scores/
    # discovery_eval_counts/val_subscores) -- GEPA owns the loop so this is
    # a post-hoc reconstruction, not something we log incrementally.
    str_key = result._str_candidate_key
    iteration_rows = []
    for idx, candidate in enumerate(result.candidates):
        candidate_text = candidate[str_key] if str_key else candidate
        parent_idxs = result.parents[idx]
        parent_text = (
            None
            if idx == 0 or not parent_idxs or parent_idxs[0] is None
            else (result.candidates[parent_idxs[0]][str_key] if str_key else result.candidates[parent_idxs[0]])
        )
        val_subscores = result.val_subscores[idx] if idx < len(result.val_subscores) else {}
        iteration_rows.append(
            {
                "iteration": idx,
                "candidate": candidate_text,
                "parent_candidate": parent_text,
                "val_aggregate_score": result.val_aggregate_scores[idx],
                "n_correct": round(result.val_aggregate_scores[idx] * len(val_subscores)),
                "n_total": len(val_subscores),
                "discovery_eval_count": result.discovery_eval_counts[idx],
                "is_seed": idx == 0,
            }
        )
    append_jsonl(iterations_path, iteration_rows)

    best_idx = result.best_idx
    best_candidate_text = result.best_candidate
    print(f"best candidate (idx {best_idx}, val_accuracy={result.val_aggregate_scores[best_idx]:.4f}):")
    print(best_candidate_text)

    # `no_improvement_stopper` only ever fires before the budget is spent
    # (MaxMetricCallsStopper -- always active -- would otherwise have ended
    # the run right at max_metric_calls); a call count short of the budget
    # is therefore evidence the no-improvement condition tripped first, i.e.
    # this task/model pairing is likely near-ceiling (see gepa_repro_README.md).
    converged_early = no_improvement_stopper is not None and result.total_metric_calls < args.max_metric_calls
    best_out = {
        "task": args.task,
        "candidate": best_candidate_text,
        "val_accuracy": result.val_aggregate_scores[best_idx],
        "baseline_val_accuracy": result.val_aggregate_scores[0],
        "iteration": best_idx,
        "total_metric_calls": result.total_metric_calls,
        "max_metric_calls": args.max_metric_calls,
        "no_improvement_patience": args.no_improvement_patience,
        "converged_early": converged_early,
    }
    if converged_early:
        print(
            f"stopped early after {result.total_metric_calls}/{args.max_metric_calls} metric calls: "
            f"no val-set improvement for {args.no_improvement_patience} iterations "
            "(likely near-ceiling for this task/model)"
        )

    if args.eval_test:
        # optimize_anything()'s built-in `test_set` held-out pass requires
        # the new OptimizeAnythingConfig API and raises ValueError when
        # combined with the legacy GEPAConfig object we pass above (see
        # gepa_repro/src/gepa/optimize_anything.py::optimize_anything), so we
        # score the seed and best candidates on the test split ourselves,
        # mirroring upstream's own examples/aime_math/utils.py::evaluate_on_dataset
        # pattern instead.
        test_eval_path = os.path.join(data_dir, "test_eval.jsonl")
        open(test_eval_path, "w").close()

        def eval_on_test(candidate_text, tag):
            responses = batched_generate(
                llm,
                tokenizer,
                candidate_text,
                [ex["question_prompt"] for ex in test_rows],
                max_tokens=forward_max_tokens,
                enable_thinking=args.enable_thinking,
            )
            n_correct = 0
            rows = []
            for ex, response in zip(test_rows, responses, strict=True):
                predicted = parse(response)
                gold = parse(str(ex["answer"]))
                correct = predicted is not None and predicted == gold
                n_correct += int(correct)
                rows.append(
                    {
                        "candidate_tag": tag,
                        "candidate": candidate_text,
                        "question": ex["question_prompt"],
                        "gold_answer": ex["answer"],
                        "model_response": response,
                        "predicted_answer": predicted,
                        "correct": correct,
                    }
                )
            append_jsonl(test_eval_path, rows)
            return n_correct / len(test_rows)

        baseline_test_accuracy = eval_on_test(seed_prompt, "seed")
        test_accuracy = eval_on_test(best_candidate_text, "best")
        print(f"baseline (seed prompt) test_accuracy={baseline_test_accuracy:.4f}")
        print(f"optimized (best candidate) test_accuracy={test_accuracy:.4f}")
        best_out["baseline_test_accuracy"] = baseline_test_accuracy
        best_out["test_accuracy"] = test_accuracy

    with open(best_prompt_path, "w") as f:
        json.dump(best_out, f, indent=2, default=_json_default)
    print(f"wrote {best_prompt_path}")
    print(f"reflection_lm calls used: {reflection_lm.num_calls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="Qwen/Qwen3-14B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="bumped automatically to a task's min_max_model_len (e.g. aime's 32768), matching "
        "textgrad_repro.py's own behavior",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="task-solving max_tokens fallback; overridden per-task by TASKS[...]['max_tokens'] when set "
        "(e.g. aime's 16000)",
    )
    parser.add_argument("--reflection_max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=3, help="reflection_minibatch_size, matches textgrad_repro")
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=3936,
        help="eval-call budget; default matches textgrad_repro.py's total for a 12-step, 300-example-val run "
        "(12 steps x 3 train + 13 val passes x 300)",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="gsm8k",
        help="which task's splits/prompt/parser to use -- see textgrad_repro.TASKS and TEXTGRAD_MULTITASK_PLAN.md",
    )
    parser.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
        help="Qwen3 thinking mode for the solver role (default OFF here, unlike the original "
        "gepa_repro_aime.py's default-on -- see module docstring). Reflection always runs with "
        "thinking off regardless of this flag (see gepa_repro_common.VLLMLanguageModel).",
    )
    parser.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--data_dir", default=None, help="default: data/gepa_repro/{model_dir}_{task}_gepa-repro/")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_improvement_patience",
        type=int,
        default=None,
        help="opt-in early-stop diagnostic (off by default, unlike the paper's own protocol -- see "
        "gepa_repro_README.md): stop once the val-set best score has gone this many iterations without "
        "improving, and record converged_early=true in best_prompt.json. Does not change what accuracy is "
        "reachable, only saves budget on tasks that have already flatlined.",
    )
    main(parser.parse_args())
