"""One-off: compute gsm8k's baseline (pre-training) val accuracy.

The original gsm8k run (data/textgrad_repro/qwen-qwen3-14b_gsm8k_textgrad-repro/)
predates textgrad_repro.py's baseline_val_accuracy field being added to
best_prompt.json, so that number was never recorded. Reuses the same
ChatVLLM engine / eval_split / batched_generate machinery as
textgrad_repro.py against the *existing* val_set.jsonl (same 300 rows the
original run used) and the *existing* baseline prompt (read back from
baseline_test_accuracy.json, which does have the pre-training prompt text).
Writes baseline_val_accuracy.json alongside the other artifacts; does not
touch best_prompt.json, test_eval.jsonl, or anything else.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from textgrad_repro import (
    ChatVLLM,
    _patch_chat_vllm_engine,
    eval_split,
    ANSWER_PARSERS,
    TASKS,
    set_seed,
)

DATA_DIR = "data/textgrad_repro/qwen-qwen3-14b_gsm8k_textgrad-repro"
MODEL_DIR = "Qwen/Qwen3-14B"
ENABLE_THINKING = False
SEED = 42

set_seed(SEED)

spec = TASKS["gsm8k"]
parse = ANSWER_PARSERS[spec["parser"]]
forward_max_tokens = spec.get("max_tokens", 2000)
_patch_chat_vllm_engine(ENABLE_THINKING, default_max_tokens=forward_max_tokens)

with open(os.path.join(DATA_DIR, "baseline_test_accuracy.json")) as f:
    baseline_prompt = json.load(f)["prompt"]

val_rows = []
with open(os.path.join(DATA_DIR, "val_set.jsonl")) as f:
    for line in f:
        val_rows.append(json.loads(line))
print(f"loaded {len(val_rows)} val rows, baseline prompt: {baseline_prompt!r}")

engine = ChatVLLM(
    MODEL_DIR,
    gpu_memory_utilization=0.85,
    max_model_len=16384,
    seed=SEED,
)

scratch_forward_outputs_path = os.path.join(DATA_DIR, "baseline_val_forward_outputs_SCRATCH.jsonl")
open(scratch_forward_outputs_path, "w").close()

baseline_val_accuracy, _ = eval_split(
    engine,
    baseline_prompt,
    val_rows,
    -1,
    "val",
    scratch_forward_outputs_path,
    ENABLE_THINKING,
    parse,
    max_tokens=forward_max_tokens,
)
n_correct = round(baseline_val_accuracy * len(val_rows))
print(f"baseline_val_accuracy={baseline_val_accuracy:.4f} ({n_correct}/{len(val_rows)})")

out_path = os.path.join(DATA_DIR, "baseline_val_accuracy.json")
with open(out_path, "w") as f:
    json.dump(
        {
            "prompt": baseline_prompt,
            "val_accuracy": baseline_val_accuracy,
            "n_correct": n_correct,
            "n_total": len(val_rows),
        },
        f,
        indent=2,
    )
print(f"wrote {out_path}")

os.remove(scratch_forward_outputs_path)
