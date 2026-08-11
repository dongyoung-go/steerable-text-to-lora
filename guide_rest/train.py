"""Guide-ReST Step 4: Improve. Fine-tunes a fresh LoRA adapter, initialized from the raw
base model (never from `M_t`'s already-tuned weights -- see docs/guide_rest_README.md's
plan notes / the user's explicit decision), on this round's `filtered.jsonl`
(question, correct completion) pairs. Trains the bare question -> completion mapping only
(no feedback prefix -- Step 4 in the README is explicit that the Grow-time feedback prefix
is stripped before fine-tuning). Merges the adapter onto the base and saves full weights as
`M_{t+1}`, loadable directly by `sampling.py`'s vLLM engine next round.

Plain PyTorch training loop (no `accelerate`), same style as
`src/steerable_t2l/oracle/train_oracle.py`'s single-GPU LoRA trainer -- this repo already
has that pattern for "one LoRA on the frozen target model", just adapted here from its
narrow `TargetSpec`-matched config to Guide-ReST's own (wider) LoRA config and from its
per-task-description dataset to plain (question, completion) SFT pairs.

**`--epochs` is a cap, not a target -- early stopping can stop sooner.** ReST-EM (Singh
et al., "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language
Models," TMLR 2024 -- the ReST variant that actually targets MATH/GSM8K, unlike the
original translation-focused ReST paper) trains the Improve step with `while reward
improves on D_val` (their Algorithm 1), explicitly because "LLMs overfit to small datasets
quickly" and they observed train accuracy climbing linearly with rounds while test
accuracy plateaued or regressed (their Figure 4) from overfitting on the (small,
model-generated) training pool. Unlike an earlier version of this file, validation data is
NOT carved out of this round's own `filtered.jsonl` -- that would shrink an already-small
per-round training set further and give a validation signal drawn from the same
distribution as training (same round, same model, same generation batch), a weaker
independence guarantee than ReST-EM's own `D_val`, which their Algorithm 1 takes as an
input separate from that round's Generate-step output `D_i`. Instead, `--dev_filtered`
points at `sampling.py`'s `dev_filtered.jsonl` -- a small, fixed pool of questions disjoint
from the Grow pool (see `tasks.py::load_*_dev_pool`), sampled in the same vLLM call as
Grow every round, always unconditioned. Early-stops on that pool's loss within the
`--epochs` cap, via the same patience/min-delta `EarlyStopper` pattern
`src/steerable_t2l/oracle/train_oracle.py::EarlyStopper` already uses for the same reason
(duplicated here, not imported -- see `tasks.py`'s module docstring for why).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tasks import TASKS, build_user_prompt

DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def load_filtered_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset(Dataset):
    """One example per filtered (question, completion) pair. Labels mask the prompt
    (question + chat-template scaffolding) to -100 so the loss is computed only over the
    completion tokens, matching standard instruction-SFT practice."""

    def __init__(self, task: str, pairs: list[dict], tokenizer, max_len: int):
        self.examples = []
        for pair in pairs:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_user_prompt(task, pair["question"])}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(pair["completion"], add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + list(completion_ids)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
            if all(label == -100 for label in labels):
                continue  # prompt alone already fills max_len; nothing to train on
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopper:
    """Patience/min-delta counter on validation loss -- ported verbatim from
    `src/steerable_t2l/oracle/train_oracle.py::EarlyStopper` (not imported, see module
    docstring)."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        if val_loss > self.best + self.min_delta:
            self.counter += 1
        return self.counter >= self.patience


def eval_loss(peft_model, loader, pad_token_id: int) -> float:
    peft_model.eval()
    total_loss, total_examples = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            total_loss += peft_model(**batch).loss.item() * batch["input_ids"].shape[0]
            total_examples += batch["input_ids"].shape[0]
    peft_model.train()
    return total_loss / total_examples if total_examples else float("nan")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_filtered_pairs(Path(args.filtered))
    if not pairs:
        # No completion passed the filter this round (e.g. the pool is already saturated,
        # or a very early round with a weak base model). Nothing to fine-tune on -- carry
        # the base model forward unchanged rather than crashing the round loop, so Grow can
        # still try again next round (possibly with an updated feedback.txt in Condition B).
        print(f"[train] no filtered pairs at {args.filtered}; copying base model forward unchanged")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    lora_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=list(args.target_modules), task_type="CAUSAL_LM", bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    train_loader = DataLoader(
        SFTDataset(args.task, pairs, tokenizer, args.max_len), batch_size=args.batch_size,
        shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    val_pairs = load_filtered_pairs(Path(args.dev_filtered)) if args.dev_filtered and Path(args.dev_filtered).exists() else []
    val_loader = None
    if val_pairs:
        val_loader = DataLoader(
            SFTDataset(args.task, val_pairs, tokenizer, args.max_len), batch_size=args.batch_size,
            shuffle=False, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
    else:
        print(f"[train] no dev pairs at {args.dev_filtered!r}; "
              f"training to the --epochs={args.epochs} cap with no early stopping")

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=args.lr,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.03 * total_steps)), num_training_steps=total_steps,
    )
    stopper = EarlyStopper(patience=args.patience) if val_loader is not None else None

    best_state = None
    best_val = float("inf")
    peft_model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = peft_model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in peft_model.parameters() if p.requires_grad), args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"[train] epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f}")

        if val_loader is not None:
            val_loss = eval_loss(peft_model, val_loader, tokenizer.pad_token_id)
            print(f"[train] epoch={epoch} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(peft_model).items()}
            if stopper.step(val_loss):
                print(f"[train] early stopping after epoch={epoch} (best val_loss={best_val:.4f})")
                break

    if best_state is not None:
        set_peft_model_state_dict(peft_model, best_state)

    peft_model.eval()
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] n_pairs={len(pairs)} (dev val={len(val_pairs)}) saved merged checkpoint to {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--base_model", required=True, help="always the raw base model, e.g. Qwen/Qwen3-14B -- never a previous round's checkpoint")
    p.add_argument("--filtered", required=True, help="this round's filtered.jsonl")
    p.add_argument("--dev_filtered", default=None, help="this round's dev_filtered.jsonl (from sampling.py's fixed dev pool) -- used as the early-stopping validation set; omit/missing = no early stopping, train to the --epochs cap")
    p.add_argument("--out_dir", required=True, help="where to save M_{t+1}'s merged weights")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3, help="cap on training epochs; early stopping (--patience) can stop sooner")
    p.add_argument("--patience", type=int, default=1, help="epochs with no val_loss improvement before stopping early")
    p.add_argument("--batch_size", type=int, default=32, help="32 measured as the throughput sweet spot on a 1x B200 (~96GB/183GB used; batch=64 gave no further speedup but used 152GB, too close to OOM risk for a long unattended run)")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
