# Behavior-Level Text-to-LoRA

A hypernetwork that maps a natural-language **steering instruction** to the complete LoRA parameter
set of a frozen target LLM. The generated adapter is injected without any gradient update on the
target; the downstream task loss backpropagates through the generated weights into the
hypernetwork.

The full specification is in [`Design.md`](Design.md).

```
instruction text
   │  chat template, left-padded tokenize
   ▼
[ instruction tokens ‖ 224 compositional query tokens ]   q_i = q_base + e_layer + e_module + e_role
   │  causal forward through a frozen, LoRA-tuned Qwen2.5-3B-Instruct
   ▼
224 causal hidden states
   │  non-causal self-attention refinement (queries only)
   ▼
   │  shared MLP  →  module-specific low-rank head
   ▼
LoRA A / B for q,k,v,o_proj × 28 layers
   │  differentiable per-sample forward hooks
   ▼
frozen target Qwen2.5-1.5B-Instruct  →  task loss
```

## Status

| phase | doc | state |
|---|---|---|
| 1. Environment | [`docs/01_env.md`](docs/01_env.md) | **implemented** — `bash run_01_env.sh` |
| 2. Model | [`docs/02_model.md`](docs/02_model.md) | **implemented** — `bash run_02_model.sh` |
| 3. Training & validation | [`docs/03_training_validation.md`](docs/03_training_validation.md) | **specification only** |

Phase 3 (data pipeline, oracle LoRAs + SVD canonicalization, reconstruction warm-start, SFT
trainer, validation protocol) is fully specified but deliberately left unimplemented. The seams
exist as `NotImplementedError` stubs under `src/steerable_t2l/{data,oracle,trainers}/` and
`src/steerable_t2l/validation.py`.

## Quick start

```bash
bash run_01_env.sh           # uv venv + sync + env check + prefetch weights
bash run_02_model.sh         # ruff + 48 tests + model self-check          (~15s)
bash run_02_model.sh --full  # ... plus real Qwen2.5 3B/1.5B fwd+bwd       (~3 min)
```

All of this runs on a CPU login node — no GPU, no CUDA toolkit, no compilation. Job submission is
deliberately not part of this repo, so the environment can be built and verified without holding a
GPU allocation.

`scripts/smoke_check.py` is device-agnostic: it picks up a GPU when one is present and otherwise
runs the same code path on CPU.

```bash
python scripts/smoke_check.py --batch 2 --seq 64   --backward   # CPU
python scripts/smoke_check.py --batch 8 --seq 1024 --backward   # GPU, auto-detected
```

## Layout

```
Design.md                      the specification this implements
docs/01_env.md                 environment & dependency policy
docs/02_model.md               architecture, invariants, tests
docs/03_training_validation.md brief for the next session (not implemented)
src/steerable_t2l/
  target_spec.py               LoRA shapes + query index layout, from AutoConfig alone
  hooks.py                     differentiable per-sample LoRA injection
  hypernet.py                  the adapter generator
  testing.py                   tiny CPU fixtures shared by tests and the self-check
  utils/env.py                 environment check
  data/ oracle/ trainers/      phase-3 stubs
  validation.py                phase-3 stub
scripts/prefetch_models.py     download backbone + target into the HF cache
scripts/smoke_check.py         real-model end-to-end check, CPU or GPU
tests/                         8 test files, 48 CPU-only tests
```

## Relationship to `text-to-lora`

`/home/dg793/text-to-lora` (SakanaAI T2L plus local extensions) is an **architectural reference
only**. Its environment is frozen at `transformers==4.51.1` / `vllm==0.9.2` / a source-built
`flash-attn` because its model code imports private transformers paths and because vLLM constrains
the whole dependency graph.

This repo deliberately diverges:

- **Public APIs only** (`AutoModel`, `inputs_embeds`, `attention_mask`, `position_ids`,
  `register_forward_hook`), so version ranges replace exact pins.
- **No vLLM on the critical path** — validation is loss-based.
- **No local CUDA compilation** — the attention backend is a runtime config field defaulting to
  `sdpa`.
- **A different architecture.** The reference conditions on a pooled or token-level `gte-large`
  embedding through a from-scratch encoder; here the instruction encoder is a pretrained LLM that
  is itself in the training graph, with the query tokens processed causally alongside the
  instruction.
