# 01 — Environment

**Status: implemented.** Run `bash run_01_env.sh`.

## Goal

A B200-first (sm_100 / Blackwell) Python environment that is *flexible* about versions rather than
frozen to them, and that can be created on a CPU-only login node with no CUDA toolkit.

## Why not reuse `text-to-lora`'s environment

`/home/dg793/text-to-lora` is a useful architectural reference, but its environment is a dead end
for this project. Per its own `B200_EVAL_ENV_FIXES.md`, it is locked to:

- `transformers==4.51.1` — an exact pin, because its model code imports
  `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding` (a private path that moved in
  later releases) and because `vllm==0.9.2` conflicts with transformers ≥ 4.57.
- `vllm==0.9.2` — which in turn drags `torch`, `transformers`, `triton`, and `xformers` into a
  narrow simultaneously-satisfiable window.
- `flash-attn==2.8.3.post1` **built from source**, because no sm_100 wheel existed; requires
  `CUDA_HOME`, `FLASH_ATTN_CUDA_ARCHS=100`, `MAX_JOBS=16`, and a `uv cache clean flash-attn` dance
  to avoid silently reusing a stale build.

This repo avoids all three failure modes by construction:

1. **Public APIs only.** The model code touches `AutoModel`, `AutoConfig`, `AutoTokenizer`,
   `inputs_embeds`, `attention_mask`, `position_ids`, `get_input_embeddings()`, and
   `nn.Module.register_forward_hook`. Every one of these is stable across transformers 4.x → 5.x.
   No private module paths, no monkey-patching of attention internals.
2. **No vLLM on the critical path.** Validation is loss-based (see `03_training_validation.md`), so
   no inference engine is needed. vLLM stays an optional extra for a future generation-based eval.
3. **No local CUDA compilation.** The attention backend is a *runtime config field*, defaulting to
   `sdpa`, which on Blackwell dispatches to the fused cuDNN kernel and needs nothing built.

## Dependency policy

`pyproject.toml` declares **lower bounds**, not pins:

| package | constraint | why |
|---|---|---|
| `torch` | `>=2.9` | first releases with settled `sm_100` support; installed from the `cu128` index |
| `transformers` | `>=5.0` | only public APIs are used, so newer is fine |
| `peft` | `>=0.18` | `LoraConfig`, `get_peft_model`, `autocast_adapter_dtype` |
| `accelerate` | `>=1.11` | used by the (later) trainers |
| `datasets` | `>=4.0` | jsonl loading |

`uv.lock` is committed, so a run is exactly reproducible; the loose `pyproject.toml` means
`uv lock --upgrade` is a one-line, low-risk operation when a newer torch/transformers is wanted.

Optional extras:

- `attn` → `kernels`. transformers ≥ 5 can fetch a **prebuilt** flash-attn kernel from the Hub via
  `attn_implementation="kernels-community/flash-attn2"` (renamed from `flash-attn` upstream; the old
  slug now just redirects). This is the flash-attn path that does not require a compiler. On a
  compute node running `HF_HUB_OFFLINE=1`, pin a revision (e.g.
  `kernels-community/flash-attn2@c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1`) and pre-fetch it once
  with network access first -- see `docs/03_training_validation.md`'s GPU-bugs section for why
  (the unpinned form re-resolves "version" against the Hub on every load, which offline mode
  blocks even when the content is already cached).
- `gen` → `vllm>=0.11`. Only if generation-based evaluation is added later.
- `dev` → `pytest`, `ruff`.

## Attention backend

Configured, never hardcoded. `HyperNetConfig.attn_implementation` accepts:

| value | requires | notes |
|---|---|---|
| `sdpa` (default) | nothing | fused cuDNN attention on Blackwell; correct everywhere including CPU tests |
| `flash_attention_2` | a locally installed `flash-attn` | only if you have already built one |
| `kernels-community/flash-attn2` | `pip install kernels` | prebuilt, downloaded at load time; pin `@<revision>` for offline nodes |
| `eager` | nothing | used by the CPU test-suite fixtures |

## Hardware

Training targets one B200 (180 GB, sm_100). Job submission is deliberately **not** part of this
repo — no slurm scripts, no `sbatch` wrappers — so the environment can be built and verified
without holding a GPU allocation. Every entry point (`run_01_env.sh`, `run_02_model.sh`,
`scripts/smoke_check.py`) detects the device and runs on CPU when there is none.

The **login node has no CUDA toolkit and 4 cores**, and nothing in the setup path compiles CUDA,
so the whole environment is built there. `scripts/smoke_check.py` then loads the real
Qwen2.5-3B/1.5B models and runs a full forward+backward on CPU — the same code path a GPU run
takes. What that cannot cover is only kernel dispatch on `sm_100` and real memory headroom.

## Model weights

`scripts/prefetch_models.py` downloads the backbone and target into the HF cache, and
`run_01_env.sh` calls it. Afterwards everything can run with `HF_HUB_OFFLINE=1`, which is what
you want on a compute node — a Hub call that blocks or rate-limits mid-job wastes the
allocation.

```
python scripts/prefetch_models.py --check    # report status, download nothing
python scripts/prefetch_models.py            # fetch what is missing
```

## Environment check

`python -m steerable_t2l.utils.env` (also importable as `check_env()`) prints a version table and
verifies:

- `torch`, `transformers`, `peft`, `accelerate`, `datasets` are importable and meet lower bounds;
- whether CUDA is available, and if so the device name, capability, and total memory;
- whether `sm_100` appears in `torch.cuda.get_arch_list()` — a mismatch here is the single most
  common cause of the "compiled for a different architecture" class of failures on Blackwell;
- which attention backends are usable in this interpreter.

It exits nonzero if a hard requirement is missing, and warns (exit 0) for GPU-only checks when run
on a CPU node.

## Files

| path | role |
|---|---|
| `pyproject.toml` | dependency ranges, uv index for cu128 torch |
| `run_01_env.sh` | create venv, sync, env check, prefetch weights |
| `src/steerable_t2l/utils/env.py` | `check_env()` + `__main__` |
| `scripts/prefetch_models.py` | download backbone + target into the HF cache |
| `scripts/smoke_check.py` | real-model end-to-end check, CPU or GPU |

## Verified state

Built and checked on the login node (CPU, no GPU):

```
torch          2.11.0+cu128    built for: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
transformers   5.14.1
peft           0.20.0
accelerate     1.14.0
datasets       5.0.1
kernels        0.16.0
wandb          0.28.1
python         3.12.3
```

Both `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` are cached. `sm_100` is read via
`torch._C._cuda_getArchFlags()` rather than `torch.cuda.get_arch_list()`, which returns `[]`
without a driver and would be useless precisely where a wrong wheel needs catching.
