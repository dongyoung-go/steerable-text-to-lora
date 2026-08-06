# 02 — Model

**Status: implemented.** Run `bash run_02_model.sh`.

Implements the architecture in `Design.md`: a hypernetwork that maps a natural-language steering
instruction to the complete LoRA parameter set of a frozen target LLM.

```
instruction text
   │  chat template, left-padded tokenize
   ▼
[ instruction tokens ‖ 224 compositional query tokens ]     ← q_base + e_layer + e_module + e_role
   │  causal forward through a frozen, LoRA-tuned Qwen2.5-3B-Instruct
   ▼
224 causal hidden states                                    [U, 224, 2048]
   │  non-causal self-attention refinement (2 layers, queries only)
   ▼
224 refined hidden states
   │  shared MLP  →  module-specific low-rank head (Linear(d→k) → Linear(k→r·f))
   ▼
LoRA A [U, 28, r, in_f] and B [U, 28, out_f, r] for q/k/v/o_proj
   │  differentiable per-sample forward hooks
   ▼
frozen target Qwen2.5-1.5B-Instruct
```

## Configuration

| | value | note |
|---|---|---|
| hypernet backbone | `Qwen/Qwen2.5-3B-Instruct` | h = 2048, 36 layers — matches `Design.md`'s `h=2048` example |
| target model | `Qwen/Qwen2.5-1.5B-Instruct` | h = 1536, **28 layers**, GQA (12 q-heads / 2 kv-heads, head_dim 128) |
| adapted modules | `q_proj, k_proj, v_proj, o_proj` | |
| LoRA rank | `r = 8`, `alpha = 16`, `use_rslora = False` | ⇒ `scaling = alpha/r = 2.0` |
| query tokens | **28 × 4 × 2 = 224** | `Design.md`'s "256" assumes a 32-layer target |
| head bottleneck | `k = 128` | |

The target is a config knob; moving to `meta-llama/Llama-3.1-8B-Instruct` (32 layers ⇒ 256 queries)
is a one-line change and everything downstream re-derives from `AutoConfig`.

---

## `target_spec.py` — shapes without weights

`TargetSpec` is a frozen dataclass built from `AutoConfig` **alone**. No model weights are loaded.
This matters for the reconstruction trainer, which needs the target's *shapes* but never its
*weights*; the reference implementation requires a live `PeftModel` and therefore holds 3 GB it
never uses.

```python
spec = TargetSpec.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                  target_modules=("q_proj","k_proj","v_proj","o_proj"),
                                  r=8, lora_alpha=16)
spec.n_layers      # 28
spec.n_queries     # 224
spec.scaling       # 2.0
spec.in_features   # {'q_proj': 1536, 'k_proj': 1536, 'v_proj': 1536, 'o_proj': 1536}
spec.out_features  # {'q_proj': 1536, 'k_proj':  256, 'v_proj':  256, 'o_proj': 1536}
```

⚠️ **GQA asymmetry.** `k_proj`/`v_proj` project to `num_key_value_heads * head_dim = 256`, not
`hidden_size`. And `o_proj.in_features == num_attention_heads * head_dim`, which merely *happens*
to equal `hidden_size` for Qwen2.5-1.5B. All four widths are derived from the config so the spec
transfers to models where they diverge.

### Query index layout

One convention, used by the query bank, the heads, and every test:

```
idx = layer * (n_modules * 2) + module_idx * 2 + role        # role: 0 = A, 1 = B
```

`spec.query_index(layer, module, role)` and `spec.query_indices()` (returning the three `[224]`
index tensors) are the only places this arithmetic appears.

---

## `hypernet.py`

### `CompositionalQueries`

```
q_i = q_base + e_layer[l_i] + e_module[m_i] + e_role[r_i]      then LayerNorm
```

35 learned vectors (`1 + 28 + 4 + 2`) of width 2048 instead of 224 independent embeddings, so
layer/module/role identity are disentangled additive factors — this is what lets rare or unseen
`(layer, module, role)` combinations still be assembled from well-trained shared parts.

The tables are initialized `N(0, 0.02)` and then **rescaled to the backbone's input-embedding RMS**.
These vectors are fed directly into a pretrained LLM; at the default init scale they would be
roughly an order of magnitude off the token-embedding distribution and land out-of-distribution
for the first attention layer.

### `SteerableHyperLoRA.encode(descs) -> [U, 224, 2048]`

```python
texts = [tok.apply_chat_template([{"role": "user", "content": d}],
                                 tokenize=False, add_generation_prompt=True) for d in descs]
enc   = tok(texts, padding=True, truncation=True, max_length=max_desc_len,
            add_special_tokens=False, return_tensors="pt")     # padding_side = "left"

tok_emb       = backbone.get_input_embeddings()(enc.input_ids)          # [U, L, 2048]
queries       = self.queries()                                          # [224, 2048]
inputs_embeds = cat([tok_emb, queries.expand(U, -1, -1)], dim=1)
attn_mask     = cat([enc.attention_mask, ones(U, 224)], dim=1)
position_ids  = (attn_mask.cumsum(-1) - 1).clamp(min=0)

h = backbone(inputs_embeds=..., attention_mask=..., position_ids=...,
             use_cache=False).last_hidden_state[:, -224:, :]
h = shared_mlp(refiner(h.float()))                                       # fp32
```

Four non-obvious points, each guarded by a test:

**`AutoModel`, never `AutoModelForCausalLM`.** The causal-LM wrapper computes logits over Qwen's
151,936-token vocabulary for all `L + 224` positions. At `U=8, L=300` that is ~1.3 GB of bf16
activations held in the autograd graph for a tensor that is immediately discarded. `AutoModel`
returns `last_hidden_state` only.

**Left padding.** With `padding_side="left"` the 224 query tokens are always the final 224
positions of every row, so `h[:, -224:]` is a constant slice. With right padding the offset varies
per row and would need a gather.

**Explicit `position_ids` is required, not cosmetic.** The default is `arange(L + 224)`, which
*counts pad tokens*. The same description batched next to a longer neighbour would then receive
different RoPE phases on its query tokens and emit a *different LoRA* — a silent,
batch-composition-dependent nondeterminism. `(attn_mask.cumsum(-1) - 1).clamp(min=0)` places the
first real token at position 0 and query `j` at `L_true + j` regardless of padding.
`tests/test_padding_invariance.py` locks this down.

**The causal mask is left alone.** `Design.md` asks for causal attention over
`[instruction ‖ queries]`, which is the default. The non-causal correction lives in the refiner.

### Backbone

Frozen bf16 base + `peft.LoraConfig(r=16, alpha=32, lora_dropout=0.0, target_modules=q/k/v/o/gate/up/down)`,
`autocast_adapter_dtype=True` (adapters in fp32), gradient checkpointing with
`use_reentrant=False`.

⚠️ `lora_dropout=0.0` on the backbone is deliberate. Nonzero dropout makes the generated LoRA
stochastic for a fixed instruction, which breaks the "identical descriptions ⇒ identical LoRAs"
invariant that batch-level description deduplication depends on, and makes train- and eval-time
behavior diverge silently.

### `QueryRefiner` — non-causal fixup

Two pre-LN blocks of `nn.MultiheadAttention(2048, 16, batch_first=True)` + MLP, **with no mask**.
The causal pass forces query `i` to only see queries `< i` in an arbitrary serialization order;
since the 224 queries are a *set*, not a sequence, this stack lets layer-0 and layer-31 query
representations inform each other. Attention here is only ever over 224 tokens, never over the
instruction text, so it is cheap. Output projections are downscaled by `1/sqrt(2·n_blocks)` at init
so the stack starts near identity.

### `SharedDecoder` and heads

- `SharedDecoder`: `LN → Linear(2048→4096) → SiLU → Linear(4096→2048) → LN`, applied identically to
  all 224 tokens.
- `LoRAHead`: `Linear(2048 → k=128) → Linear(k → r·f)`. A direct `2048 → r·f` map would be
  parameter-heavy; the rank-`k` factorization costs `h·k + k·r·f` instead of `h·r·f`, with
  negligible expressivity loss since the target is already a low-rank structured object.

⚠️ **Deviation from `Design.md`, and why.** The design says "only 4 module-specific projection
heads … shared across all layers". Sharing across layers is implemented as written (layer identity
is carried additively by `e_layer`). Sharing a single head across the A and B *roles* is not
possible: A's output width is `r · in_features` and B's is `r · out_features`, and under GQA those
differ (12288 vs 2048 for `k_proj`/`v_proj`). The resolution keeps the spirit — **the
`Linear(2048→128)` bottleneck is shared per module across both roles**, so there are genuinely four
module-specific projections — and only the small `Linear(128 → r·f)` output layer is
per-(module, role).

### dtype policy

| component | storage | compute |
|---|---|---|
| backbone base weights | bf16, frozen | bf16 |
| backbone LoRA adapters | fp32 | bf16 under autocast |
| query bank | fp32 | cast to bf16 only at the `cat` |
| refiner / shared MLP / heads | fp32 | **fp32** (`autocast(enabled=False)`) |
| generated A / B | fp32 → cast | bf16 into the hooks |

The from-scratch stack only ever touches `[U, 224, 2048]` — at `U=8` that is 3.7 M activations, so
fp32 is free. It is also necessary: the zero-init contract requires bit-exact zeros and a bit-exact
PEFT initialization, and early-training gradients are tiny because they pass through `B ≈ 0`, which
bf16 would flush.

---

## The zero-init contract

At step 0 the generated LoRA must be an **exactly** zero delta, so training starts from the
unmodified base model and step-0 loss equals the frozen model's loss bitwise.

```python
nn.init.zeros_(head[f"{m}_A"].out.weight)
head[f"{m}_A"].out.bias.copy_(peft_init_lora_A[m].flatten())   # Kaiming, from a throwaway PEFT LoRA
nn.init.zeros_(head[f"{m}_B"].out.weight)
nn.init.zeros_(head[f"{m}_B"].out.bias)
```

Because the B head's out-linear has both weight *and* bias zero, `B ≡ 0` for **every** input — so
the values of the refiner, shared MLP, and bottleneck are irrelevant to the zero-delta property,
and `ΔW = B @ A = 0` exactly, not approximately.

### ⚠️ The real hazard: a dead gradient, not a nonzero delta

Be precise about what is live on the first step. With `out_B.weight == 0`:

```
dL/dA                    ∝  B == 0                                        →  out_A idle
dL/d(bottleneck output)  =  out_A.weightᵀ·dL/dA  +  out_B.weightᵀ·dL/dB
                         =         0             +          0             →  idle
```

So **on step 0 only `out_B`'s weight and bias receive gradient** — exactly as in standard LoRA,
where `B = 0` means only B moves first. The bottleneck, shared MLP, refiner, query bank and
backbone LoRA all unblock on step 1, once `out_B.weight` is nonzero. Measured on the tiny fixture:

| | step 0 | step 1 | step 2 |
|---|---|---|---|
| `heads` | 13.9 | 13.5 | 15.9 |
| `shared_decoder` | **0** | 0.011 | 0.023 |
| `queries` | **0** | 0.021 | 0.038 |
| `refiner` | **0** | 0.0035 | 0.0065 |
| `backbone_lora` | **0** | 0.0027 | 0.0049 |

That one-step lag is intended and is not a saddle. What *would* be fatal is the unblocking never
happening: `dL/d(out_B.weight) ∝ dL/dB ⊗ bottleneck_activation`, so if the shared MLP or the
bottleneck emitted zeros, `B` would be pinned at 0 **forever** — a complete training failure that
raises no error and shows up only as a flat loss curve. Three design constraints follow:

1. The shared MLP is **not** zero-initialized and does **not** end in a zero-gated residual. It
   ends in a `LayerNorm`, which guarantees O(1) output RMS regardless of the incoming scale.
   (`x + gate·mlp(x)` with `gate=0` would be fine — output `= x ≠ 0`. A LayerScale-style block that
   zeroes the *entire* output would not. Do not introduce one.)
2. `LoRAHead.bottleneck` is `normal_(0.02)`-initialized. Only the **out** linears are zeroed.
3. No dropout inside the shared MLP or the bottleneck.

`tests/test_grad_flow.py` asserts the step-0 pattern *and* that every group is live after one
optimizer step. Both halves matter: the first documents the intended lag so a change that removes
it is noticed, the second catches permanent starvation.

`zero_init` is disabled when warm-starting (`init_from is not None`) so a reconstruction-trained
checkpoint is never re-zeroed.

---

## `hooks.py` — differentiable injection

```python
with lora_hooks(target_model, sites, scaling):
    out = target_model(input_ids=..., attention_mask=..., labels=...)
loss = out.loss                    # gradients flow back into the hypernetwork
```

`sites` is a list of `(layer_idx, module_name, A [bs, in_f, r], B [bs, r, out_f])`. The hook is a
`register_forward_hook` on the target `nn.Linear`, so `args[0]` is that linear's input `x` and
`output` is `W x`; the hook returns `W x + scaling · (x A) B`. Because `A` and `B` carry a batch
dimension, **every sample in the batch gets its own LoRA** — which is what makes multi-task
batching possible.

```python
delta = torch.bmm(torch.bmm(x, A), B).mul_(scaling)
```

⚠️ **The reference implementation's hook must not be copied.**
`text-to-lora/src/hyper_llm_modulator/hooks.py:163-166` does
`A.repeat_interleave(inp_len, dim=0)`, materializing a `[bs·L, in_f, r]` tensor **and keeping it
alive for backward**. At `bs=8, L=1024, q_proj` that is 201 MB per site, ≈**22 GB** across
28 layers × 4 modules, for data that is 1024× redundant. `torch.bmm` already broadcasts over the
leading batch dimension, so no repeat is needed: extra saved-for-backward memory drops to
`[bs, L, r]` ≈ 131 KB per site, **≈15 MB total**. (This is why the reference could only ever run
`q_proj,v_proj` at `batch_size=4`.)

Added FLOPs are ~1 % of the base linear's; the real cost is launching 224 extra small `bmm`s per
forward, so expect 10–20 % wall-clock overhead.

Two further contracts:

- `lora_hooks` is a context manager with `try/finally`, so handles are removed even if the wrapped
  forward raises. The reference leaks handles on exception, which corrupts every subsequent step.
- ⚠️ If gradient checkpointing is ever enabled on the **target** model, `use_reentrant=False` is
  mandatory. Reentrant checkpointing re-runs forward hooks during recompute and would build a
  second, detached path around the closed-over `A`/`B`.

No layer-major flattening contract is needed here (the reference needs one, and has a latent bug in
it): the hypernetwork natively produces `[bs, n_layers, ...]`, so `build_sites()` simply indexes
`A[m][:, layer]`.

---

## Parameter budget

| group | params |
|---|---|
| backbone LoRA, r=16, 7 modules × 36 layers | ≈ 7 M |
| compositional queries (35 × 2048) | 0.07 M |
| query refiner, 2 layers, d=2048, mlp×4 | ≈ 100 M |
| shared decoder 2048→4096→2048 | ≈ 17 M |
| 8 low-rank heads, k=128 | ≈ 12 M |
| **total trainable** | **≈ 136 M** |

fp32 params + grads + AdamW ⇒ ≈ 2.2 GB. If the refiner needs to be cheaper, `mlp_ratio=2` halves it.

`python -m steerable_t2l.hypernet --self-check` prints the real numbers for the configured spec.

---

## Tests

CPU-only, no network, no GPU, no CUDA toolkit. Fixtures build tiny random `Qwen2Config` models
(`hidden_size=32`, 4 layers, 4 q-heads / 2 kv-heads) for both backbone and target, in fp32 with
`attn_implementation="eager"`.

| file | asserts |
|---|---|
| `test_shapes.py` | `TargetSpec.from_pretrained` GQA widths; `n_queries` arithmetic; `scaling` |
| `test_query_layout.py` | `(layer, module, role) ↔ index` round-trips; heads read the token they claim to |
| `test_zero_init.py` | every `B` is **exactly** 0 (`(B == 0).all()`, not `allclose`); `ΔW == 0`; `A` equals PEFT's init; hooked target logits `==` unhooked with `atol=0` |
| `test_grad_flow.py` | after one backward, **every** trainable group has a nonzero grad — `q_base`, all three tables, refiner, shared MLP, each head's `bottleneck` **and** `out`, backbone LoRA. Catches the dead-shared-MLP failure above. **Gate on this test.** |
| `test_padding_invariance.py` | a description encoded alone equals the same description left-padded beside a longer one; regression-guards `position_ids` |
| `test_dedup.py` | `A_uniq[inv]` equals the naive per-sample path, forward and gradient |
| `test_hook_vs_peft.py` | hook injection vs. writing a PEFT adapter and loading it with `PeftModel` → identical logits; catches transpose, `scaling`, and `use_rslora` mistakes |
| `test_hook_cleanup.py` | handles are removed even when the wrapped forward raises |

## Files

| path | role |
|---|---|
| `src/steerable_t2l/target_spec.py` | `TargetSpec`, query index layout |
| `src/steerable_t2l/hypernet.py` | `CompositionalQueries`, `QueryRefiner`, `SharedDecoder`, `LoRAHead`, `SteerableHyperLoRA`, `HyperNetConfig` |
| `src/steerable_t2l/hooks.py` | `lora_hooks`, `build_sites`, `get_layers` |
| `run_02_model.sh` | ruff → pytest → self-check |
