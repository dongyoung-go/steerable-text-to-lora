# Behavior-Level Text-to-LoRA: Steering Model Reasoning via Instruction-Conditioned Adapter Generation

## Overview

The hypernetwork is formulated as a conditional parameter generator that maps a natural language task description into the complete set of LoRA parameters for a target backbone model.

Rather than learning an instruction encoder from scratch, the architecture reuses a pretrained decoder-only LLM as the reasoning backbone. The pretrained LLM provides rich semantic understanding of task descriptions, while lightweight prediction heads generate the final LoRA parameters.

---

## Architecture

### Input

The hypernetwork receives a natural language task description composed of two parts:

- **Domain**: specifies the task domain or context the target model will operate in.
- **Behavior instruction**: specifies the desired reasoning strategy, output format, or behavioral policy the target model should follow within that domain.

This mirrors the standard structure of a system prompt, which typically combines a statement of domain/context with a policy governing how the model should behave within it.

Example:

```
Domain:
"You will be asked math questions."

Behavior instruction:
"Solve the problem step-by-step.
Verify every arithmetic operation before producing the final answer."
```

The domain and behavior instruction are concatenated (with a lightweight separator or tag) into a single task description string before being passed into the backbone. Representing the input this way allows the hypernetwork to independently vary the domain and the behavior policy during training, encouraging it to learn a generation function that generalizes across domain/behavior combinations rather than memorizing fixed pairings.

---

### Backbone

A pretrained decoder-only language model (e.g., Qwen2.5-3B) serves as the backbone.

The model is initialized from public pretrained checkpoints.

The backbone is fine-tuned using LoRA while the original model weights remain frozen.

---

#### Compositional Query Construction

Each query token corresponds to one LoRA parameter group, identified by its layer, its module, and its role (LoRA A or LoRA B).

Rather than learning each query as an independent embedding, every query is constructed compositionally from a small set of shared, reusable factors:

```
q_i = q_base + e_layer + e_module + e_role
```

```
shared query token

+

layer embedding

+

module embedding

+

A/B role embedding
```

- `q_base` is a single shared base vector common to all queries.
- `e_layer` is drawn from a small embedding table indexed by transformer layer (e.g., 32 entries).
- `e_module` is drawn from a small embedding table indexed by adapted module type (e.g., q_proj, k_proj, v_proj, o_proj — 4 entries).
- `e_role` is drawn from a 2-entry table distinguishing LoRA A from LoRA B.

For a transformer with 32 layers and four adapted modules, this produces the same 256 query tokens as before:

```
#queries = 32 × 4 × 2 = 256
```

but each is now built from `32 + 4 + 2 + 1` learned vectors rather than 256 independently learned ones. This factorized construction gives the query set explicit structure: layer identity, module identity, and role are disentangled additive components rather than entangled in a single opaque embedding, which is what allows queries with unseen or rare (layer, module, role) combinations to still be built from well-trained shared factors.

---

#### Contextual Reasoning

The pretrained decoder processes

```
domain tokens
+
behavior instruction tokens
+
compositional query tokens
```

using standard causal self-attention.

Each query token attends to:

- the complete task description (domain and behavior instruction)
- all previous query tokens (under the causal mask)

allowing every generated LoRA matrix to be conditioned on the full task semantics and on the parameter generation process up to that point in the sequence.

The output of this stage is one hidden state per query token, i.e. 256 contextualized hidden states.

---

#### Query Refinement (Non-Causal Fixup)

Because the transformer pass above is causal, a query token's hidden state only incorporates information from query tokens that came earlier in the fixed serialization order (e.g., layer 0 before layer 31). Since the 256 query tokens represent a set of parameter groups rather than a true temporal sequence, this ordering is an artifact of serialization rather than a meaningful dependency.

To correct for this before decoding, the 256 causal hidden states are passed through a small, non-causal self-attention stack operating only over the query tokens:

```
256 causal hidden states (queries only)
            ↓
non-causal self-attention (1–2 layers)
            ↓
256 refined hidden states
```

This stack is trained from scratch, is small (attention is only ever computed over 256 tokens, never over the instruction text), and lets every query freely exchange information with every other query regardless of serialization order — so, for example, the layer-0 query representations can be informed by layer-31 query representations and vice versa. The instruction text itself is not re-attended to at this stage; the instruction semantics have already been folded into each query's hidden state during the causal pass.

The final hidden state of every query token, after this refinement stack, becomes the representation used for parameter generation.

---

#### Parameter Heads (Shared Decoder)

A shared parameter decoder maps the 256 refined query representations into LoRA parameters.

```
Query embedding
    ↓
Shared MLP
    ↓
Module-specific projection head
    ↓
LoRA A / LoRA B matrix
```

Each refined query embedding first passes through a shared multi-layer perceptron, which learns a common transformation from contextual query representations into the LoRA parameter space. A lightweight, module-specific projection layer then maps this shared representation into the final LoRA matrix for that query's (layer, module, role).

The shared MLP is applied identically to all 256 queries; only the final projection differs by module type. This lets parameter generation across different transformer layers and attention modules share common structural knowledge, while the final projection preserves module-specific characteristics, reducing the number of independent parameters and encouraging the hypernetwork to learn general parameter generation patterns rather than memorizing individual LoRA matrices.

**Weight tying across layers.** Since layer identity is already injected additively at the query-construction stage (`e_layer`), the final projection heads do not need to be layer-specific. Only 4 module-specific projection heads are learned (one per `q_proj`, `k_proj`, `v_proj`, `o_proj`), shared across all 32 layers — not 32 × 4 separate heads. This is a free parameter reduction: layer differentiation is already carried by the query representation entering the head, so a separate head per layer would be redundant.

**Low-rank factorized projection.** A direct linear map from the refined query hidden state (dim `h`) to a full LoRA matrix (`d × r` values) is parameter-heavy — e.g. `h=2048`, `d=2048`, `r=16` gives a `67M`-parameter head per module type. Each module-specific projection is instead factorized through a narrow bottleneck `k` (e.g. `k=128`):

```
hidden (h) → Linear(h → k) → Linear(k → d·r)
```

This caps the head's weight matrix at rank `k`, reducing parameters to `h·k + k·d·r` (≈`4.5M` for the example above, a ~15x reduction) with negligible loss of expressivity, since the target itself (a rank-`r` LoRA matrix) is already a low-dimensional structured object rather than an arbitrary dense one.

> **⚠️ REVISED AT IMPLEMENTATION — role-specific output layer.**
>
> *This paragraph was added while implementing the architecture (see `docs/02_model.md`, `src/steerable_t2l/hypernet.py::ModuleHead`). It revises, but does not replace, the "Weight tying across layers" paragraph above.*
>
> The claim above that **only 4 module-specific projection heads are learned** holds for the *shared-across-layers* axis exactly as written, but **cannot be taken literally across the A/B role axis.** The A and B matrices of the same module have different output widths:
>
> ```
> A head output width = r · in_features
> B head output width = r · out_features
> ```
>
> Under grouped-query attention these are not equal. For the target used in this implementation (Qwen2.5-1.5B-Instruct: 12 query heads, 2 key/value heads, `head_dim` 128, `r=8`):
>
> | module | `in_features` | `out_features` | A width | B width |
> |---|---|---|---|---|
> | `q_proj` | 1536 | 1536 | 12288 | 12288 |
> | `k_proj` | 1536 | **256** | 12288 | **2048** |
> | `v_proj` | 1536 | **256** | 12288 | **2048** |
> | `o_proj` | 1536 | 1536 | 12288 | 12288 |
>
> A single projection per module therefore cannot emit both roles, since one linear layer has exactly one output width. Note this is a property of the *target architecture*, not a flaw in the design: a target with `num_key_value_heads == num_attention_heads` (no GQA) and `num_heads · head_dim == hidden_size` would have all eight widths equal, and the literal reading would work unchanged.
>
> **Resolution as implemented.** Split only the final layer, preserving the intent (parameter sharing plus module-specific specialization):
>
> ```
> refined query (h)
>     ↓
> Linear(h → k)                ← shared per module, across BOTH roles and all layers
>     ↓
> Linear(k → r·in_features)    ← role A          }  per (module, role)
> Linear(k → r·out_features)   ← role B          }
> ```
>
> There are still exactly **4 module-specific projections** — the `h → k` bottlenecks, one per `q_proj`/`k_proj`/`v_proj`/`o_proj`, shared across all layers precisely as the paragraph above specifies. The role distinction is confined to the small `k → r·f` output layer. Because `k` is narrow the extra cost is minor: for the configuration above the eight output layers total ≈`11.1M` parameters, against ≈`1.0M` for the four shared bottlenecks.
>
> Role identity is still injected additively at query construction via `e_role`, exactly as layer identity is via `e_layer`. The split output layer only accommodates the width mismatch; it does not duplicate the role information.

---

### Output

The hypernetwork predicts the complete LoRA parameter set:

```
{
Layer 0:
    q_proj: A, B
    k_proj: A, B
    ...
Layer 31:
    ...
}
```

These predicted weights are directly injected into the target model without any gradient updates on the target model itself.

---

# Training

## Learnable Parameters

Training updates only

- backbone LoRA adapters
- query token embeddings
- parameter prediction heads

The pretrained backbone parameters remain frozen.

---

## Objective

Given

```
instruction

↓

hypernetwork

↓

predicted LoRA
```

the generated LoRA is inserted into the frozen target model.

The downstream task loss is backpropagated through the generated parameters into the hypernetwork.

No supervision on intermediate representations or parameter regression is required.

---

# Design Rationale

The pretrained LLM contributes

- semantic understanding of natural language instructions
- compositional reasoning over instruction contents
- strong language priors acquired during large-scale pretraining

The query tokens provide

- a fixed correspondence between hidden representations and LoRA parameter groups
- joint reasoning across all generated LoRA matrices
- scalable generation for arbitrary target transformer depths

The lightweight parameter heads specialize only in mapping contextual query representations into low-rank adapter weights, substantially reducing the amount of task-specific data required compared to learning an instruction encoder from scratch.

---

## Example

```
Instruction

"Answer math questions by explicitly verifying every intermediate calculation."

        │
        ▼

Pretrained Qwen2.5-3B

        │

Hidden(Q_L0_q_A)
Hidden(Q_L0_q_B)
...
Hidden(Q_L31_o_B)

        │

Linear Heads

        │

Predicted LoRA Parameters

        │

Frozen Target LLM

        │

Task Prediction
```

---

### Shared Parameter Decoder

A shared parameter decoder is used to generate LoRA parameters from contextual query representations.

Each query embedding first passes through a shared multi-layer perceptron, which learns a common transformation from semantic task representations into the LoRA parameter space. A lightweight module-specific projection layer then maps the shared representation into the final LoRA matrix.

Example:

```
Query embedding

↓

Shared MLP

↓

Module-specific projection head

↓

LoRA A / LoRA B matrix
```

The shared decoder enables parameter generation across different transformer layers and attention modules to share common structural knowledge, while the final projection layer preserves module-specific characteristics. This design reduces the number of independent parameters, improves scalability to larger target models, and encourages the hypernetwork to learn general parameter generation patterns rather than memorizing individual LoRA matrices.

# Data

The training dataset consists of triplets:

(steering instruction, task question, model response)

where the steering instruction specifies the desired behavior or reasoning strategy, the task question represents the downstream task instance, and the response corresponds to the expected model output under the given steering instruction.

Example:

The dataset covers diverse task domains, including but not limited to:

- mathematical reasoning
- code generation
- factual question answering
- summarization
- instruction following
- planning and reasoning
- scientific and technical QA
- domain-specific knowledge tasks

Each steering instruction is paired with multiple questions from the same task domain to encourage the hypernetwork to learn general behavioral adaptation rather than memorizing individual examples.

To improve instruction generalization, multiple semantically equivalent steering instructions are generated for each target behavior. For example:

```
Steering instruction:

"Verify every intermediate calculation before producing the final answer."

Question:

"If a train travels 120 km in 2 hours, what is its average speed?"

Response:

"The average speed is calculated as distance divided by time.

120 / 2 = 60 km/h.

Therefore, the average speed is 60 km/h."
```

---

## Training Pipeline

T2L is a hypernetwork that takes a natural-language task description and generates LoRA adapter weights for a base model, conditioned on a text embedding of the description. There are two ways to train it:

### 1. SFT Training (SSL-loss based)

- Trains the hypernetwork end-to-end by directly optimizing the downstream supervised fine-tuning loss on task data.
- The hypernetwork generates a LoRA on the fly for a given task description, and that adapter is applied to the base model to compute a standard language-modeling/SFT loss on training examples for the task.
- Gradients flow back through the generated adapter weights into the hypernetwork, so it learns to produce LoRAs that directly minimize task loss, without ever needing a "ground-truth" LoRA to imitate.
- Requires an asynchronous watcher process that periodically evaluates saved checkpoints on validation tasks and tracks the best one, since training itself doesn't directly optimize a validation metric.
- Relies on having a broad set of task descriptions and corresponding training data across many datasets.

#### 2. Reconstruction Training (regression on oracle LoRAs)

Two-stage process:

a. First, train a large collection of independent, single-task "oracle" LoRA adapters conventionally (one adapter per task), which serve as regression targets.

b. Then, train the hypernetwork to regress onto these oracle adapters — given a task's description, the hypernetwork predicts LoRA weights and is optimized to match the oracle weights via a reconstruction/distance loss, rather than through the downstream task loss.

**SVD-based canonicalization.** The decomposition `AB` is not unique: for any invertible `R`, `(AR)(R⁻¹B)` represents the same effective update as `AB`. Independently trained oracle adapters therefore land in arbitrary, mutually inconsistent bases, and regressing directly onto their raw `A`/`B` matrices gives the hypernetwork an ill-posed target — two oracles encoding the same function can have unrelated weights.

Before use as regression targets, each oracle adapter is canonicalized via SVD:

```
ΔW = A · B = U Σ V^T   (rank-r SVD)

A_canon = U Σ^(1/2)
B_canon = Σ^(1/2) V^T
```

This fixes a unique, consistent basis per adapter (up to sign/permutation of tied singular values, which is negligible in practice) and defines the reconstruction target as `(A_canon, B_canon)` rather than the raw oracle weights. The hypernetwork's predicted `(A, B)` is optimized against these canonical targets.

- Because it's a direct weight-matching objective, this approach can leverage multiple description paraphrases per task, encouraging the hypernetwork to generalize across varied phrasings of the same task.
- This is a form of amortized/meta-learning: the hypernetwork learns a mapping from task descriptions to a canonical LoRA weight space, rather than learning task behavior directly from data.

Common ingredient: both approaches use a text embedding model to encode task descriptions into a conditioning vector for the hypernetwork — the difference lies in what supervises the hypernetwork's output (task loss vs. weight-reconstruction loss against pretrained oracle adapters).