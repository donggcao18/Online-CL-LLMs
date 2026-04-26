# Changes in `llama_prompt_new.py` — Paper Implementation Summary

**Paper:** *Exploiting Presentative Feature Distributions for Parameter-Efficient Continual Learning of Large Language Models*

This file documents all modifications made to the standard HuggingFace LLaMA implementation to support the paper's continual learning framework.

---

## 1. New Imports

Added at the top of the file (lines 34–41):

```python
import torch.distributed as dist
import torch.multiprocessing as mp

from cl_dataset import GaussianDistribution
from assets import merge_distributions
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
```

- `GaussianDistribution`: stores the running mean/variance of task feature distributions ($D_k^l$).
- `merge_distributions`: utility for combining distributions from different models (plug-and-play expansion).
- `flash_attn_*`: Flash Attention 2 kernel for efficient training.

---

## 2. New Class: `LoRALayer` — Task-Specific LoRA Blocks (Paper §3.2)

**Lines 83–117** — Added from scratch, not in standard LLaMA.

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha=1, lora_dropout=0.):
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = lora_alpha / r
        ...

    def forward(self, x):
        result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)
```

**Paper connection:** Each task $T_k$ gets its own LoRA block $(A_k, B_k)$ per layer. This implements:
$$B_j^l A_j^l h_l(x)$$

Initialization: $A$ is Kaiming uniform, $B$ is zeros — so the block starts at zero output and grows during training.

---

## 3. `LlamaAttention.__init__` — CL State Added via `prompt_config`

**Lines 203–280** — The constructor signature changes from `(config)` to `(config, prompt_config)`. All CL-specific state is injected here.

### 3.1 Current Task LoRA Blocks

```python
self.lora_q = LoRALayer(hidden_size, num_heads * head_dim, r=..., lora_alpha=..., lora_dropout=...)
self.lora_v = LoRALayer(hidden_size, num_heads * head_dim, r=..., lora_alpha=..., lora_dropout=...)
```

Applied to the **Q** and **V** projections. K projection is left unmodified.

### 3.2 Current Task Feature Distributions

```python
self.distribution_q = GaussianDistribution()
self.distribution_v = GaussianDistribution()
```

Running mean of Q/V projected features for the current task — implements $D_k^l = \mathbb{E}[W^l h^l(x_k)]$.

### 3.3 Previous Task LoRA Blocks (Frozen)

```python
self.previous_lora_weights_q = nn.ModuleList()  # loaded from previous_lora_path
self.previous_lora_weights_v = nn.ModuleList()
```

Loaded from checkpoint paths (comma-separated). All weights are frozen (`torch.no_grad()`).

### 3.4 Previous Task Distributions (Frozen)

```python
self.previous_lora_distribution_q = []  # list of GaussianDistribution
self.previous_lora_distribution_v = []
```

Loaded from `previous_lora_distribution_path`. Used as keys for similarity matching.

### 3.5 Routing Weight State

```python
self.key_attention_weights_q = None   # computed α_j for Q
self.key_attention_weights_v = None   # computed α_j for V
self.log_key_attention_weights_q = None  # optional logging buffer
self.log_key_attention_weights_v = None
```

### 3.6 Similarity Metric Config

```python
self.distances_way = prompt_config['distances_way']          # 'L2', 'Cosine', 'Gaussian', 'Attention'
self.distances_temperature = prompt_config['distances_temperature']  # softmax temperature T
```

### 3.7 Top-K / Top-P Selection Config

```python
self.train_key_weight_top   = prompt_config["train_key_weight_top"]    # K during training
self.test_key_weight_top    = prompt_config["test_key_weight_top"]     # K during inference
self.train_key_weight_top_p = prompt_config["train_key_weight_top_p"]  # p during training
self.test_key_weight_top_p  = prompt_config["test_key_weight_top_p"]   # p during inference
```

Different thresholds for train and test allow tuning routing sparsity independently.

---

## 4. New Method: `agg_lora_states` — Weighted LoRA Combination (Paper §5.1)

**Lines 281–295**

```python
def agg_lora_states(self, hidden_states, lora_layer, pre_lora_layer, key_attention_weights):
    cur_lora_states = lora_layer(hidden_states).unsqueeze(0)
    pre_lora_states = torch.cat([pre_lora(hidden_states).unsqueeze(0) for pre_lora in pre_lora_layer], dim=0)
    concat = torch.cat([cur_lora_states, pre_lora_states], dim=0)  # [num_tasks, B, seq, dim]
    # reshape and weighted sum
    agg = torch.matmul(key_attention_weights.transpose(1,2), concat_reshaped).squeeze()
    return agg.reshape(bsz, -1, self.num_heads * self.head_dim)
```

**Paper connection:** Implements the dynamic combination:
$$\hat{h}_l(x) = W_l h_l(x) + \sum_{j=1}^{k} \alpha_j B_j^l A_j^l h_l(x)$$

Previous LoRA outputs are computed under `torch.no_grad()` (frozen).

---

## 5. New Method: `calculate_distances` — Similarity-Based Task Matching (Paper §4)

**Lines 299–340**

Computes $\Phi(x, D_j^l)$ for each stored distribution and returns softmax-normalized weights $\alpha_j$:

| `distance_type` | Formula | Softmax sign |
|---|---|---|
| `'L2'` | $-\|f^l(x) - D_j^l\|_2$ | Negate before softmax |
| `'Cosine'` | $\cos(f^l(x),\, D_j^l)$ | Standard softmax |
| `'Attention'` | $f^l(x) \cdot D_j^l / \sqrt{d}$ | Standard softmax |
| `'Gaussian'` | Log-likelihood under $\mathcal{N}(\mu_j, \sigma_j^2)$ | Standard softmax |

Returns shape `[B, num_tasks, 1]` (ready for `matmul` with stacked LoRA outputs).

```python
α_j = softmax(Φ(x, D_j^l) / T)
```

All computation is wrapped in `torch.no_grad()`.

---

## 6. New Method: `top_k_weights` — Top-K Sparse Selection (Paper §5.2)

**Lines 343–350**

```python
def top_k_weights(self, key_weights, top_k):
    topk_values, topk_indices = torch.topk(key_weights.squeeze(-1), top_k, dim=1)
    topk_weights = torch.zeros_like(key_weights)
    topk_weights.scatter_(1, topk_indices.unsqueeze(-1), topk_values.unsqueeze(-1))
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    return topk_weights
```

Keeps only the K highest $\alpha_j$, zeros the rest, then re-normalizes. Prevents irrelevant task LoRA blocks from contributing noise.

---

## 7. New Method: `top_p_weights` — Nucleus Filtering (Paper §5.2)

**Lines 354–428**

```python
def top_p_weights(self, key_weights, top_p, norm=True):
    # Sort α_j descending
    # Compute cumulative sum
    # Find cutoff index where cumsum >= top_p
    # Zero out weights beyond cutoff
    # Re-normalize
    return top_p_weights  # shape [B, K, 1]
```

Nucleus filtering adapted to task routing: includes the minimum set of tasks whose combined weight exceeds `top_p`. More adaptive than top-K when task count varies.

---

## 8. New Method: `updata_distribution_q` — Feature Distribution Update (Paper §3.3, §6)

**Lines 431–462** — Called during **training only**.

### Per-sample update (local GPU):

```python
for each_q, each_ids_w, each_ids in zip(hidden_states, ...):
    each_q = self.q_proj(each_q)           # project to Q space
    each_q = each_q[input_start:input_end]  # strip padding + label tokens
    each_q = torch.mean(each_q, dim=0)     # average over tokens → feature vector
    self.distribution_q.update(each_q)    # update running mean
```

The token span `[input_start:input_end]` is computed from `input_ids` and `input_ids_wo_label` to exclude padding and label tokens — only the input context is used.

### Multi-GPU synchronization:

```python
torch.distributed.all_gather_object(all_gpu_up_q_list, each_gpu_up_q)
for t != local_rank:
    for row in all_gpu_up_q_list[t]:
        self.distribution_q.update(row.to(f'cuda:{local_rank}'))
```

Gathers per-sample feature vectors from all GPUs and updates the local distribution copy — ensuring $D_k^l$ is consistent across data-parallel ranks.

**`updata_distribution_v`** (lines 465–496) is identical but uses `v_proj`.

---

## 9. New Method: `calculate_key_attention_weights_q` — Routing at Forward Pass (Paper §7)

**Lines 500–523** — Called when `previous_lora_weights_q` is loaded (task 2+) and sequence length > 1 (not autoregressive decode step).

```python
# 1. Project and average input features (same span logic as distribution update)
key_q = mean(q_proj(hidden_states)[input_span])  # [B, D]

# 2. Compute similarity to all task distributions
self.key_attention_weights_q = calculate_distances(
    key_q, [self.distribution_q] + self.previous_lora_distribution_q, ...
)  # [B, num_tasks, 1]

# 3. Apply Top-K or Top-P if configured
if train and train_key_weight_top > 0:
    self.key_attention_weights_q = top_k_weights(...)
elif test and test_key_weight_top > 0:
    self.key_attention_weights_q = top_k_weights(...)
# same pattern for top_p

# 4. Optional logging
if self.log_key_attention_weights_q is not None:
    self.log_key_attention_weights_q.append(...)
```

**`calculate_key_attention_weights_v`** (lines 526–549) is identical for V.

---

## 10. Modified: `LlamaAttention.forward` — Dynamic Routing in Standard Attention

**Lines 553–660** — Key changes from standard LLaMA forward:

```python
# 1. Update current task distribution (training only)
self.updata_distribution_q(hidden_states, input_ids_wo_label, input_ids)

# 2. Routing weights: reuse from previous layer OR recompute
if past_key_attention_weights_q is not None:
    self.key_attention_weights_q = past_key_attention_weights_q
else:
    self.calculate_key_attention_weights_q(hidden_states, ...)

# 3. Query states: standard + weighted LoRA combination
if self.key_attention_weights_q is not None:
    query_states = q_proj(x) + agg_lora_states(x, lora_q, prev_lora_q, weights)
else:
    query_states = q_proj(x) + lora_q(x)  # single task, no routing needed

# key_states: unchanged (no LoRA on K)
key_states = self.k_proj(hidden_states)

# 4. Same pattern for value states
self.updata_distribution_v(...)
self.calculate_key_attention_weights_v(...)
if self.key_attention_weights_v is not None:
    value_states = v_proj(x) + agg_lora_states(x, lora_v, prev_lora_v, weights)
else:
    value_states = v_proj(x) + lora_v(x)
```

New forward parameters:
- `input_ids` — needed for token span detection
- `input_ids_wo_label` — needed for label stripping
- `attention_mask_flash` — Flash Attention mask
- `past_key_attention_weights_q/v` — weights inherited from previous layer

---

## 11. New Class: `LlamaFlashAttention2` — Flash Attention Variant

**Lines 668–897** — Subclass of `LlamaAttention`. Inherits all CL logic from the parent constructor. Only the `forward` method differs — it feeds Q/K/V into the Flash Attention kernel instead of standard scaled dot-product attention.

The entire routing pipeline (`updata_distribution_*`, `calculate_key_attention_weights_*`, `agg_lora_states`) is **identical** to `LlamaAttention.forward`.

Key addition in `LlamaModel.__init__`:
```python
if prompt_config['flash_attention'] == True:
    self.self_attn = LlamaFlashAttention2(...)
else:
    self.self_attn = LlamaAttention(...)
```

---

## 12. Modified: `LlamaDecoderLayer` — CL Signal Propagation

**Lines 898–980** — Constructor and forward signature updated:

```python
def __init__(self, config, prompt_config):  # added prompt_config
    ...
    self.self_attn = LlamaFlashAttention2 or LlamaAttention  # based on flash_attention flag

def forward(self, ..., input_ids, input_ids_wo_label, attention_mask_flash,
            past_key_attention_weights_v, past_key_attention_weights_q):
    ...
    hidden_states, attn_weights, kv = self.self_attn(
        ...,
        input_ids=input_ids,
        input_ids_wo_label=input_ids_wo_label,
        attention_mask_flash=attention_mask_flash,
        past_key_attention_weights_v=past_key_attention_weights_v,
        past_key_attention_weights_q=past_key_attention_weights_q
    )
```

---

## 13. Modified: `LlamaModel.__init__` — Successor Layer Logic

**Lines 1100–1120** — Added:

```python
self.successor = prompt_config['successor']   # set of layer indices where weights are recomputed
```

`successor` defines which layers compute fresh routing weights. Other layers inherit the weights from the previous layer — avoiding redundant distance computation in back-to-back similar layers.

---

## 14. Modified: `LlamaModel.forward` — Layer Loop with Routing Weight Sharing

**Lines 1262–1300** — The decoder loop gains the successor logic:

```python
for idx, decoder_layer in enumerate(self.layers):
    if self.successor is None:
        past_key_attention_weights_v = None
        past_key_attention_weights_q = None
    else:
        if idx in self.successor:
            # This layer recomputes routing weights fresh
            past_key_attention_weights_v = None
            past_key_attention_weights_q = None
        else:
            # Inherit routing weights from the previous layer
            past_key_attention_weights_v = self.layers[idx-1].self_attn.key_attention_weights_v
            past_key_attention_weights_q = self.layers[idx-1].self_attn.key_attention_weights_q

    layer_outputs = decoder_layer(
        ...,
        past_key_attention_weights_v=past_key_attention_weights_v,
        past_key_attention_weights_q=past_key_attention_weights_q
    )
```

Also added `input_ids_wo_label` as a new parameter threaded through the entire model forward.

---

## 15. Modified: `LlamaForCausalLM` — `input_ids_wo_label` Plumbing

**Lines 1338–1510** — Both `forward` and `prepare_inputs_for_generation` accept and forward `input_ids_wo_label`:

```python
def forward(self, ..., input_ids_wo_label=None):
    outputs = self.model(..., input_ids_wo_label=input_ids_wo_label)

def prepare_inputs_for_generation(self, ..., **kwargs):
    input_ids_wo_label = kwargs.get("input_ids_wo_label", None)
    model_inputs.update({"input_ids_wo_label": input_ids_wo_label, ...})
```

This is essential for generation: the model needs `input_ids_wo_label` at every decode step to correctly identify which tokens form the "input context" vs. generated tokens, for accurate routing weight computation.

---

## Summary Table

| Paper Concept | Code Location | Key Attribute/Method |
|---|---|---|
| Task-specific LoRA blocks | `LoRALayer` class | `lora_A`, `lora_B`, `forward` |
| Current task LoRA | `LlamaAttention.__init__` | `self.lora_q`, `self.lora_v` |
| Previous task LoRA (frozen) | `LlamaAttention.__init__` | `self.previous_lora_weights_q/v` |
| Feature distribution $D_k^l$ | `LlamaAttention.__init__` | `self.distribution_q/v` |
| Previous distributions (frozen) | `LlamaAttention.__init__` | `self.previous_lora_distribution_q/v` |
| Distribution incremental update | `updata_distribution_q/v` | `GaussianDistribution.update()` |
| Multi-GPU distribution sync | `updata_distribution_q/v` | `all_gather_object` |
| Similarity routing $\Phi(x, D_j^l)$ | `calculate_distances` | L2 / Cosine / Attention / Gaussian |
| Softmax routing weights $\alpha_j$ | `calculate_distances` | `F.softmax(dist / T)` |
| Top-K sparse selection | `top_k_weights` | `torch.topk` + renormalize |
| Top-P nucleus selection | `top_p_weights` | Cumsum cutoff + renormalize |
| Weighted LoRA combination | `agg_lora_states` | `torch.matmul(α, LoRA outputs)` |
| Routing weight computation | `calculate_key_attention_weights_q/v` | Called per forward pass |
| Layer routing weight reuse | `LlamaModel.forward` | `successor` index set |
| Flash Attention support | `LlamaFlashAttention2` | Inherits all CL logic |
| Label-aware feature span | All distribution/routing methods | `input_ids_wo_label` |
