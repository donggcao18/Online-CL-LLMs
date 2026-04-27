# LoRA Dead Gradient Postmortem

## Symptom

During training of Qwen2.5-Coder-1.5B with a custom hand-injected LoRA on 8 code tasks sequentially, the loss did not decrease and all LoRA gradients appeared to be zero:

```
[DEBUG step=0]   lora_q.lora_A: data_norm=0.00000000, grad_norm=None
[DEBUG step=0]   lora_q.lora_B: data_norm=0.00000000, grad_norm=None
[DEBUG step=100] lora_q.lora_A: data_norm=0.00000000, grad_norm=None, data_norm_drift=0.0
{'loss': 2.1224, 'learning_rate': 0.0001, 'epoch': 0.32}   ← constant, not decreasing
```

---

## Investigation

### False lead: ZeRO-2 `grad_norm=None`

Under DeepSpeed ZeRO-2 with `reduce_scatter=true` and `contiguous_gradients=true`, `param.grad` is `None` or zero after backward. This is **not** a real zero gradient — DeepSpeed has already consumed it into its internal partitioned flat gradient buffer. This was a red herring.

### Real indicator: `data_norm=0` for `lora_A`

A parameter's `data_norm` drifting over steps is the reliable indicator of whether it is being updated. Both `lora_A` and `lora_B` had `data_norm=0` at step 0 **and** step 100 with zero drift — confirming no updates were happening at all.

### Chain of causation

```
lora_A = 0  →  LoRA output = dropout(x) @ A.T @ B.T = 0
            →  grad_B = output_grad @ x @ A.T = 0   (A is zero)
            →  grad_A = B.T @ ... = 0^T @ ... = 0   (B is zero from init)
            →  neither parameter ever updates
```

The LoRA was completely dead from step 0 due to `lora_A` being all-zeros.

---

## Root Cause

### HuggingFace `_init_weights` re-zeroing `lora_A` after `reset_parameters()`

`LoRALayer.__init__` constructed `lora_A` with `torch.zeros(...)` and then called `reset_parameters()` to apply `kaiming_uniform_`. This order is correct in isolation.

However, `Qwen2PreTrainedModel.from_pretrained` calls `_init_weights` on **every submodule** after the model is constructed. The original `_init_weights` had no branch for `LoRALayer`:

```python
# BROKEN — no LoRALayer branch
def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        ...
    elif isinstance(module, nn.Embedding):
        ...
    # LoRALayer falls through — no init applied, lora_A stays as torch.zeros
```

Because `LoRALayer` is not an `nn.Linear`, it falls through silently. The `reset_parameters()` call inside `LoRALayer.__init__` **does run**, but `_init_weights` is called **after** `__init__` completes during the `post_init()` → `apply(self._init_weights)` sweep triggered by `from_pretrained`. Since `lora_A` is declared as `nn.Parameter(torch.zeros(...))`, the HuggingFace init sweep visits the module, finds no matching branch, and leaves `lora_A` at the zeros value it was created with — the kaiming_uniform values written by `reset_parameters()` in `__init__` are effectively overwritten by nothing (the zeros from construction are what persist after the sweep).

The exact sequencing inside HuggingFace is:
1. `Qwen2ForCausalLM.__init__` runs → `LoRALayer.__init__` runs → `reset_parameters()` applies kaiming_uniform to `lora_A`
2. `post_init()` is called at the end of `__init__`
3. `post_init()` calls `self.init_weights()` → `self.apply(self._init_weights)` visits every submodule
4. `_init_weights(lora_layer)` has no `LoRALayer` branch → no-op → `lora_A` retains whatever value it had from `torch.zeros(...)` (step 1's `reset_parameters` ran in `__init__`, but the `apply` sweep in step 3 calls `_init_weights` again, not `reset_parameters` directly — so because the sweep does nothing for `LoRALayer`, the parameter keeps the zeros from the `nn.Parameter(torch.zeros(...))` declaration)

---

## Fix

Two changes were made to `qwen_prompt_new.py`:

### 1. Add an explicit `LoRALayer` branch to `_init_weights`

```python
def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, LoRALayer):
        module.reset_parameters()          # ← kaiming_uniform_(lora_A), zeros_(lora_B)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
```

This ensures the HuggingFace init sweep explicitly calls `reset_parameters()` on every `LoRALayer`, applying the correct kaiming_uniform initialization to `lora_A`.

### 2. Change `lora_A` initial allocation from `zeros` to `empty`

```python
# Before
self.lora_A = nn.Parameter(torch.zeros((r, in_features)))

# After
self.lora_A = nn.Parameter(torch.empty((r, in_features)))
```

Using `torch.empty` means the parameter never has a well-defined zero state between allocation and `reset_parameters()`. This eliminates the window where a missed init branch can leave garbage zeros.

---

## Verification

After the fix, training output confirmed both parameters were alive from step 0:

```
[DEBUG step=0]   lora_q.lora_A: data_norm=2.31014657,  data_norm_drift=0.0
[DEBUG step=0]   lora_q.lora_B: data_norm=0.01165585,  data_norm_drift=0.0116   ← updating from zero as designed
[DEBUG step=100] lora_q.lora_A: data_norm=2.41170359,  data_norm_drift=0.1015   ← growing
[DEBUG step=100] lora_q.lora_B: data_norm=0.37091342,  data_norm_drift=0.3709   ← growing fast

loss: 1.2966 → 0.6396 → 0.5275 → 0.5022 → 0.4612   ← decreasing normally
```

---

## Lessons Learned

| Observation | Meaning |
|---|---|
| `param.grad is None` after backward with ZeRO-2 | **False negative** — DS consumed the grad into its flat buffer. Not a real zero gradient. |
| `data_norm=0` with `data_norm_drift=0` over steps | **Real indicator** — the parameter is not being updated at all. |
| `_init_weights` has no branch for a custom `nn.Module` | HuggingFace silently skips it. The `reset_parameters()` you called in `__init__` may have been undone or never persisted past the `apply` sweep. |
| `nn.Parameter(torch.zeros(...))` + custom init | Risky pattern. Use `torch.empty` to make it obvious when init was not applied. |
