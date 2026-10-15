# ComP isolated adapter audit

The adapter uses the already implemented ComP causal path and its `temporal_causal=False` original attention/prompt path. No new causal design or model components were introduced. No GPU was used for these adapter checks and no remote write occurred.

## Sources

- Adapted repository: https://github.com/YB123-DT/git_gcnet, branch `feature/comp-meld`, commit `aa538318fda0075a48f7ba8474b08fd239bea0f2`.
- Read-only source: `/data2/yb/paper/GCNet_TPAMI/.worktrees/comp-meld/external_repos/ComP_causal/ComP` (tracked working tree clean when inspected).
- Official upstream: https://github.com/WenjueHE/2026-AAAI-ComP, commit `28192d3a5683543d7383e40898f9a98d1f114a08`.
- Official checkout: `/data2/yb/paper/05_reproduction/external_repos/ComP`; observed untracked `__pycache__` directories, no tracked source modifications.
- File-level source and vendored SHA256 values are recorded in `adapters/comp/SOURCE.json`.

Copied model, attention and loss files into `adapters/comp/vendor/` so future changes in the user's original checkout do not change this runner. The model import is package-relative. The only compatibility edit to copied model calculations replaces two `umask.view(-1)` operations with `umask.reshape(-1)`: a transposed mask otherwise raises a noncontiguous-stride RuntimeError for a batch of two in Original mode. This preserves the same flattening order and arithmetic.

The existing causal fork also already repairs an upstream Original-mode runtime defect: the official PromptFormer flattens its prototype tensor before returning it, whereas PromptFormer_2 expects a three-dimensional prototype. The fork retains its structured prototype and flattens a separate local copy for matrix multiplication. Original here means the original noncausal information access with this required compatibility repair. Report this provenance rather than claiming byte-identical unmodified upstream execution.

## Public interface

```python
from adapters.comp import build_model, stage_loss
model = build_model(config, (audio_dim, text_dim, visual_dim), nclasses, nspeakers)
output = model(a, t, v, availability, padding, first_stage=first_stage)
loss = stage_loss(output, labels, padding, first_stage)
loss.backward()
modulation_report = model.after_backward(output, labels, padding, first_stage)
optimizer.step()
```

- Features: batch-first float tensors `[B,L,D]`, in A/T/V order. Availability: `[B,L,3]`, `True` means observed. Padding: `[B,L]`, `True` means a real utterance.
- `config.seq_len` is mandatory and must be the fixed dataset maximum (do not change per batch). Adapter internally pads all tensors to this maximum to preserve fixed prompt-layer dimensions and returns logits truncated to original `[B,L,C]`. `aux` retains the full eleven-element original return tuple, and `full_padding` supports the original losses.
- Config keys: `hidden=256`, `depth=4`, `heads=2` (alias `num_heads`), `dropout=0.5` (alias `drop_rate`), `attention_dropout=0` (alias `attn_drop_rate`), `lbd=0.3` (alias `lambda`), `mlp_ratio=1`, `temporal_causal=True`. Set only `temporal_causal=False` for Original. Set dataset-specific official hyperparameters in frozen runner config.
- Seven, four or six class output is controlled by `nclasses`. Speaker IDs are not used by the upstream architecture; `nspeakers` is accepted without a fake two-speaker mapping.
- The adapter enforces zero input for every unavailable or padded modality using `torch.where`; changing unavailable underlying features does not affect outputs. It never passes real labels into model forward (dummy zero labels only).
- Fixed test mask generation, train mask sampling, optimizer creation, two-stage epoch schedule, fold split, genuine checkpoint snapshots, predictions, and Test-best Stage-2 selection are the unified runner's responsibility.

## Original training behavior preserved

Let `C_A,C_T,C_V` denote masked modality cross-entropy, `C_fused` fused masked CE, `R` the upstream projected-feature reconstruction MSE, and `D` the sum over the three modality pairs of `DisentangledLoss.foward_2` (absolute private/shared cosine penalties plus JSD).

- Stage 1: `sum(C_modality + CE(1 - private_logits)) + D`.
- Stage 2: `C_fused + 2 * (C_A + C_T + C_V) + R + D`.

The duplicated Stage-2 modality CE, unmasked reconstruction MSE across padded positions, JSD normalization, and Stage-1 private-head implementation are kept as in train_mr.py. The fused classifier is unused by Stage-1 loss; it starts training in Stage 2. This is not replaced by fused CE alone. The original Adam default is weight decay `1e-5`; configure optimizer in the runner.

Two existing source issues are reported without silently changing their effective behavior:

1. `train_mr.py` gradient modulation matches names such as `prop_former_a.proj_n.fc1`, but actual model names are `prop_former_a_1.proj_n.fc1`. It therefore matches **zero parameters**. `after_backward` preserves that no-op and reports the zero count. A corrected modulation algorithm would be a distinct implementation, not the existing causal reproduction.
2. Stage-transition code stores repeated references to the same mutable model and calls `model.state_dict().update(...)` on a temporary dictionary; it does not actually restore historical best unimodal weights. The adapter continues the actual final Stage-1 model into Stage 2. The runner must save independent checkpoints for the requested Test-best evaluation rather than reproducing the broken historical checkpoint bookkeeping.

Original and existing Causal prompts both pool prototypes across dialogues in the batch. This batch dependence is intentionally preserved and makes batch composition/order part of the protocol. The tests verify that Original still responds to another dialogue in the batch and to future features; neither of these effects was silently removed. Existing Causal restricts temporal access to each dialogue's prefix but remains batch-dependent.

## Verification

Executed on CPU with Python 3.8 / torch `1.12.0+cu113` in `comp-repro`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/yangbin/miniconda3/envs/comp-repro/bin/python -m unittest discover \
  -s experiments/baseline_5models_local_20261015/tests -p test_comp_wrapper.py -v
```

Result: **6 unittest methods passed**, including four Causal/Original × Stage-1/Stage-2 forward/backward cases; finite losses and all populated gradients; expected encoder/prompt parameter changes after an Adam step; fused-head gradient only in Stage 2; hidden/padding feature noninterference; label isolation; existing Causal future-feature and future-availability perturbation in both stages; causal prefix equivalence in both stages; Original future sensitivity and retained cross-batch prompt mixing. No scientific performance claim is made by these synthetic tests.

All formal results from the runner must say `checkpoint_selection=test_best_acc`, `evaluation_protocol=test_oracle`, Stage-2-only selection, and fixed evaluation mask. This is test-oracle selection with optimistic test-selection bias, not an unbiased independent test estimate.
