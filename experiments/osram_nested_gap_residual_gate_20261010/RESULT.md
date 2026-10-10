# Nested Gap residual Gate: implementation check

INTERNAL DIAGNOSTIC ONLY

Status: implementation and CPU correctness checks completed. No formal training
started; no W-F1 result is claimed. Baseline code: `288ab18`, original
`nested_gnn_rooted_evidence`, not the root-aware, Local8, replacement or small-Flat
variants. This change introduces a separately selectable implementation.

## Approved architecture

The original Nested GNN/tokenizer/head decoders remain unchanged. Insert a Gate
after decoding the Gap residual, before adding it back to the corresponding original
read. For each current utterance and modality m in A/T/V:

```
Gap_m_out = Gap_m_original + gate_m * Nested_delta_Gap_m
gate_m = 2 * sigmoid(MLP([P_L(Local), P_G(Gap_m), P_D(delta_Gap_m), type_m, availability]))
```

Each scalar is independent by modality/sample and shared across its eight heads.
The projections produce 64 dimensions each, modality identity embedding16, current
availability3. The shared gate network is LayerNorm211 → Linear(211,32) → GELU →
Linear(32,1). Only the last gate layer is zero-initialized: every active initial
Gate is exactly1, so even nonzero learned Nested residuals reproduce the original.
New module construction is isolated with `torch.random.fork_rng(devices=[])`:
old parameters and subsequent RNG state are preserved.

No overall Gap attenuation: the original Gap is not multiplied by the Gate.
No change to Local/Base correction, original Local Skip, Flat, task head,
OSRAM writes/queries/reads, losses or missing-mask schedules. The joint training
path remains available; no parameter is frozen by this module. No extra regularizer,
historical label input, JEPA/completion, persistent mix or second view is added.

Inactive Gate inputs and outputs use `torch.where` safety masking; decoded inactive
Gap correction is zero. The existing public adapter excludes first turns and empty
history, preserves the backward half, and masks padding. Initial gate gradients
can be zero while Nested's original zero-initialized decoders output zero; gate
updates become possible after nonzero corrections exist. This is expected rather
than evidence of a disconnected graph.

## Configuration and parameter count

Existing training argument accepts the separate method:

```text
--osram-meaningful-block nested_gnn_gap_residual_gate
```

Use this instead of the original `nested_gnn_rooted_evidence`, keeping all other
effective cfg84 options identical. Do not activate another OSRAM gate switch.
The original method/default paths retain their old architecture and state keys.

Cfg84 latent256, eight64-dimensional heads:

| Input block | Parameters |
|---|---:|
| Original Nested tokenizer/core/decoders | 159,235 |
| Added Gap Gate | 89,399 |
| New input block total | 248,634 |

This is the input block count, not the complete model count. Added parameters do
not scale with output_dim1600. If importing an original Nested checkpoint into
the new variant, explicitly check that only `gap_residual_gate.*` keys are missing;
do not silently allow other missing/unexpected keys or call this full resume.

## Diagnostics

Under the existing OSRAM `meaningful_block` diagnostics, `gap_residual_gate`
contains A/T/V active counts, active gate means, raw residual norm and gated
residual norm. Empty slots have count0 and mean0, meaning **not evaluated**, not a
learned rejection. Empty-history forwards clear diagnostics instead of reusing
statistics from a prior batch. These are internal magnitudes, not reliability or
emotion-shift probabilities.

## Verification

Tests added before production code; initial run failed on the absent registration
and factory, as expected. New tests cover:

- Separate registration and training CLI parsing.
  The actual parser function is extracted from source for the lightweight CPU
  test because local `msa_extract` lacks the optional `torch_geometric` training
  dependency; importing/running the full trainer is not covered by that check.
- Original parameter tensors and downstream RNG unchanged.
- Gate1 exact equivalence with explicitly nonzero decoders.
- Gate .5/.75/1.5 affects only Gap delta, preserving original Gap, Local and Base.
- Finite gradients and nonzero Gate updates over three ordinary optimizer steps.
- Public first-turn/padding/inactive masks, masked NaN/Inf safety and backward half.
- Old checkpoint import has only explicit Gate key additions; original strict load
  continues to work.

Command:

```bash
/home/yangbin/miniconda3/envs/msa_extract/bin/python -m pytest -q \
  tests/test_nested_gap_gate.py tests/test_nested_rootaware.py \
  tests/test_nested_input_gradient.py
```

Outcome: 23 tests passed, one existing environment warning. Python compilation
and `git diff --check` also passed.

Additional original sweep tests passed except a pre-existing catalog assertion:
`test_each_ablation_changes_only_its_named_mechanism` expects a historical10-method
list, but HEAD already contains14 methods. `git show HEAD:...` verified this mismatch
predates the Gate change. Its four existing additional entries are `nested_dim704`,
`nested_mlp256`, `nested_mlp512`, `nested_groups1_dim512`. That unrelated test/catalog
was not edited. The existing environment emits a `pynvml` deprecation warning.

Remaining verification gaps: no GPU execution, complete real-checkpoint prediction
replay, formal training or performance measurement for this new variant. No expected
W-F1 gain is asserted from the preceding gradient/prediction diagnostics.
