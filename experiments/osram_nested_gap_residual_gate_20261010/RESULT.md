# Nested Gap residual Gate: implementation check

INTERNAL DIAGNOSTIC ONLY

Status: all three authorized runs completed100 epochs. Source commit `1db4bda`.
`LAUNCH.json` is the historical startup snapshot, not current live status.
Completion provenance, selected metrics and `SUMMARY.json` record the final results.
Baseline code used in implementation checks: `288ab18`, original
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

## Completed three-seed results

All three runs completed100 epochs on biggpu GPU7. Every runner verified all eight
BEST checkpoints and selected prediction files,100-row training history, complete
recovery checkpoint, original evaluation mask equivalence and source snapshot
integrity. All20 expected artifacts per run have hashes in completion provenance.
Effective configuration was compared to each same-seed original Nested; only the
method name differs. Original Nested and Flat were not retrained.

Scores below use per-rate BEST checkpoints under the inherited Test-oracle
selection. Each seed's eight selected W-F1 scores are equally averaged; high
missing is the equal average of .5/.6/.7. Differences are percentage points.

| Seed | Original Nested eight-rate | Gate eight-rate | Difference | Original Nested high | Gate high | Difference |
|---|---:|---:|---:|---:|---:|---:|
| 66 | 80.992 | 80.651 | −0.341 | 76.077 | 76.025 | −0.052 |
| 67 | 80.851 | 80.356 | −0.494 | 75.497 | 76.081 | +0.584 |
| 68 | 79.624 | 79.669 | +0.045 | 74.181 | 74.241 | +0.060 |
| Mean | 80.489 | 80.225 | −0.264 | 75.252 | 75.449 | +0.197 |

| Missing rate | Original Nested W-F1 | Gate W-F1 | Difference |
|---|---:|---:|---:|
| .0 | 88.114 | 87.654 | −0.459 |
| .1 | 85.925 | 85.681 | −0.244 |
| .2 | 84.456 | 83.364 | −1.092 |
| .3 | 80.760 | 80.653 | −0.108 |
| .4 | 78.900 | 78.101 | −0.799 |
| .5 | 76.843 | 76.793 | −0.051 |
| .6 | 75.780 | 76.112 | +0.332 |
| .7 | 73.132 | 73.443 | +0.311 |

The original three-seed Flat reference remains80.559 eight-rate /75.594 high.
The Gate version is therefore also lower than Flat by0.334 /0.145 pp respectively.
These are internal results, not paper claims or evidence of statistical significance.

The Gate variant does not improve overall W-F1. High-missing gain is modest and
primarily from seed67, not consistent across all three seeds. Lower missing-rate
scores decline, including .0 where no Gap is active: joint training can change
the shared backbone/Flat even when the new branch is inactive at inference.
This observation does not isolate the causal reason for the performance change.
No extra ablation, inference, threshold scan or additional training was performed
to produce this summary; no further expansion is automatically started.

Completed UTC: seed66 2026-10-10 03:03:51, seed67 03:04:21, seed68 03:05:14.
Final full-model/trainable parameter count:13,758,427. The added Gate remains89,399
parameters. Runtime Gate diagnostics exist in the module, but the existing Flat
trainer does not serialize them into selected per-rate metrics; no selected
checkpoint Gate-distribution claim is made. Nonzero updates were directly verified
in the epoch6 recovery checkpoint at launch.

Recalculate without training/inference:

```bash
python experiments/osram_nested_gap_residual_gate_20261010/summarize.py
```

Remaining verification gaps: no separate Gate-off intervention, new no-Text subset
comparison, causal attribution or statistical significance assessment for these new
checkpoints. Reported W-F1 comes from completed original-protocol evaluation.
