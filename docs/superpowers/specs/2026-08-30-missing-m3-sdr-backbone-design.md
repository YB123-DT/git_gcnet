# Missing-M3 SDR-GNN Whole-Backbone Replacement Design

Date: 2026-08-30
Status: approved architecture, pending written-spec review

## 1. Objective

Determine whether the conversation backbone is limiting the current Slot
Missing-M3 model on CMU-MOSI by replacing the complete GCNet conversation path
with SDR-GNN graph processing while leaving the incomplete-modality method and
benchmark protocol unchanged.

The experiment implements two explicitly named variants because the SDR-GNN
paper and its public code do not execute the same classifier path:

- `sdr-paper`: the paper-aligned Temporal and Speaker SDR branches are both
  used, concatenated, and projected to the existing Missing-M3 hidden width;
- `sdr-public`: only the Temporal SDR branch is used, matching the representation
  that the released code actually passes to its classifier.

Neither variant is allowed to be silently described as the other.

## 2. Evidence motivating the replacement

The independent public-code SDR-GNN reproduction uses the same frozen MOSI
features and split as the existing GCNet experiments.  The first locked trial
was higher than Original GCNet at seven of eight requested missing rates.  The
paper reports an eight-rate W-F1 mean of 78.9, versus 76.10 for the archived
ten-seed Original GCNet run.

This evidence motivates testing the SDR graph mechanism inside Missing-M3, but
does not establish causality: public SDR-GNN also changes the recurrent type,
graph layers, reconstruction objective, rate protocol, and hyperparameters.
The present experiment therefore changes only the conversation backbone.

An earlier equal-active-parameter full-context Transformer replacement failed
on all primary validation gates.  That result closes generic Transformer
replacement, not SDR-GNN's relational, hypergraph, and frequency-aware graph
mechanism.

## 3. Existing boundary

For each official incomplete view, the current model computes:

```text
frozen incomplete feature [L, B, 2560]
  -> Slot Observed-Set Encoder
  -> slot node [L, B, 256] and source latents {A,T,V}: [L,B,256]
  -> GCNet encode_hidden()
  -> conversation hidden [L, B, 500]
  -> shared MOSI regression head

conversation hidden + source latents + availability
  -> existing Contextual Missing-M3 Predictor
  -> training-only EMA latent prediction losses
```

The only replacement seam is `MissingM3GraphModel.encode_hidden()`.  The new
backbone contract is:

```python
forward(
    node_input,   # [L, B, 256]
    qmask,        # [B, L]
    umask,        # [B, L]
    seq_lengths,  # list[int]
) -> hidden       # [L, B, 500]
```

The subclass must remove inherited GCNet recurrent and graph modules so they do
not remain as dead registered parameters.

## 4. Shared SDR conversation core

Both variants use the same source-attributed SDR operations:

```text
slot node [L,B,256]
  -> 2-layer bidirectional GRU, hidden 200 per direction
  -> recurrent state [L,B,400]
  -> graphify each conversation with past/future window 2
  -> SDR relation branch
       RGCNConv(400 -> 100)
       -> SDR HypergraphConv(100 -> 100)
       -> SDR frequency-aware highConv(100 -> 100)
       -> concatenate recurrent node and graph node [500]
       -> 2-layer bidirectional GRU fusion
       -> Linear + ReLU
  -> branch hidden [L,B,500]
```

The implementation preserves the released RGCN, hypergraph normalization, and
frequency-aware neighbor gate formulas.  It may refactor names and tensor
packing for clarity, but a synthetic parity test must prove numerical equality
to the relevant public branch after copying weights.

The upstream repository does not contain a detected software license.  The new
package therefore reimplements the published equations and observed tensor
semantics instead of copying substantial source text verbatim.  The paper and
official repository are cited in module documentation, and the public code is
used only as a numerical oracle in tests.

Padding positions must be zero in the returned hidden.  Relation definitions,
edge directions, self edges, and graph window construction must not be changed.
Relation IDs are assigned by explicit ordered tables rather than enumeration of
a Python `set`; copied public RGCN weights are permuted in the parity fixture so
that the test compares relation semantics rather than accidental hash order.

## 5. Variant definitions

### 5.1 `sdr-public`

```text
BiGRU state
  -> Temporal SDR relation branch (past / now / future)
  -> hidden [L,B,500]
```

This variant matches the public classifier's effective representation.  In the
released implementation, each graph branch returns a one-element tuple;
`hidden1 + hidden2` concatenates tuples and `hidden0[0]` selects only Temporal.
The adaptation represents that behavior intentionally and directly, without
computing a discarded Speaker branch.

Reporting language: "public-code-effective SDR backbone," not "the complete
paper SDR-GNN."

### 5.2 `sdr-paper`

```text
BiGRU state
  -> Temporal SDR branch [L,B,500]
  -> Speaker SDR branch  [L,B,500]
  -> concatenate [L,B,1000]
  -> Linear(1000,500)
  -> ReLU
  -> hidden [L,B,500]
```

This follows the paper equation that concatenates Speaker and Context local
enhancement representations.  It repairs the tuple error explicitly rather
than inheriting it.  The projection is required solely to preserve the existing
Missing-M3 predictor and classifier interface.

On MOSI, the Speaker branch has one relation type because the dataset interface
contains one speaker.  It remains a learned graph transformation and cannot be
described as learning multi-speaker interaction on this dataset.

Reporting language: "paper-aligned corrected SDR backbone."  It is not claimed
to reproduce the released checkpoint or code numerically.

## 6. Deliberately excluded SDR components

The first experiment does not import:

- SDR's raw 2560-dimensional reconstruction head;
- its 256-head `MultiheadAttention` block;
- its reconstruction loss;
- its classifier;
- its public tuple addition;
- an extra completion or feedback path.

These components are outside the requested backbone-only intervention.  The
public MHA receives `[B,L,D]` while `batch_first=False`, so it attends across
conversations and lacks a padding mask.  Adding it would both reproduce a known
axis defect and duplicate Missing-M3's training-only missing-latent predictor.

The existing Missing-M3 EMA predictor remains training-only.  Evaluation must
continue to call the model with `predict_missing=False`.

## 7. Preserved controls

The following remain identical to the archived Slot Missing-M3 main result:

- CMU-MOSI official train/validation/test split;
- frozen `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, and `manet_UTT` features;
- Slot Observed-Set Encoder and explicit availability input;
- student projectors and EMA teacher update;
- Contextual Missing-M3 Predictor and target-specific MMoE;
- MOSI regression MSE and JEPA weight 0.1;
- all-rates-per-batch mixed-rate training;
- validation eight-rate mean W-F1 checkpoint selection;
- deterministic natural mask schedules and their SHA256 records;
- seeds 66, 67, 68, 69, and 70;
- hidden 200, latent 256, graph width 100, window 2/2;
- Adam learning rate `5e-4`, weight decay `1e-5`, dropout 0.5;
- 100 epochs, batch size 32, fold 1, and official W-F1 evaluation.

The existing five-seed GCNet control is inherited.  No Original task appears in
the new runner.

## 8. Experiment matrix and decision rules

The formal matrix contains ten new models:

```text
2 SDR variants x 5 seeds x 1 mixed-rate model = 10 jobs
each model -> test at requested rates 0.0, 0.1, ..., 0.7
```

No test metric may choose the variant, checkpoint, graph window, or any
hyperparameter.  The implementation is frozen before formal test execution.

Primary comparison uses paired validation eight-rate mean W-F1 against the
inherited GCNet control.  A variant is considered promising only if:

1. mean paired validation delta is positive;
2. at least three of five seed deltas are positive;
3. high-missing validation mean (`0.4--0.7`) does not decrease;
4. no non-finite loss, constant-sign output, or representation collapse occurs.

Test W-F1 is then reported descriptively from the validation-selected
checkpoints.  If a variant passes, parameter count is a competing explanation;
the next experiment must add a parameter-matched control before claiming an
SDR-specific mechanism gain.  Parameter matching is not part of this first
diagnostic.

## 9. Required implementation layout

Create an isolated branch and parallel package:

```text
gcnet_missing_m3_sdr_backbone/
  __init__.py
  layers.py          # attributed SDR graph layers
  model.py           # SDRConversationBackbone and MissingM3SDRModel
  train_gcnet.py     # thin reuse of existing Missing-M3 lifecycle
  run_mosi.py        # two variants x five seeds, resume and audit
  README.md
  tests/
results/sdr_backbone/  # compact Git-tracked summaries only
```

Raw checkpoints, prediction arrays, histories, and logs remain in the remote
experiment root.  Compact configuration, manifest, provenance, and result
tables are copied into Git after verification.

## 10. Required tests

Before training, tests must cover:

1. exact input/output shapes for padded multi-conversation batches;
2. padding invariance and zero returned padding;
3. Temporal/Speaker relation construction and node-order alignment;
4. copied-weight synthetic parity between the public Temporal branch and
   `sdr-public`;
5. `sdr-paper` concatenation/projection uses both branches and propagates
   gradients to both;
6. the public tuple bug cannot occur in either variant;
7. no inherited GCNet recurrent/graph modules remain registered;
8. Missing-M3 predictor receives hidden width 500 unchanged;
9. evaluation does not execute the predictor or teacher;
10. finite CPU/GPU FP32 forward and backward;
11. exact parameter counts for both variants and inherited GCNet control;
12. deterministic mask, model initialization, and result resume behavior;
13. runner contains exactly ten treatment jobs and no Original commands;
14. manifest binds source commit, source SHA, config SHA, feature paths, mask
    hashes, environment versions, and variant identity.

Tests use the established remote PyTorch environment directly.  Environment
discovery is performed once and recorded; it is not repeated per module.

## 11. Risks and interpretation

- The public SDR score may depend on its reconstruction auxiliary loss, so a
  backbone-only failure does not refute the complete SDR training recipe.
- `sdr-paper` has substantially more active parameters than GCNet and
  `sdr-public`; a raw gain is not yet a mechanism-isolated result.
- MOSI's one-speaker interface limits interpretation of the Speaker branch.
- The Slot node is 256-dimensional rather than SDR's raw 2560-dimensional
  concatenated feature.  This is an adaptation to Missing-M3, not a bitwise
  reproduction of the original model.
- The existing mixed-rate protocol is intentionally retained because the
  registered fixed-rate Missing-M3 experiment was 1.604 W-F1 points worse.

## 12. Completion condition

The task is complete only when both variants have verified code, all ten formal
jobs have complete 100-epoch histories and prediction artifacts, metrics have
been independently recomputed, the paired comparison is written, and code plus
compact results are committed and pushed to `git_gcnet`.  Failed scientific
results are still complete results and must not be hidden or retuned using test
scores.
