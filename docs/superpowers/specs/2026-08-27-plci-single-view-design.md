# Single-View PLCI-JEPA Design

## 1. Objective

Replace the current pattern-balanced auxiliary PLCI training view with the
official natural missing-modality view. Classification, Original GCNet
reconstruction, and PLCI latent prediction share one masked input and one GCNet
forward pass.

The experiment isolates one protocol variable:

```text
Dual-View PLCI:   natural GCNet forward + balanced auxiliary GCNet forward
Single-View PLCI: natural GCNet forward only
```

The source-anchored predictor, conditional innovation, student projectors,
explicit availability encoding, EMA teacher, GCNet backbone, reconstruction
head, and classifier remain unchanged.

## 2. Scientific Claim

The method tests whether latent prediction over the missing targets already
present in the official rate-matched view can regularize incomplete GCNet
representations without an independently sampled pattern distribution or a
second backbone forward.

It does not claim balanced coverage of all six incomplete patterns at every
missing rate. Pattern frequency is treated as part of the official benchmark
distribution.

## 3. Training Data Flow

For each batch, the existing deterministic conversation-keyed Natural schedule
produces availability `a_nat` and the corresponding masked feature tensor
`x_nat`. Dataset, split, fold, seed, epoch, side, and missing rate uniquely
determine the mask without consuming global RNG.

```text
complete frozen targets x_full
          │
          ├─────────────── EMA teacher projectors ──> target latents
          │
Natural mask schedule
          ↓
x_nat + a_nat
          ↓
student projectors and zero-init residual adapters
          ↓
Original pre-graph recurrent encoder + explicit pattern residual
          ↓
one shared Temporal/Speaker GCNet forward
          ↓
shared hidden h_nat
   ├─> emotion classifier
   ├─> Original linear reconstruction
   └─> source-anchored PLCI predictor
```

There is no balanced auxiliary sampler, auxiliary mask RNG, auxiliary feature
tensor, or second call to the GCNet hidden encoder.

## 4. Pattern Behavior

For natural patterns `A`, `T`, `V`, `AT`, `AV`, and `TV`, the predictor retains
the existing source-anchor, graph-context, and conditional-innovation formulas.

For `ATV`, no modality target is missing. The utterance contributes to emotion
classification but contributes zero PLCI targets and zero PLCI loss. In a mixed
conversation, `ATV` utterances remain in the shared temporal/speaker graph and
may provide context to incomplete utterances; they are skipped only by the PLCI
target enumerator.

For a complete batch at missing rate zero:

- the student adapter and pattern residual retain the existing exact Original
  bypass;
- PLCI loss is exactly zero;
- classification and reconstruction follow the Original path;
- the predictor does not execute any target path.

The explicit pattern residual remains in the first Single-View experiment.
Removing it simultaneously would confound the view change with a representation
change and is reserved for a later ablation.

## 5. Loss

The total training objective is:

```text
L = L_classification
  + lambda_rec * L_original_missing_reconstruction
  + lambda_J * L_PLCI_natural
```

`L_PLCI_natural` preserves the existing hierarchy:

1. average ordered paths within each missing target;
2. average missing targets within each utterance;
3. average only over utterances that have at least one missing target.

This avoids giving singleton patterns twice the utterance weight merely because
they contain two missing targets.

Teacher targets continue to use complete frozen features through a no-gradient
EMA projector. Complete targets never enter the masked student or GCNet forward.

## 6. EMA Update

The teacher remains outside the optimizer and in deterministic evaluation mode.
After each successful optimizer step:

```text
teacher = tau * teacher + (1 - tau) * updated_student
```

At missing rate zero the student projectors are bypassed, so the update is a
numerical no-op apart from the recorded step count.

## 7. Inference

Inference is unchanged from the existing natural PLCI path:

```text
official incomplete input
  -> student residual adaptation and explicit pattern residual
  -> GCNet
  -> emotion prediction
```

The EMA teacher and source-anchored predictor are not called. No modality is
generated or filled at test time.

## 8. Repository Boundary

The new protocol is represented by `gcnet_plci_single_view/`. It reuses the
already tested PLCI mechanism implementation rather than copying hundreds of
lines:

- `gcnet_plci_jepa.modules`: source-anchor, conditional innovation, and EMA;
- `gcnet_plci_jepa.loss`: hierarchical latent prediction loss;
- `gcnet_plci_jepa.model`: shared natural-view representation path;
- `gcnet_plci_single_view.model`: Single-View target selection and protocol
  boundary.

The shared trainer receives a distinct `plci-single` architecture choice. The
existing `plci` choice continues to mean Dual-View and must remain behaviorally
unchanged.

## 9. Implementation Constraints

Keep unchanged:

- datasets and splits;
- deterministic Natural mask schedule and conversation-to-mask mapping;
- mask rate during training and evaluation;
- GCNet graph construction and relations;
- temporal and speaker branches;
- Original linear reconstruction;
- classifier, optimizer, scheduler, epoch selection, and metric;
- PLCI predictor dimensions, caps, normalization, loss weight, and EMA rate.

Delete from the Single-View execution path:

- `sample_balanced_patterns`;
- `plci_aux_generator` and its checkpoint state;
- `auxiliary_source` construction;
- `forward_auxiliary` invocation;
- the second GCNet forward.

Do not delete these from the Dual-View implementation branch.

## 10. Required Regression Tests

Only tests that prove the changed protocol are required before the first
experiment:

1. One training batch calls `encode_hidden` exactly once.
2. Natural availability is passed unchanged to the PLCI target selector.
3. `ATV` utterances are skipped by PLCI loss without being removed from GCNet.
4. All-`ATV` forward, reconstruction, and hidden outputs remain exactly equal to
   Original for copied shared weights, and PLCI loss is zero.
5. Changing a naturally missing target feature changes only the teacher target
   and loss, not student latents, GCNet hidden, or prediction.
6. Gradients reach student projectors, predictor, and GCNet from an incomplete
   batch; teacher gradients remain absent.
7. Existing Dual-View PLCI tests remain green.
8. The Single-View manifest records the same train/validation/test mask schedule
   hashes as the inherited Dual-View control and does not record an auxiliary
   pattern sampler identity.

No repeated one-epoch smoke matrix is required after these tests pass. A single
GPU forward/backward integration check is the only runtime preflight.

## 11. Experiment Protocol

### Stage 1: discriminative run

- dataset: IEMOCAPSix;
- fold: 5;
- missing rates: `0.0`, `0.5`, `0.7`;
- seeds: `66, 67, 68, 69, 70`;
- mask source: the same deterministic Natural schedule used by Dual-View;
- primary control: inherited Dual-View PLCI, never retrained;
- secondary context: inherited Original, never retrained and not used as the
  Single-vs-Dual protocol gate;
- scheduling: at most three tasks per healthy GPU;
- comparison: strictly paired to Dual-View by dataset, fold, seed, missing rate,
  evaluation lifecycle, and all three mask schedule config hashes.

Rate zero is a preservation control rather than an expected gain. Rates 0.5 and
0.7 test the active Single-View mechanism.

### Stage 2: expansion gate

Expand to all eight missing rates with five seeds only if Single-View retains or
improves the two nonzero Dual-View means without introducing collapse. Original
remains a contextual reference. Cross-dataset experiments remain blocked until
the IEMOCAPSix sweep is complete.

## 12. Failure Interpretation

- Improvement at 0.5/0.7 with about half the Dual-View runtime supports the
  claim that natural-view latent prediction is sufficient.
- Improvement only at one rate indicates rate-specific pattern dependence, not
  a general Single-View result.
- A material drop from the strictly paired Dual-View control rejects the
  simplification hypothesis; it does not validate the previously stopped
  cross-dataset sweep.
- A rate-zero change before training indicates an implementation regression.
  A rate-zero difference after training may arise from auxiliary gradients
  changing shared parameters and must not be described as fixed-parameter path
  inequivalence.

## 13. Deferred Directions

- Dual-View balanced pattern learning remains GitHub Issue #2.
- Complete-modality M3-JEPA transfer remains GitHub Issue #3.
- Predictor vectorization remains GitHub Issue #1 and is not mixed into the
  first Single-View A/B experiment.
