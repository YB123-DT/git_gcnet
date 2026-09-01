# Mask-Aware SAM Backbone for Missing-M3

## Objective

Replace the GCNet/SDR conversation core with a compact backbone adapted from
SAM-LML's intra-modal and directed inter-modal attention. The first gate is a
complete-modality CMU-MOSI experiment. Missing-M3 is attached only after the
new backbone proves that it can improve the complete-modality representation.

The implementation must use the existing GCNet feature package:

- audio: `wav2vec-large-c-UTT` (512D);
- text: `deberta-large-4-UTT` (1024D);
- visual: `manet_UTT` (1024D).

It must not add or fine-tune a second language encoder.

## Source Mechanism and Adaptation Boundary

The primary anchor is *Supervised Attention Mechanism for Low-quality
Multimodal Data* (SAM-LML, EMNLP 2025). The source model applies intra-modal
self-attention and directed inter-modal attention to temporal feature
sequences. It reports complete-modality Acc-2/F1 of 89.2/89.1 on CMU-MOSI and
87.9/87.9 on CMU-MOSEI.

This adaptation does not claim to reproduce those numbers or faithfully copy
the full source model. The source model operates on lower-level aligned
sequences and uses several auxiliary supervisory losses. Our benchmark stores
one frozen vector per modality per utterance. The transferable primitive is
therefore the attention topology, applied over the utterance sequence of each
conversation.

The first implementation intentionally excludes SAM-LML's modality
decomposition, mutual-information objective, reconstruction objective,
variance/context constraints, and noise-ranking losses. They cannot be added
until the backbone-only result is known.

## Architecture

### Inputs

For a padded batch:

- modality features: `A [L,B,512]`, `T [L,B,1024]`, `V [L,B,1024]`;
- availability: `M [L,B,3]`;
- utterance validity: `umask [B,L]`.

Missing feature blocks remain exactly zero. Availability is never inferred
from feature values.

### Modality temporal encoders

Each modality has an independent projection to shared width `d=120`:

```text
LayerNorm(input_dim) -> Linear(input_dim, d) -> GELU -> Dropout
```

The projected sequence enters one lightweight pre-norm Transformer encoder
layer. Padding and missing modality positions are masked. A residual connection
preserves the projected observed feature.

This produces `H_A`, `H_T`, and `H_V`, each shaped `[L,B,d]`.

### Directed cross-modal interaction

The backbone owns six candidate directed attention paths:

```text
A -> T, T -> A, A -> V, V -> A, T -> V, V -> T
```

For `m -> n`, an observed token from modality `m` is the query and attends to
observed `n` tokens in the same conversation. Missing or padded `n` tokens are
excluded as keys and values. A missing local query never creates an output
track.

The paths are candidate parameterizations, not six unconditional operations:

- ATV activates all local modality pairs;
- AT, AV, and TV activate their corresponding two directed local paths;
- A, T, or V retains its unimodal temporal track even when no local pair is
  available.

Conversation context may use another utterance's genuinely observed modality.
For example, an A-only utterance may use observed T tokens from neighboring
utterances through `A -> T`. It may not read the missing local T feature.

Each directed result uses pre-norm residual attention:

```text
cross_mn = H_m + MultiHeadAttention(query=H_m, key=H_n, value=H_n)
```

No synthetic Gaussian feature, learned missing token, or predicted latent is
inserted into a key/value stream.

### Mask-aware interaction pooling

At each utterance, the available tracks consist of:

- its observed unimodal tracks;
- valid directed cross-modal tracks whose query is locally observed and which
  have at least one observed key in the conversation.

A learned query performs attention pooling over those tracks. Invalid tracks
receive `-inf` before softmax. The pooled representation passes through a
pre-norm feed-forward residual block and becomes the emotion representation
`h [L,B,d]`.

The prediction head is:

```text
LayerNorm(d) -> Linear(d, d) -> GELU -> Dropout -> Linear(d, 1)
```

CMU-MOSI retains the existing regression target and MSE training loss. Acc-2
and weighted F1 exclude zero-labelled samples using the existing official
evaluation function.

## Missing-M3 Integration Boundary

The complete-modality gate uses only the SAM-style backbone and emotion head.
It does not instantiate the Missing-M3 predictor or EMA teacher.

If the complete-modality gate passes, the existing Missing-M3 components are
attached without changing their target objective:

```text
official incomplete input
        -> mask-aware SAM backbone -> emotion head
                                \-> Missing-M3 predictor -> EMA target loss
```

The predictor receives the backbone representation and observed student
latents. It remains training-only in the first missing-modality adaptation and
does not feed predicted latents back into the backbone.

## Experimental Gate

### Stage 1: complete-modality backbone discrimination

- dataset: CMU-MOSI;
- missing rate: 0.0;
- seeds: 66, 67, 68, 69, 70;
- official train/validation/test split;
- model selection: best validation loss, followed by one associated test
  report;
- features and labels: current GCNet package;
- inherited controls: reuse saved Original GCNet and current Missing-M3
  results; do not retrain them;
- checkpoint policy: save metrics and the best checkpoint only.

Pass criteria:

1. mean non-zero test weighted F1 exceeds the strongest existing single-model
   complete-modality result in the repository;
2. at least three of five seeds improve over the paired inherited control;
3. no seed has constant predictions, one-class binary predictions, NaN/Inf, or
   a validation/test selection mismatch;
4. the improvement is not obtained by selecting the best test epoch.

If Stage 1 fails, close this backbone without running missing rates.

### Stage 2: missing-modality adaptation

Only after Stage 1 passes:

- attach the existing Missing-M3 training-only predictor and EMA targets;
- train one mixed-rate model using the already locked mixed-rate protocol;
- evaluate rates 0.0 through 0.7 with the same fixed mask bank;
- start with five seeds;
- inherit all applicable controls.

## Required Tests

1. Shape and finite forward/backward for variable conversation lengths.
2. Missing and padding key/value positions receive zero attention probability.
3. Changing a locally missing feature value does not change any valid output.
4. A singleton-pattern utterance retains a finite unimodal path.
5. AT/AV/TV activate only valid local pair tracks.
6. A query may use an observed neighbor modality but never a missing neighbor.
7. All-missing effective conversations are rejected; padding stays zero.
8. GPU FP32 train step and parameter-count/runtime logging.
9. Metric parity with the existing MOSI non-zero Acc-2/W-F1 implementation.
10. Checkpoint selection uses validation loss and records the associated test
    epoch rather than the maximum test score.

## Repository Layout

The completed variant lives in a new isolated package:

```text
gcnet_missing_m3_sam_backbone/
  __init__.py
  attention.py
  model.py
  train_mosi.py
  run_mosi.py
  tests/
```

Existing GCNet, Missing-M3, SDR, and raw-SDR files remain unchanged except for
shared utilities that are demonstrably generic and covered by regression tests.

## Explicit Non-goals

- no DeBERTa/RoBERTa dual encoder;
- no upstream feature fine-tuning;
- no graph branch in the SAM backbone;
- no synthetic feature completion at inference;
- no test-epoch model selection;
- no Original rerun;
- no complete set of missing-rate experiments before Stage 1 passes;
- no transplantation of every SAM-LML auxiliary loss into the first version.

## Main Risk

SAM-LML's published gain partly comes from word/frame-level temporal features
and auxiliary attention supervision. Our utterance-level adaptation may not
retain that gain. Stage 1 exists specifically to reject the backbone cheaply
before mixing it with Missing-M3. A negative result will not be interpreted as
a faithful reproduction failure of SAM-LML.
