# GCNet Full-Fused Reconstruction Design

## Objective

Test whether reconstructing the complete multimodal utterance state is a better
training signal than reconstructing only the missing modalities for IEMOCAPSix
emotion recognition.

The experiment isolates the loss target. The model architecture, parameter
count, input masks, initialization, optimizer, classifier, training budget, and
evaluation protocol remain identical between conditions.

## Conditions

### B0: Missing-only reconstruction

This is the existing GCNet reconstruction baseline. Given the masked input, the
existing `linear_rec` head predicts the concatenated Audio/Text/Visual feature.
The loss selects only modalities that are missing at each real utterance.

### FFR: Full-fused reconstruction

FFR uses the same GCNet and the same `linear_rec` output. For every real
utterance with at least one missing modality, it reconstructs all three complete
modalities:

```text
x_full = concat(x_audio, x_text, x_visual)
x_hat_full = linear_rec(h_masked)
```

The per-utterance loss is modality-balanced:

```text
L_full(t) = mean_m mean_dimension((x_hat_t^m - x_t^m)^2)
```

The batch loss averages `L_full(t)` over real utterances with at least one
missing modality. Padding and fully observed utterances do not contribute.

## Controlled variables

- Dataset: IEMOCAPSix.
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`.
- Protocol: existing official fold-5 evaluation.
- Missing rates: 0.0 through 0.7.
- Seeds: 66 through 75.
- Epochs: 100.
- Baseline and FFR load the same shared initialization for each
  missing-rate/seed pair.
- Baseline and FFR reuse the same train, validation, and test mask schedules.
- Classification consumes the masked GCNet hidden state, not reconstructed
  features (`reccls_flag=false`).
- No JEPA Predictor, EMA target, target projector, additional encoder, or new
  inference path is introduced.
- Both conditions instantiate exactly the same modules and total/trainable
  parameter counts.

The existing 80-run GCNet missing-only baseline is reused read-only. The new
formal matrix therefore contains 80 FFR training jobs:

```text
8 missing rates x 10 seeds x 1 new FFR condition
```

Each new result is paired with the already completed baseline result having the
same missing rate and seed. The runner must validate the baseline evidence but
must never overwrite or rerun it.

## Parity and failure gates

Before formal training:

1. A unit test must fail before implementation and then verify the full-fused
   loss selection and modality-balanced normalization.
2. The FFR loss must be zero when no real utterance has a missing modality.
3. Padding and fully observed utterances must not affect FFR loss.
4. Baseline and FFR parameter names, shapes, total count, and trainable count
   must match exactly.
5. At missing rate 0.0, baseline and FFR logits and gradients must match within
   numerical tolerance because both primary reconstruction losses are zero.
6. A one-epoch FFR smoke run and read-only baseline audit must complete before
   the 80-job sweep starts.

## Outputs and analysis

Each run stores command, status, fold metrics, manifest, mask hashes, shared
initialization hash, parameter counts, and training log in an isolated output
directory. The final report includes:

- ten-seed Weighted-F1 mean and sample standard deviation per missing rate;
- paired FFR-minus-baseline differences and seed-level win counts;
- paired Wilcoxon tests with Holm correction across eight missing rates;
- seed-collapse audit using the same robust rule as the completed 640-run
  analysis;
- explicit missing=0 parity evidence.

Primary results retain every seed. Any collapse-filtered summary is diagnostic
only.
