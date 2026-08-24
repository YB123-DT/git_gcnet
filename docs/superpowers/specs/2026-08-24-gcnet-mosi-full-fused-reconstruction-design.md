# GCNet MOSI Full-Fused Reconstruction Design

## Objective

Test whether the same Full-Fused reconstruction target evaluated on
IEMOCAPSix transfers to CMU-MOSI regression under incomplete multimodal input.
The experiment continues from the completed official GCNet baseline and trains
only the new Full-Fused condition.

## Controlled comparison

The model remains GCNet. Given an utterance with at least one missing modality,
the existing reconstruction head predicts the complete Audio/Text/Visual
feature tuple. The loss averages the per-modality dimension-normalized MSE:

```text
L_full(t) = mean_m mean_dimension((x_hat_t^m - x_t^m)^2)
```

Fully observed utterances and padding do not contribute. Reconstructed features
do not enter the classifier/regressor. No JEPA predictor, EMA target, additional
encoder, DFGCN component, or new inference path is introduced.

## Protocol

- Dataset: `CMUMOSI`.
- Evaluation: existing official fold 1 protocol.
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`.
- Missing rates: 0.0 through 0.7.
- Seeds: 66 through 75.
- Epochs: 100.
- Existing baseline root:
  `/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/CMUMOSI`.
- New result root:
  `/data2/yb/experiments/gcnet_mosi_full_fused_10seed_20260824`.
- Only 80 Full-Fused jobs are trained. Baseline artifacts are validated and
  read only.
- Exactly three idle non-GPU4 devices are used, with at most three jobs per GPU.

The runner is generalized with a constrained `--dataset` option supporting the
already validated default `IEMOCAPSix` and the new `CMUMOSI` path. Dataset,
expected fold, baseline root, command, manifest validation, identity, and output
evidence must remain internally consistent. Existing IEMOCAP behavior is locked
by regression tests.

## Verification gates

1. Tests must first fail for a CMUMOSI job matrix and command.
2. The default IEMOCAPSix matrix must remain 80 jobs and unchanged.
3. CMUMOSI must create exactly 80 Full-Fused jobs and no baseline subprocess.
4. CMUMOSI commands must omit `--fold` and manifests must report fold 1.
5. Baseline preflight must validate all 80 existing MOSI baseline artifacts.
6. A one-epoch smoke at missing 0.0 and 0.4 must complete before formal runs.
7. Formal completion requires 80/80 jobs, zero worker errors, and 80/80 paired
   audits.

## Analysis

The primary table retains every seed and reports ten-seed Weighted-F1 mean and
sample standard deviation, paired delta, wins, Wilcoxon p-values, and Holm
correction across eight missing rates.

Collapse filtering is diagnostic only. The registered strict rule is:

```text
median(group) - score >= 0.10
and robust MAD z <= -3.5
```

A MAD-only sensitivity result is also reported. A pair is removed from a
filtered comparison if either condition is flagged in the corresponding
missing-rate group. Raw results remain the scientific primary result.
