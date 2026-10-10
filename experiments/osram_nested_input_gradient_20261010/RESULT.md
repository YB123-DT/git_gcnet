# Original Nested versus Flat: pre-readout input gradient norms

INTERNAL DIAGNOSTIC ONLY

## Scope and measurement

MOSI, seeds 66/67/68, eight missing rates 0.0–0.7, original large cfg84
no-JEPA Flat and original `nested_gnn_rooted_evidence`. Existing per-rate BEST
checkpoints were reused without training, optimizer steps, or checkpoint selection.
The source checkpoints were selected by the existing Test-oracle protocol.
These results describe frozen trained checkpoints, **not training-time parameter
gradient histories**.

For each model, run its own original causal scan once, then detach Local/Base/Gap
at the common input interface and replay its exact readout with autograd enabled.
Nested measurements are before its evidence block; Flat measurements are before
its adapter. Local gradients include both the original Local Skip and adapter/block
path. Memory derivatives use only the effective forward 512 dimensions per slot;
the backward half remains constant zero. No gradient is propagated through the
causal scan in this diagnostic.

Measure `||d loss_t / d input_t||_2` with `loss_t=(prediction_t-label_t)^2`.
Sum valid per-utterance losses to avoid batch-size normalization. Also measure
`||d prediction_t / d input_t||_2`, which removes the MSE error multiplier.
Per-utterance gradients obey `d loss/d input = 2*(prediction-label)*d prediction/d input`.

Each seed/rate is first averaged over its eligible utterances, then the seed/rate
means receive equal weight. Local includes all 686 valid test utterances per run,
including neutral labels as in the task loss. Memory summaries exclude first turns.
Gap summaries additionally require that modality to be currently missing. Local,
Base and joint Memory have 24 seed/rate groups; Gap has 21 because rate0.0 has no
active Gap. Joint Memory is the norm of all active forward slots concatenated,
not the sum of slot norms. RMS and medians are available in `per_rate.csv`.

## Task-loss input gradient norms

| Input slot | Flat | Nested | Nested / Flat | Relative change |
|---|---:|---:|---:|---:|
| Local | 0.165133 | 0.164871 | 0.9984 | −0.16% |
| Base | 0.044255 | 0.035129 | 0.7938 | −20.62% |
| Gap-A | 0.034652 | 0.028581 | 0.8248 | −17.52% |
| Gap-T | 0.078151 | 0.058861 | 0.7532 | −24.68% |
| Gap-V | 0.052817 | 0.047865 | 0.9062 | −9.38% |
| Joint active Memory | 0.067420 | 0.054086 | 0.8022 | −19.78% |

## Output Jacobian and forward feature norms

| Input slot | Flat output Jacobian norm | Nested output Jacobian norm | Ratio | Flat feature norm | Nested feature norm |
|---|---:|---:|---:|---:|---:|
| Local | 0.087171 | 0.086422 | 0.9914 | 43.7304 | 45.7100 |
| Base | 0.023379 | 0.018368 | 0.7856 | 14.9632 | 16.5844 |
| Gap-A | 0.018637 | 0.015509 | 0.8321 | 8.7363 | 10.4357 |
| Gap-T | 0.030325 | 0.022499 | 0.7419 | 11.7178 | 12.6864 |
| Gap-V | 0.028909 | 0.025991 | 0.8991 | 11.1778 | 12.6519 |
| Joint active Memory | 0.034000 | 0.027250 | 0.8015 | 18.1687 | 20.2153 |

The aggregate Memory difference is also present in output Jacobians, rather than
only in MSE gradients. Input feature magnitudes differ between independently
trained models, so absolute gradient norms are coordinate/scale-dependent. This
comparison does not isolate a causal effect of inserting Nested into one fixed
backbone, establish historical usefulness, or demonstrate gradient vanishing.

## Seed variability

Ratios below are ratios of each seed's eight-rate mean norms (not means of ratios).

| Seed | Local Nested / Flat | Base Nested / Flat | Joint Memory Nested / Flat |
|---|---:|---:|---:|
| 66 | 1.0193 | 0.7028 | 0.7324 |
| 67 | 1.2908 | 1.2123 | 1.1798 |
| 68 | 0.7114 | 0.6169 | 0.6226 |

Seed67 reverses the aggregate direction. Do not describe reduced Memory gradient
as a uniform three-seed behavior.

## Per-rate task-loss gradient norms

Equal mean over the three seeds.

| Rate | Local Flat | Local Nested | Joint Memory Flat | Joint Memory Nested |
|---|---:|---:|---:|---:|
| 0.0 | 0.144742 | 0.144336 | 0.034988 | 0.027374 |
| 0.1 | 0.139517 | 0.144515 | 0.041571 | 0.034475 |
| 0.2 | 0.153694 | 0.148196 | 0.060170 | 0.041887 |
| 0.3 | 0.159847 | 0.156530 | 0.063245 | 0.048214 |
| 0.4 | 0.169846 | 0.171876 | 0.077051 | 0.058336 |
| 0.5 | 0.177684 | 0.180955 | 0.078411 | 0.064562 |
| 0.6 | 0.175435 | 0.183363 | 0.084681 | 0.072136 |
| 0.7 | 0.200299 | 0.189196 | 0.099242 | 0.085707 |

## Verification and provenance

- All 48 conditions completed; optimizer steps: 0.
- Same conversation/utterance IDs, labels and availability masks paired across
  Flat and Nested for all 24 seed/rate pairs.
- Original full-forward prediction versus gradient replay maximum absolute
  difference: 4.76837158203125e-7. All original saved W-F1 values reproduced.
- Reproduced three-seed eight-rate W-F1: Flat 80.559048%, Nested 80.488818%.
- Every checkpoint hash and frozen model state unchanged; no parameter gradients.
- Six analytic tests passed: known gradient/Jacobian scaling, duplication/batch
  invariance, Local Skip, identity Nested, inactive Gap/padding, frozen parameters.
  One existing environment `pynvml` deprecation warning; no test failure.
- Diagnostics implementation/source archive commit: `cde25c2`.
- Source checkpoint paths, hashes, selected epochs and W-F1 are in `SUMMARY.json`.
  Flat selected epochs span 39–88, Nested 47–98; these are not matched training
  epochs and should not be interpreted as a training dynamics comparison.
- Server: biggpu physical GPU7, UUID
  `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`.
- Execution UTC: 2026-10-10 02:03:10–02:05:30.
- Source and raw per-utterance CSV remain at
  `/data2/yb/remote_experiments/osram_nested_input_gradient_20261010/`.

Command (from the sealed remote source archive):

```bash
CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_nested_input_gradient_20261010.run \
  --output /data2/yb/remote_experiments/osram_nested_input_gradient_20261010/results \
  --gpu 7
```

No model, Memory formula, loss, mask schedule or training protocol was changed.
