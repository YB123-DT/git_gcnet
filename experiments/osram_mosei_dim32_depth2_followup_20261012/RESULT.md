# MOSEI Nested follow-up

**INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle checkpoint selection.**

## Dim32 + depth2, three seeds

All compared checkpoints selected before epoch 60. Seed66 is reused from the
original 100-epoch screen; seed67/68 were trained for 60 epochs.

| Seed | Flat 8-rate | Nested 8-rate | Delta | Flat high | Nested high | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 66 | 83.671 | 84.095 | +0.424 | 80.890 | 81.279 | +0.389 |
| 67 | 83.886 | 83.857 | -0.029 | 81.650 | 81.771 | +0.121 |
| 68 | 83.836 | 83.430 | -0.406 | 81.846 | 81.146 | -0.700 |
| Mean | 83.798 | 83.794 | **-0.004** | 81.462 | 81.399 | **-0.063** |
| Sample SD | 0.113 | 0.337 | — | 0.505 | 0.329 | — |

The seed66 gain does not reproduce on seeds67/68. The three-seed aggregate is
effectively tied overall and slightly lower at high missingness.

## Seed66 low-learning-rate pairing (60 epochs, LR 1e-4)

| Model | 8-rate mean | High missing |
|---|---:|---:|
| Flat | 83.939 | 81.244 |
| Nested dim32+depth2 | 83.864 | 81.131 |
| Nested - Flat | **-0.075** | **-0.113** |

The lower learning rate does not recover a gain for this Nested configuration.

## Pending seed68 variants

The MLP128 and depth2 seed68 runs remain in progress and are not included in
the completed comparisons above.
