# CMU-MOSI Text-Anchor Residual Fusion

## Intervention

Replace only Slot node fusion when Text is observed:

`T-only Slot anchor + norm-bounded residual from the complete observed Slot`.

The residual output is zero initialized and capped at 25% of the Text-anchor
norm. When Text is absent, the existing Slot fusion is used unchanged. GCNet,
Missing-M3, JEPA, masks, optimizer, frozen features, and readout are unchanged.

## Formal protocol

- Seeds 66--70, 100 epochs, cyclic mixed-rate training.
- One checkpoint per seed selected by eight-rate validation W-F1 mean.
- The selected checkpoint is evaluated at all eight test rates.
- Original Slot results are inherited, not rerun.

## Test result

| Miss | Anchor W-F1 | Slot W-F1 | Delta |
|---:|---:|---:|---:|
| 0.0 | 83.92 | 85.50 | -1.59 |
| 0.1 | 82.14 | 83.57 | -1.44 |
| 0.2 | 79.30 | 81.10 | -1.80 |
| 0.3 | 78.46 | 79.54 | -1.08 |
| 0.4 | 76.52 | 76.84 | -0.32 |
| 0.5 | 74.50 | 75.07 | -0.56 |
| 0.6 | 73.29 | 74.74 | -1.45 |
| 0.7 | 71.53 | 71.96 | -0.43 |
| Eight-rate mean | **77.46** | **78.54** | **-1.08** |

Seed-level eight-rate deltas are `-2.07`, `-0.74`, `+0.25`, `-2.62`, and
`-0.23` points. Only 1/5 seeds improves. The formal screen fails.

## Interpretation

The localization result was real, but a hard Text anchor is too restrictive.
It prevents the large Slot shift, yet also limits the model's ability to learn
joint representations; this is most visible at miss=0 (`-1.59`). The smaller
loss at miss=0.7 is consistent with Text being absent more often, which invokes
the unchanged Slot fallback.

This variant must not be promoted as the main fusion mechanism or tuned by
changing only the residual cap after seeing test. The separate test-oracle
diagnostic shows that useful peaks exist but validation does not select them
reliably.

