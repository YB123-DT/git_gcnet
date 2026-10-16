# Original Flat seed66 restoration

INTERNAL PER-RATE TEST-ORACLE ACC; original Flat, seed66, 100 epochs

Completed fold runs: 10/10. Status: complete.

| Dataset | Scope | ACC | UA | W-F1 |
|---|---|---:|---:|---:|
| IEMOCAPFour | fivefold mean | 79.549 | 80.024 | 79.439 |
| IEMOCAPFour | fivefold high | 76.899 | 77.519 | 76.743 |
| IEMOCAPSix | fivefold mean | 62.011 | 59.945 | 61.382 |
| IEMOCAPSix | fivefold high | 59.696 | 57.846 | 58.906 |

Fivefold averages require all five valid folds and weight folds equally. Seed66 only; no multi-seed uncertainty estimate.
Eight independently ACC-selected epochs are internal test-oracle diagnostics. Exact UA uses the same selected logits and masks.
All 80 BEST checkpoints and selected artifacts plus ten last_training states are retained; old discarded weights are not loaded.
