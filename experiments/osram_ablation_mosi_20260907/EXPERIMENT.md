# OSRAM CMU-MOSI mechanism ablation

## Locked protocol

- Dataset: CMU-MOSI, fold 1
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0 through 0.7
- Training schedule: cyclic mixed-rate
- Epochs: 100; batch size: 32
- Diagnostic extraction: each rate selects its own best Test weighted-F1
  epoch from the same cyclic mixed-rate history (`per-rate-test-oracle`),
  matching the current OSRAM comparison convention
- Features: frozen `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- Backbone capacity: OSRAM heads=8, key/value=32, output=700
- Unchanged: Student Projectors, EMA Teacher, MMoE, JEPA loss, masks,
  optimizer, learning rate, dropout, weight decay, and cyclic schedule

Full OSRAM is inherited from `experiments/osram_heads8_out700_20260906/`.
Only `local-only` and `local-base` are newly trained.

This directory is **INTERNAL DIAGNOSTIC ONLY** and uses test information for
per-rate checkpoint selection.  It is not a formal benchmark result.

## Outputs

The raw remote runs are copied under `raw/` after completion.  `summary.csv`
and `per_seed_rate.csv` are generated from the immutable per-seed histories.
