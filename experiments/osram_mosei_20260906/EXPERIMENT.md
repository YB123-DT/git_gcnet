# CMU-MOSEI OSRAM diagnostic protocol

## Locked configuration

- Dataset: CMUMOSEI, official train/validation/test split, fold 1
- Features: frozen wav2vec-large-c-UTT, deberta-large-4-UTT, manet_UTT
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0--0.7 in steps of 0.1
- Training: one cyclic mixed-rate model per seed, 100 epochs, batch size 32
- Backbone: OSRAM, 8 heads, key/value dimension 32, output dimension 700
- Optimizer and loss: inherited Adam/JEPA configuration, lr 1e-3, weight
  decay 1e-5
- Device: Tesla V100-SXM2-32GB; GPU4 was not used
- Total parameters: 7,181,217; trainable parameters: 6,321,057

## Selection rule

Every epoch evaluated all eight test rates. The run checkpoint was selected by
the mean Test weighted-F1 across the eight rates
(8-rate-mean-test-oracle). For the requested per-rate view,
per_seed_rate.csv extracts the maximum Test weighted-F1 independently for
each rate from the immutable histories; the selected epoch can therefore
differ by rate and seed. This is an optimistic internal diagnostic, not a
formal benchmark protocol.

The saved metrics.json files identify the actual single checkpoint selected
by the eight-rate mean. The per-rate CSV is a separate diagnostic extraction
and must not be confused with those saved checkpoint metrics.

## Provenance and integrity

The raw history.json, metrics.json, config.json, and diagnostics.json
files were copied from the completed remote runs under raw/seed_*.
All five histories contain 100 epochs, all values are finite, and every
metrics file contains all eight test rates. No same-protocol GCNet control was
run in this batch, so this directory does not report an unpaired OSRAM delta.
