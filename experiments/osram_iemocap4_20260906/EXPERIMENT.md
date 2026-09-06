# IEMOCAP-4 OSRAM diagnostic protocol

## Locked configuration

- Dataset: `IEMOCAPFour`, fold 5, official Session-5 protocol
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0--0.7 in steps of 0.1
- Training: one cyclic mixed-rate model per seed, 100 epochs, batch size 32
- Backbone: OSRAM, heads=8, key/value dim=32, output dim=700
- Optimizer and loss: inherited Adam/JEPA configuration, lr=1e-3, weight decay=1e-5

## Selection rule

Every epoch evaluated all eight test rates. The run checkpoint itself was
selected by the eight-rate mean Test weighted-F1 (`8-rate-mean-test-oracle`).
For the requested per-rate view, `per_seed_rate.csv` extracts the maximum
Test weighted-F1 independently for each rate from the immutable histories;
the selected epoch can consequently differ by rate and seed. This is an
optimistic internal diagnostic, not a formal benchmark protocol.

## Provenance

The raw `history.json`, `metrics.json`, `config.json`, and `diagnostics.json`
files are copied from the completed remote runs. `positive_seed_count` is left
as `NA` in `summary.csv` because no same-protocol GCNet control was run in
this IEMOCAP batch; no unpaired delta is reported.
