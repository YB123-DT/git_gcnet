# MOSI OSRAM learning-rate screen protocol

- Dataset: CMU-MOSI official split, fold 1
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0--0.7
- Training schedule: cyclic mixed-rate
- Epochs: 100; batch size: 32
- Selection: one checkpoint per seed using the mean Test weighted-F1 over all
  eight rates
- Diagnostic extraction: independent best Test epoch for every rate
- Backbone: OSRAM, heads=8, key/value dimension=32, output dimension=700
- Features: frozen wav2vec-large-c-UTT, deberta-large-4-UTT, manet_UTT
- Unchanged: Student Projector, EMA Teacher, MMoE, JEPA loss, masks,
  optimizer family, dropout, weight decay, temperature, and graph-independent
  OSRAM implementation

This is an internal Test-oracle diagnostic only. The per-rate table must not be
reported as a formal benchmark result. The existing 1e-3 OSRAM H8/700 run is
the reference; it is not retrained in this screen.

## Completion

Both variants completed five seeds and 100 epochs per seed. The immutable
histories are under raw/ for lr=5e-4 and raw_lr3e-4/ for lr=3e-4.
RESULT.md, summary.csv, and per_seed_rate.csv are generated directly from
those histories. The lower-rate branch is closed because both variants are
below the 1e-3 reference.
