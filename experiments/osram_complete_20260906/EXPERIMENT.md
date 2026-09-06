# OSRAM CMU-MOSI diagnostic protocol

## Provenance

- Branch: `feature/osram-complete`
- Code commit: `41ccd4e`
- Dataset: CMU-MOSI official train/validation/test split
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- Seeds: 66, 67, 68, 69, 70
- Fold: 1
- Missing rates: 0.0 through 0.7 in steps of 0.1
- Training schedule: cyclic (one rate per batch, rate rotates over the eight rates)
- Epochs: 100
- Batch size: 32
- Optimizer: existing Adam configuration, learning rate `1e-3`, weight decay `1e-5`
- Device: Tesla V100-SXM2-32GB; GPU4 was not used

## Checkpoint selection

The run itself used the requested eight-rate Test-oracle checkpoint protocol:
each seed trained one model, evaluated all eight test rates after every epoch,
and saved one checkpoint at the epoch with the largest eight-rate mean weighted
F1.  The run metadata records:

```text
selection_split = test
selection_protocol = 8-rate-mean-test-oracle
```

Because the latest diagnostic request asked to inspect the best score at each
rate, `per_seed_rate.csv` additionally extracts the maximum weighted F1 for
each rate independently from the already recorded epoch histories.  Its
`selected_epoch` is therefore rate-specific and is an optimistic per-rate
Test-oracle diagnostic; it is not the single-checkpoint result and must not be
reported as a formal benchmark number.

## Control

The historical controls were not directly reusable: one used
`text-anchor-residual` fusion and another used `all` rate training with `slot`
fusion.  A fresh GCNet Control was therefore run with exactly the OSRAM command
line except for `--backbone-type gcnet` (and the corresponding legacy backbone
path).  Its raw histories are stored under `raw_control/`.

## Required checks

The following were completed before the formal run:

1. OSRAM shape, hard gap mask, padding, reset, read-before-write, and block
   permutation tests;
2. absent-slot finite-gradient test;
3. structured predictor target-slot separation test;
4. full existing Missing-M3 regression suite;
5. one-epoch MOSI OSRAM smoke with finite forward, loss, and backward values.

The remote result directories contain the per-epoch histories and run configs;
the compact artifacts in this directory are generated from those immutable
records.
