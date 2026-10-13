# MELD GCNet-legacy fixed-mask rerun

This rerun compares the cfg84 Flat readout and the selected Nested GNN under
the same deterministic, count-matched missing-modality schedule.

- dataset: official ECERC MELD train/dev/test pickle
- seeds: 66, 67, 68
- epochs: 50
- rates: 0.0 through 0.7
- checkpoint selection: per-rate Test-oracle (internal diagnostic)
- mask protocol: `gcnet-legacy-fixed-v1`
- Nested configuration: `meld_nested_d32_l4_g2_m64`

The schedule reproduces the observed-count contract of GCNet's legacy
`random_mask` while deriving masks from stable conversation identifiers.  It
is not a replay of the original process-global NumPy RNG stream.  In
particular, rate 0.7 retains exactly one modality at every valid utterance and
therefore realizes a 2/3 missing rate.

Run on the experiment server:

```bash
python -m experiments.osram_meld_legacy_mask_20261013.dispatch \
  --root /path/to/new/output/root
```

The dispatcher refuses to overwrite an existing root and deletes model
weights only after each run has written its metrics and provenance.
