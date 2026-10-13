# ComP MELD GCNet-legacy fixed-mask rerun

This rerun evaluates causal ComP with exactly the same deterministic
conversation-level availability schedule used by the OSRAM Flat/Nested fair
comparison.

- seeds: 66, 67, 68
- epochs: 50; stage boundary: 25
- rates: 0.0 through 0.7
- mask protocol: `gcnet-legacy-fixed-v1`
- selection: per-rate Test-oracle (internal diagnostic)

The schedule matches GCNet's legacy observed-count contract but does not claim
to replay its process-global NumPy RNG stream.  Run:

```bash
bash experiments/comp_meld_legacy_mask_20261013/run_all.sh /path/to/new/output/root
```

Each seed runs its eight rates sequentially on a fixed healthy GPU.  The three
seeds run concurrently.  Completed checkpoints are discarded after durable
metrics and prediction artifacts are written.
