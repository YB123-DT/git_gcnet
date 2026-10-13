# MELD Flat Local/Base/Gap source combinations

**INTERNAL DIAGNOSTIC ONLY.** This is frozen inference on the retained MELD
Flat checkpoints selected by the established per-rate Test-oracle protocol.

For each seed and missing rate, the causal encoder and OSRAM memory are run
once. The exact tensors already entering the Flat readout are then replayed as
the seven non-empty source combinations `L`, `B`, `G`, `LB`, `LG`, `BG`, and
`LBG`. `L` includes both the Local Skip and the Local slice in the adapter;
`B` is the Base slice; `G` is the three already-masked Gap slots. Disabled
groups are zeroed only at the readout, so Memory trajectories never change.

The `LBG` replay must match the original logits and classes. Results report
ACC, macro recall (UA), and weighted F1 for seeds 66/67/68 at rates 0.0--0.7.

Remote source checkpoints:

```text
/data2/yb/paper/remote_runs_20261012/results/
  osram_meld_gcnet_legacy_fixed_50e_3seed_20261013/runs/flat_seed{66,67,68}
```

The replay uses the same ECERC MELD pickle recorded by the source-run
provenance; the feature-root arguments remain unused by the MELD loader.
