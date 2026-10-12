# ComP on MELD (ECERC features)

This experiment compares the same vendored ComP implementation in causal and
original noncausal temporal modes. Both modes use the ECERC MELD pickle,
official train/test dialogue split, seed 66, and the original two-stage ComP
schedule (300 epochs; transition at epoch 150). Each missing rate is selected
by the repository's existing per-rate test-oracle rule, so these are internal
diagnostic results rather than leakage-free paper results.

MELD has nine speaker identities while ComP's input adapter only selects one
of two legacy host/guest slots. The adapter therefore duplicates each
utterance feature into both slots. This preserves the observed features but
does not add explicit nine-speaker modeling to ComP.

Run one condition with:

```bash
experiments/comp_meld_20261012/run_one.sh causal 0.0 3
experiments/comp_meld_20261012/run_one.sh noncausal 0.0 3
```

For a remote environment, set `COMP_PYTHON`, `MELD_ECERC_PICKLE`, and
`COMP_MELD_RUN_ROOT`. A queue can be split across devices with the matched
`COMP_SHARD_INDEX` and `COMP_SHARD_COUNT` environment variables.
`COMP_CPU_THREADS` defaults to four to prevent PyTorch/BLAS oversubscription
when several independent rates share a server.

For disposable tuning screens, set `COMP_KEEP_CHECKPOINTS=0`. The runner then
deletes only checkpoint-like files (`*.pt`, `*.pth`, and `*.pkl`) under that
run's `saved/` directory after successful training, while retaining metrics,
logs, and the manifest. The default remains `1` for backward compatibility.
