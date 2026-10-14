# MELD no-Local-skip retraining with the canonical GCNet mask

This experiment repeats the no-Local-skip retraining diagnostic while changing
only:

```text
osram_disable_local_skip = true
```

It uses the retained MELD Flat seed-66 configuration from the corrected
original-GCNet-mask rerun and hard-requires:

```text
missing_mask_protocol = gcnet-random-mask-v1
```

The run is 50 epochs and uses the established per-rate Test-oracle selection
protocol. It is an internal diagnostic, not a paper result.

Do not substitute `gcnet-legacy-fixed-v1`; that schedule exists only for
replaying historical frozen controls.
