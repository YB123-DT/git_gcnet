# Causal ComP completion lanes

This queue uses idle physical GPUs 0 and 1 on `biggpu` to complete the causal
ComP matrix without duplicating the existing GPU-3 queue. The launchers share
the protocol-specific output roots and acquire each run's existing `flock`
lock. Completed and currently running settings are therefore skipped safely.

The remaining matrix is:

- IEMOCAP-4/6: seeds 66/67/68, rates 0.0--0.7, 200 epochs, stage 100;
- CMU-MOSEI: seeds 66/67/68, rates 0.0--0.7, 100 epochs, stage 50;
- CMU-MOSI: seed 68 only, rates 0.0--0.7, 300 epochs, stage 150;
- MELD: no launch because all three causal seeds are already complete.

Each lane uses six workers and four CPU compute threads per worker. GPU UUIDs
are passed explicitly; physical GPU 4 is never eligible.
