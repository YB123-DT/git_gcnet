# Causal ComP three-seed completion

This directory completes the auditable three-seed causal ComP baseline without
rerunning results that already have matching provenance.

The launch order is deliberately:

1. IEMOCAPFour, seeds 66/67/68, missing rates 0.0--0.7;
2. IEMOCAPSix, seeds 66/67/68, missing rates 0.0--0.7;
3. CMUMOSI seed 68 (seeds 66/67 are already complete);
4. CMUMOSEI, seeds 66/67/68.

MELD causal seeds 66/67/68 are already complete and are not repeated. Every
new run uses the original ComP missing-mask implementation and explicitly sets
`--temporal-mode=causal`. The upstream training budget is retained: 300 epochs,
stage transition at epoch 150, batch size 32, hidden size 256, depth 4, two
heads, and learning rate 1e-4. Checkpoint selection remains the repository's
existing test-oracle protocol and must be labelled internal diagnostic.

On `biggpu`, start the GPU-3 queue with:

```bash
bash experiments/comp_causal_all_3seed_20261013/dispatch_gpu3.sh
```

The dispatcher reads `MAX_PARALLEL` from the output root on every scheduling
cycle. It is initialized to four so concurrency can be raised without killing
or duplicating existing jobs after checking actual GPU throughput.

