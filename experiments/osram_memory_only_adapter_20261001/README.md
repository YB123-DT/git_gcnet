# Memory-only Adapter, original Local Skip

Authorized comparison: original cfg84 Flat versus
`LN(local_skip(Local) + emotion_adapter([Base, masked Gap-A/T/V]))`.
Keep the original complete Skip including bias. Remove Local channels from
the Adapter input, not from the model, encoder, or memory/query computation.
The Adapter input normalization and first linear layer therefore shrink by
latent_dim; this is a structural change, not a parameter-count matched control.

Only configuration delta: `osram_memory_only_adapter=true` (default false).
All gates disabled. No JEPA/completion, auxiliary losses or persistent training.
Reuse original reference configs for seeds 66/67/68: 100 epochs, batch 32,
Adam lr .001, weight decay 1e-5, random cyclic missing rates 0–.7,
causal cfg84 output_dim 1600. Train from scratch jointly; freeze nothing.
W-F1 is the primary comparison; retain the original MOSI regression loss.

Server biggpu, host GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153;
GPU4 prohibited. Up to three staggered concurrent jobs; isolated code snapshot,
separate outputs, no original baseline rerun or checkpoint overwrite.

Historical per-rate Test-oracle best checkpoints (8 per seed) are retained for
matched INTERNAL DIAGNOSTICS ONLY, not formal validation-selected paper scores.
The launcher verifies original reference config/provenance and evaluation masks.
Run with the established remote Python environment:
`python experiments/osram_memory_only_adapter_20261001/run.py --launch`.

Hypothesis: separating Local Skip from a memory-only Adapter improves W-F1.
A non-improvement is a valid negative result, not evidence of a software fault.
