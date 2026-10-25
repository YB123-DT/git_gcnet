# Original Nested on frozen current MOSI Flat

This is a new, finite seed66 experiment, not a continuation of the historical
20261009 validation-selected100epoch frozen Nested run.

## Fixed comparison

Eight protected own-rate Flat parents from
`/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`.
Reuse the completed NoArea stage2 preparation and exact test masks. Original
Flat, NoArea parallel, NoArea serial and this Nested run therefore share parent
selection, data, splits, mask sampler and fixed testing masks.

Only original `MeaningfulInputAdapter(..., nested_gnn_rooted_evidence,8,64)`
trains,159235parameters. It corrects Local/Base/Gap only at the Adapter input.
Local Skip receives the original uncorrected Local. CHM, queries, joint Block
Write, read-before-write, task head, and all parent parameters stay frozen/eval.
No Signature descriptor, attention extension, extra loss or topology change.

Stage2:150epochs, Adam, batch32, WD1e-5;3e-4 first100epochs,3e-4/10 last50.
CyclicGCNet-random-mask-v1 training, only the requested own-rate evaluation.
Original MSE includes neutral labels. Strict earliest own-rate Test-best W-F1;
ACC comes from exactly the same prediction/mask. Full block/Adam/RNG/history
recovery state is saved, with the separately protected parent reference/hash.

**INTERNAL TEST-ORACLE SCREEN.** The eight-rate aggregate combines eight frozen
parent/block pairs, not a single model checkpoint or independent test evidence.
Added parameters differ from NoArea parallel281856/serial238798.

## Finite persistent queue

Use the cmerc-repro Python, with an immutable overlay ahead of pinned numerical
source `e0ca94c4c33de0efda2c2f4786fd289067e1ab2c` in PYTHONPATH. The unchanged
`osram_recent_noarea_serial_20261025` utility modules must also be in the overlay.
Do not use the live editable checkout as the numerical training source.

```sh
python -m experiments.osram_frozen_nested_gcnet_20261025.scheduler \
  --root /data2/yb/remote_experiments/osram_frozen_nested_gcnet_20261025/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
  --per-gpu 4
```

One real rate0.7 two-epoch/one-batch smoke gates eight formal jobs. Measured VRAM
and RAM determine admission. GPU4 and other users' processes are untouched.
Compatible unfinished jobs can be resumed using the same source with`--resume`;
no weights-only restart is represented as full recovery.

Each run saves baseline/selected prediction arrays, checkpoint hashes, config,
source/data/environment identity, parent invariance, scan count and gradient
coverage. Final summarization independently checks selected metrics, labels,
exact mask and earliest selected epoch. Failures remain visible and cannot rank.

Tests: `tests/test_frozen_nested_gcnet_model.py` and
`tests/test_frozen_nested_gcnet_runner.py`; old stage2 optimizer/trace/RNG
utilities are reused unchanged. Tests and smoke do not demonstrate a gain.
