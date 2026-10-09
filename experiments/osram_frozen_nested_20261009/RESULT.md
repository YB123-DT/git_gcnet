# Frozen large Flat + original Nested

INTERNAL DIAGNOSTIC ONLY

Status: bounded CPU verification passed; launch pending.

- MOSI seed66, one fixed original large Flat `best_miss_0p7.pt` (source selected by Test-oracle).
- All eight rates use that same parent; the historical 81.068 per-rate checkpoint mixture is NOT this baseline.
- Only `osram.meaningful_block` trains: original `nested_gnn_rooted_evidence`, 159,235 parameters.
- Original modules/Dropout stay in eval mode; original parameters and buffers are hashed each epoch.
- Original Nested corrects Local/Base/Gap inputs to the retained original Flat; no graph enlargement or extra loss.
- Stage2: 100 epochs, Adam .001, weight decay .00001, batch32, seed66, original cyclic random missing schedule/task loss.
- Selection: one epoch by equal mean of eight **validation** W-F1 values. Epoch0 included, strict improvement (earliest tie). No test evaluation during training.
- Tests: identical eight-rate Flat vs zero-init Nested predictions; CPU pretrained-parent gradient/update/freeze/padding check.
- CPU verification passed on biggpu: exact pretrained-parent output parity, three finite update steps, Nested changes, all original state unchanged, padding zero. Eight-rate CUDA parity is a mandatory launch-time assertion.
- Baseline test is measured once; selected Nested is tested after training, with identical labels/order/masks and corrections/harms recorded.
- Checkpoints: `best.pt`, `last.pt` contain Nested + optimizer/RNG and immutable parent reference/hash. No automatic resume implementation; these are not self-contained full models.

Server: biggpu; physical GPU0 (never GPU4). Remote root `/data2/yb/remote_experiments/osram_frozen_nested_20261009`.

Run from sealed source:

```sh
python -m experiments.osram_frozen_nested_20261009.dispatch --root /data2/yb/remote_experiments/osram_frozen_nested_20261009
```

Report `seed_66/metrics.json`: same-parent baseline/Nested per-rate W-F1 and ACC, mean8/high, validation-selected epoch, paired corrections/harms. No performance claim before completion.
