# Frozen large Flat + original Nested

INTERNAL DIAGNOSTIC ONLY

Status: RUNNING, verified epoch2/100 on 2026-10-09. First launch stopped before training by bitwise CUDA parity assertion; corrected verification retry passed all eight rates.

Source snapshot: `fced0fd`; dispatcher PID564489; training PID564639; tmux `frozen_nested_retry_20261009`; started 2026-10-09T08:24:01 UTC. Epoch1/2 frozen-state checks passed. No post-training test score yet.

Actual same-parent baseline (one `.7` source evaluated at all eight rates):

| Model | Mean8 W-F1 (%) | High .5/.6/.7 (%) |
|---|---:|---:|
| Fixed seed66 `.7` original large Flat | 79.540135 | 75.768636 |
| Frozen Flat + Nested, validation-selected | pending | pending |

Zero-init CUDA maximum absolute prediction difference across eight rates: 4.7683716e-7; all signs/W-F1/labels/masks exactly equal. Historical source `.7` score reproduced. This baseline is not the historical per-rate-selected 81.068 score.

- MOSI seed66, one fixed original large Flat `best_miss_0p7.pt` (source selected by Test-oracle).
- All eight rates use that same parent; the historical 81.068 per-rate checkpoint mixture is NOT this baseline.
- Only `osram.meaningful_block` trains: original `nested_gnn_rooted_evidence`, 159,235 parameters.
- Original modules/Dropout stay in eval mode; original parameters and buffers are hashed each epoch.
- Original Nested corrects Local/Base/Gap inputs to the retained original Flat; no graph enlargement or extra loss.
- Stage2: 100 epochs, Adam .001, weight decay .00001, batch32, seed66, original cyclic random missing schedule/task loss.
- Selection: one epoch by equal mean of eight **validation** W-F1 values. Epoch0 included, strict improvement (earliest tie). No test evaluation during training.
- CPU verification passed on biggpu: exact pretrained-parent output parity, three finite update steps, Nested changes, all original state unchanged, padding zero.
- CUDA diagnosis: first-batch Local Skip/adapter/norm inputs and outputs exactly equal; whole-rate predictions differ at most 2.3841858e-7 at rate0, zero polarity changes, identical W-F1. This is numerical-scale discrepancy, not evidence of a learned change. Full eight-rate launch check uses atol/rtol 1e-6, records maximum errors and requires EXACT labels/masks/polarity/W-F1. No model math changed for this fix.
- Baseline artifacts are saved from the actual Flat model, not substituted with zero-init Nested outputs. Source .7 W-F1 must reproduce original saved metrics.
- Baseline test is measured once; selected Nested is tested after training, with identical labels/order/masks and corrections/harms recorded.
- Checkpoints: `best.pt`, `last.pt` contain Nested + optimizer/RNG and immutable parent reference/hash. No automatic resume implementation; these are not self-contained full models.

Server: biggpu; physical GPU0 (never GPU4). Active remote root `/data2/yb/remote_experiments/osram_frozen_nested_20261009_retry`. Original root without `_retry` preserves the pre-training failed assertion and original source snapshot.

Run from sealed source:

```sh
python -m experiments.osram_frozen_nested_20261009.dispatch --root /data2/yb/remote_experiments/osram_frozen_nested_20261009_retry
```

Report `seed_66/metrics.json`: same-parent baseline/Nested per-rate W-F1 and ACC, mean8/high, validation-selected epoch, paired corrections/harms. No performance claim before completion.
