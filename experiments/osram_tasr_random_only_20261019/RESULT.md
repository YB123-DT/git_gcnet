# MELD F2 random-only completed

INTERNAL DIAGNOSTIC ONLY. Seed66,30epochs, Adapter LR1e-5/wd1e-5.
The original Flat backbone remains frozen. Source10699fd; biggpu GPU0,
cmerc-repro, runtime473.419seconds. Adapter validation selected epoch30.
This is a frozen incremental-value experiment, not one-stage joint training.

| Configuration | Random8 ACC | UA | W-F1 | Persistent6 ACC | UA | W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Same frozen F0 |61.336207|38.860509|58.642901|57.605364|32.040601|50.850579|
| F2 paired .8/.2, LR1e-5 |61.355364|38.869950|58.659877|57.605364|32.040601|50.852059|
| F2 random-only, LR1e-5 |61.360153|38.876305|58.664634|57.598978|32.038706|50.846801|

Random-only vs F0:+0.023946pp ACC,+0.015796pp UA,+0.021733pp W-F1.
Persistent6:-0.006386pp ACC,-0.001896pp UA,-0.003779pp W-F1.
Removing the Persistent training view does not produce a meaningful change in
this fixed-backbone single-seed experiment. It does not establish how a fresh
jointly trained model would perform. Do not interpret the tiny deltas as a
robust improvement or proof that the Adapter has no possible end-to-end value.

Both jobs are statuscomplete, epoch30, frozen backbone unchanged. Protected
anchor configuration/SHA, split, validation masks, every evaluation condition's
F0 and mask hashes were checked equal. Initial logits exactly matched F0.
First epoch:33 batches/33 task forwards/33 optimizer updates, loss=randomCE;
no Persistent samples, finite nonzero gradients in all12 Adapter parameter
tensors. Frozen case and evaluation metadata are retained in the two JSON files.

SUMMARY_RANDOM_ONLY.json is copied unchanged from:
`/home/qinjinghui/remote_experiments/osram_tasr_random_only_20261019/runs_v1/MELD_real_lr1em05/SUMMARY.json`.
REFERENCE_PAIRED_LR1EM05.json is copied unchanged from:
`/home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/runs_v1/MELD_lr_1em05/SUMMARY.json`.
These contain all random8/high/Persistent7 scores, recalls, support, corrections,
harms, diagnostics, checkpoint/mask/data hashes and training identity. Weights,
full predictions and actual masks stay remote; no new inference was run here.

One Adapter checkpoint per method selected by validation random8 ACC then CE;
historical common backbone was originally Test-oracle, so this is not an
independent paper test score. F0 here is one fixed miss=.3 anchor for all rates,
not the old three-seed/per-rate Test-best baseline mean.

User subsequently proposed one-stage training. That would mean training the
original backbone plus zero-output-initialized TASR jointly from scratch with
only original random task loss, without loading a trained Flat first. No such
joint run has been launched by this result report; no old run is relabeled.
