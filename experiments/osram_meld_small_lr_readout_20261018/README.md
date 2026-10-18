# Frozen Local / Memory readout diagnostic for the small-LR MELD Flat

INTERNAL DIAGNOSTIC ONLY. No training, optimizer, new model, masks or threshold
selection. Applies to the fixed LR1e-5 / Adam weight_decay2e-4 original Flat.

Start with completed O6 seed66. Seeds67/68 continue their existing50epoch
training untouched and are evaluated automatically only after each source
SUMMARY.json says complete. Three frozen inference jobs total, no new training.

| Condition | Adapter input | Local Skip |
|---|---|---|
| L / Local-only | Local, Base=0, Gap=0 | unchanged |
| BG / Memory-only | Local=0, original Base and active Gap | entire output0, including bias |
| LBG / Full | original Local/Base/Gap | unchanged |

One ordinary causal CHM forward per batch. Capture the actual Flat input and
Local Skip input, then repeat only the original adapter/LayerNorm/task head.
Do not alter writes, Query, availability, original current observations or any
parameter. Thus “Memory-only” denotes the **direct classifier inputs**, not
history independent of current observations: original memory queries still
depend on current input. Inactive Gap and padding remain masked. This is not
a separately trained Local/Memory classifier or an information-capacity probe;
zeroing an input can be out of distribution.

For each of8 rates, load its exact ACC-best checkpoint and the checkpoint-linked
`selection_artifact` NPZ. Do NOT use final `predictions_miss_*.npz`, which can
come from a fresh random evaluation. Verify SHA256, IDs, utterance indices,
speakers and labels before inference; scatter only the saved selected-speaker
availability. Original Full classes must reproduce saved predictions exactly,
logits within stated floating-point tolerance. Full replay must match the
original forward and the accepted Test-best ACC/UA/W-F1. No mismatch bypass flag.

Per-rate results include ACC, seven-class UA (Macro Recall), W-F1, CE, per-class
recall/support, Local→Full corrections/harms and Local-vs-Memory correctness
cross counts. Store all three predictions/logits with sample metadata remotely.
Summary gives eight-rate and high-missing(.5/.6/.7) means for each condition.
No test-label-derived gate or switching rule is learned.

Runtime and source are the same as the O6 training: cmerc-repro,
Torch1.13.1+cu117, original b4a3138 numerical checkout and accepted ECERC feature
SHA256 `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.
The shared helper manifest is unchanged; this extension has its own manifest.

On biggpu healthy GPU1, one inference worker may run alongside the existing
two seed training workers; existing measured memory provides ample headroom.
GPU4 forbidden. No training job is paused, killed or moved.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_small_lr_readout_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_small_lr_readout_20261018/runs_v1
```

Use the immutable deployed helper snapshot with its root SOURCE_MANIFEST.json.
The persistent queue evaluates seed66 immediately and watches seeds67/68; it
does not hold a GPU process while waiting. Keep previous source outputs intact.
