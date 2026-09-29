# Post-GRN regression diagnosis, 2026-09-29

Code/results archived first in `9b4006a`. No model code was changed and no
training was restarted during this diagnosis. All performance statements
below are internal Test-oracle diagnostics, not formal validation-selected
results. No new hyperparameter or checkpoint selection was performed.

## Frozen residual intervention

Use the existing 24 selected checkpoints (seeds 66/67/68, rates .0–.7) from
`/data1/yb/remote_experiments/osram_cfg84_post_grn_20260929/runs` on biggpu.
Use its sibling isolated `code/`, the original s0 Python environment,
CUDA_VISIBLE_DEVICES=0, GPU UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,
and GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset.
GPU 0 had 31755 MiB free and zero reported utilization before this check.

Load each checkpoint strictly, call the existing `_build_model`, `_schedules`
and `evaluate_rate` under `torch.no_grad()` / eval mode. First reproduce
the GRN-on score; all 24 agree with recorded W-F1 within 1e-10 and match
the recorded mask SHA256. Then install this temporary forward hook:

```python
handle = model.osram.post_grn.register_forward_hook(
    lambda module, args, output: args[0]
)
# evaluate_rate with exactly the same checkpoint, loader and schedule
handle.remove()
```

This bypasses only the additive residual at inference, retaining the
trained backbone, Flat parameters, task head and Flat input sanitization.
Padding is excluded by the existing metric mask; this hook is not a
deployable replacement module. On/off mask hashes agree in every pair.
The initial attempt lacked GCNET_DATASET_ROOT and stopped before inference;
the successful run supplied the original dataset root explicitly.

| Seed | Rate | Selected epoch | GRN on W-F1 (%) | Residual bypass W-F1 (%) |
|---|---|---:|---:|---:|
|66|.0|72|87.286058|86.448559|
|66|.1|62|85.081409|84.949872|
|66|.2|77|82.360455|81.974347|
|66|.3|63|79.570201|79.415682|
|66|.4|92|78.930568|76.545775|
|66|.5|79|75.706055|75.734644|
|66|.6|79|74.340583|73.803050|
|66|.7|80|74.510586|74.102706|
|67|.0|100|87.032949|87.026134|
|67|.1|49|85.852135|84.880527|
|67|.2|49|82.896352|82.852576|
|67|.3|100|80.151676|79.088731|
|67|.4|69|75.454636|73.925690|
|67|.5|62|74.010737|73.119671|
|67|.6|48|75.890204|75.246296|
|67|.7|55|73.892660|74.740046|
|68|.0|51|87.004112|86.541394|
|68|.1|61|83.575948|84.030669|
|68|.2|61|81.974347|81.057789|
|68|.3|57|80.462238|81.658279|
|68|.4|51|78.876382|77.674917|
|68|.5|57|77.758835|76.500640|
|68|.6|74|76.487138|76.177758|
|68|.7|60|69.386373|67.986977|

Overall mean: on **79.5205265811**, bypass **78.9784470665**;
on minus bypass **+0.5420795146 pp**. On is better in 20/24 pairs.
Original independently trained Flat remains **80.5590478741**.

**Interpretation:** removing the residual does not recover the original
Flat score. The residual helps this co-adapted model on average, while
the jointly trained system remains worse than the original Flat run.
This rules against a simple inference-time "harmful additive noise"
explanation, but does NOT prove backbone degradation as the unique cause:
removing a jointly trained branch creates an intervention distribution
shift, and the checkpoints were selected with that branch enabled.

## Scope and training evidence

Three-seed mean deltas against original Flat by missing rate:
.0 −1.311, .1 −1.009, .2 −1.021, .3 −0.938,
.4 −1.242, .5 −1.501, .6 −0.276, .7 −1.010 pp.
Thus degradation is not confined to high missingness; even fully observed
input decreases. This is not a whole-conversation missing-modality test.

Config differences besides post-GRN are serialized defaults absent from
old JSON: beta_mode=embedded, gap_residual_strength=1.0,
joint_pretrain_freeze=True, osram_history_query_adapter=False,
completion_write_to_memory=False. They match current defaults; no
persistent mix, completion, history query adaptation or JEPA was enabled.

Selected epochs span 48–100. Fixed-epoch curves fluctuate in both models;
they do not establish a uniform early/late overtraining explanation.
For example, seed 66 at epoch 50 has train loss 1.010 / 1.629 and mean
test W-F1 79.488 / 72.546 for Flat / GRN, whereas seed 67 at epoch 100
has train loss 1.155 / .915 and mean test W-F1 69.310 / 77.034.
Do not equate these fixed-epoch metrics with the per-rate selected scores.

GRN adds training dropout draws and a new gradient path through the
condition and residual. Identical initialization and configured seed do
not guarantee identical later dropout draws or optimizer trajectories.
This is a possible contributor, not a measured causal attribution.

## Implementation verification discrepancy

Local Python/Torch 2.2.2+cu121 on TITAN Xp: 16/17 post-GRN tests passed;
full-model `torch.equal` failed. Diagnostic inspection found identical
shared initial weights and exactly zero GRN residual, with maximum
output differences 3.5762786865e-7 (eval) and 9.6857547760e-8 (train).
`allclose` passed. Repeating Flat alone with restored RNG gave zero
difference. Therefore do not label this simply nondeterministic replay:
the exact numerical source in the differing readout paths is unresolved.

On biggpu in the actual isolated training snapshot/environment, all
17 post-GRN tests passed, including full-model strict initialization
identity and finite joint updates. No assertion was weakened. There is
no evidence here that the tiny local-only difference caused the remote
performance regression, but cross-device bitwise identity is not proven.

## Next action (not run)

If pursuing this architecture further, first isolate the training-path
effect with a controlled diagnostic, rather than adding another block or
assuming the residual should simply be removed. The present evidence
does not uniquely separate optimization, co-adaptation and generalization.
