# Accepted MELD baseline: persistent A/T/V

INTERNAL DIAGNOSTIC ONLY. Frozen evaluation, no training or new selection.

Use exactly the accepted Flat1600 + Local Skip, Adam lr1e-5/wd1e-5,
seeds66/67/68,50epoch,GCNet-random-mask-v1 training checkpoints indexed by
`../osram_meld_wd_seed66_20261018/BASELINE.json`. Each seed's existing miss0.7
ACC-selected BEST is reused for A-only, T-only and V-only. No new weights.

Every valid turn of every test conversation receives the same named singleton
availability, including both host/guest masks; padding is zero. Original feature
preparation, speaker selection, causal read-before-write Memory, queries, joint
write, Local Skip, Flat and head are untouched. Memory starts at zero on each
ordinary model call. This is not a random-test subset grouped by current mask,
nor a newly trained unimodal model. T means Text.

First replay the checkpoint's exact saved random selection mask, verifying all
IDs/indices/labels/speakers, original logits/classes and ACC. Then evaluate three
persistent patterns on the identical utterance cohort. Save ACC, seven-class UA,
W-F1, CE, per-class recall/support and predictions, plus3seed means/sampleSD.
Selected original checkpoint/prediction hashes and model state remain unchanged.

Runtime remains ssh biggpu,cmerc-repro. Healthy physicalGPU3 UUID
GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a, logicalcuda:0. GPU4 forbidden.
One small frozen process; no other tasks paused or killed. Twelve finite passes:
three selected-mask sanity replays plus nine persistent evaluations.

Reuse the already sealed/manifest-verified diagnostic helper snapshot470e653,
which in turn loads original numerical b4a3138. Only this separate runner and
baseline registry are copied into the new immutable experiment code directory.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u run.py \
 --baseline BASELINE.json \
 --output /home/qinjinghui/remote_experiments/osram_meld_persistent_unimodal_20261018/runs_v1 \
 --commit <committed-runner-SHA>
```

Tests: `python -m pytest tests/test_meld_persistent_unimodal.py -q`.
The fixed mask is constant across time/conversations, nonempty, padding-safe,
does not consume RNG, and rejects invalid inputs. No extra training smoke.
