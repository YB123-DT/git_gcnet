# Fixed-sum redistribution spike

Frozen MOSI causal Flat eta=.6 checkpoints, seeds66–70; AT/AV/TV only, nonzero
sentiment labels. Local and memory unchanged. Evaluate t=-1,-.5,0,.5,1 without
training, new epoch selection, normalization, projection or new architecture.

`U=(B+G)/2; D=(B-G)/2; B(t)=U+tD; G(t)=U-tD`.
Only the Base and unique active Gap classification slots are modified.
End points use direct copies to preserve exact Normal/Swap numerical behavior.

## Main result

Descriptive averages: rates .1–.7 equally within seed, then five seeds equally.
Rate0 has no eligible samples. Not a full-test or eight-rate benchmark score.

| t | W-F1 (%) | Score MAE versus Normal | Sign flips versus Normal |
|---|---:|---:|---:|
| -1 (Swap) | 79.206 | 0.04233 | 0.980% |
| -0.5 | 79.243 | 0.03122 | 0.803% |
| 0 (U,U) | 79.273 | 0.02047 | 0.622% |
| 0.5 | 79.128 | 0.01008 | 0.309% |
| 1 (Normal) | 79.182 | 0 | 0 |

No central performance collapse appears. At t=0, the mean difference from Normal
is +0.092 pp, positive in 4/5 seeds; seed67 is -0.409 pp. This is descriptive,
not a new selected model or evidence of significant improvement.

## Odd / even score sensitivities

`O(t)=(f(t)-f(-t))/2`, `E(t)=(f(t)+f(-t))/2-f(0)` are computed per sample using
the **raw regression score**, not W-F1, probabilities or calibrated logits.
Absolute values are taken before averaging.

| Pattern | Mean abs O(1) | Mean abs E(1) | Normal W-F1 | t=0 W-F1 | t=0 sign flips |
|---|---:|---:|---:|---:|---:|
| All | 0.02116 | 0.00484 | 79.182 | 79.273 | 0.622% |
| AT | 0.01765 | 0.00463 | 87.073 | 87.119 | 0.177% |
| AV | 0.01204 | 0.00360 | 62.959 | 63.293 | 1.202% |
| TV | 0.03382 | 0.00634 | 86.848 | 86.861 | 0.508% |

Overall at t=.5, mean abs O=.01072 and E=.00125. Both sensitivities are nonzero;
the odd component is larger than the even component. TV has the largest odd
amplitude (signed mean O(1)=-.03114): it retains systematic role sensitivity in
scores despite stable W-F1. AV changes decisions most often despite smaller
score deviations. Do not infer unchanged individual outputs from stable F1.

## Interpretation boundaries

On this path and subset, task performance is robust to redistributing the fixed
Base+Gap evidence and to eliminating the difference D. This is **consistent with
the sum-preserving component being sufficient to retain most classification
performance**, but not proof that the readout depends only on B+G. Local remains
present, and context deviations may simply not cross many decision boundaries.

There is no strong task-level evidence here for a difference-sensitive mechanism
that requires nonzero D; however, measurable odd and even score responses remain.
The five-point W-F1 curve is relatively flat, while per-sample scores are not
strictly exchange-invariant. No formal equivalence threshold was prespecified.

This result argues against explaining the earlier Local-gated drop solely by
requiring rigid original Base/Gap identities. It **does not establish the gate
bottleneck as the cause**: compression, shared projection, gating, averaging,
parameter counts and retraining dynamics were changed together in that experiment.
No follow-up training or geometry analysis is automatically started.

## Verification and artifacts

- `interpolate.py`: five fixed inputs, only eligible slots patched; no model edits.
- Existing `run.py` accepts an optional patch function/mode list; default old audit unchanged.
- Red-to-green tests: all three missing targets plus prior patch test, **4 passed**.
- Every batch checks fixed-sum conservation at FP32 tolerance (atol1e-6/rtol1e-5).
- All 40 Normal and Swap prediction arrays exactly equal their previous saved
  counterparts; all labels, masks and five checkpoint hashes identical.
- Normal logits equal original forward exactly; ineligible rows unchanged.
- `interpolation_results/`: raw five-output NPZs, per-sample odd/even CSV,
  per-seed/rate/pattern curves and components, grouped summaries, input hashes,
  checkpoint provenance and reproducible `RESULT.md`.
- `summarize_interpolation.py`: score-level analysis, no model execution.
- `git diff --check` passed. Existing artifacts are preserved.

Reproduce with existing remote s0 Python:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /data2/yb/reproduction_envs/s0/bin/python3.10 \
  experiments/osram_cross_substitution_20260910/interpolate.py --output NEW_DIRECTORY
python3 experiments/osram_cross_substitution_20260910/summarize_interpolation.py NEW_DIRECTORY
```

**INTERNAL DIAGNOSTIC ONLY.** Existing checkpoints were historically selected by
eight-rate-mean Test W-F1; this experiment does not select any new checkpoint.
