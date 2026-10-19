# TASR F2 learning-rate screen

INTERNAL DIAGNOSTIC ONLY. User-authorized finite follow-up,2026-10-19.

Only Adapter Adam LR changes:3e-6,1e-5,3e-5. Six new jobs: MELD seed66 and
IEMOCAP6 seed66/fold5, each three rates. Existing3e-4 F1/F2 results remain in
`experiments/osram_tasr_20261018/RESULT.md`, not rerun or overwritten.

All backbone parameters remain frozen. Same protected original miss=.3 anchors,
initialization, GCNet random-mask-v1, independent Persistent view, .8/.2 CE,
30epochs, batch32, wd1e-5, original clipping, exact evaluation masks. One Adapter
checkpoint selected on validation random8 ACC then CE. No test-based selection.
IEMOCAP validation is an Adapter-only training-conversation holdout already seen
by the historical backbone; old backbone checkpoints were Test-oracle.

No OSRAM/model/sampler change. The old runner gains one optional positive finite
`--adapter-learning-rate`; omitted means exactly the original3e-4 configuration.
Its effective value is in optimizer and checkpoint identity, so changed-LR
resumption is rejected. No additional GPU structural smoke: existing verified
core hashes must match; each job retains real zero-init parity/frozen/gradient
and checkpoint gates. CPU regression covers override/default and finite plan.

Server biggpu, cmerc-repro environment, physicalGPU1 UUID-bound, at most4 jobs;
remaining two automatically fill vacated slots. Reserve3GiB/job from original
measured peaks below1.02GiB; require5GiB free. Never use faulty GPU4 or modify
external jobs. Separate immutable source and output directories under
`/home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/`.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_tasr_lr_20261019.launch \
  --output /home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/runs_v1 \
  --source-commit <immutable-source-commit>
```

Queue is recoverable and protected by the existing scheduler lock. Completed
jobs are not relaunched. Same-source last.pt resumes optimizer/RNG/config.
At queue completion automatically write LR_RESULTS.json/CSV and RESULT.md.
All per-rate, Persistent7, high-missing, pairedF0 corrections/harms and source
diagnostics stay in each SUMMARY.json/evaluation. Ranking on test is not used
to select trained weights, and single-seed LR comparisons are not paper claims.
No automatic search expansion, new loss, module, seed or fold.
