# MOSI direct-source three-seed follow-up

INTERNAL TEST-ORACLE SCREEN

User authorized three seeds after completing seed66. Reuse the completed
seed66 run from immutable source bd2e12e, not retrain or overwrite it.
Add ONLY seed67 and seed68, from scratch,150epochs each, one-stage original
MOSI scalar MSE,Flat1600,8heads,K/V64,write_step.6,batch32,Adam,wd1e-5.
LR.001 for100epochs then.0001 for50. Same original s0 runtime/data/features,
GCNet-random-mask-v1 cyclic training, exact common evaluation banks, per-rate
earliest Test-best W-F1 and corresponding ACC/predictions/masks/checkpoints.

The algorithm remains2*sigmoid(score) independently weighting12source paths
with shared Partner Read network and no original Base/Gap add-back. NO numeric
model/source-tracker/sampler/optimizer/learning-rate changes. No extra datasets,
methods,ablation,HPO or persistent training. Local/Local Skip/CHM untouched.
Existing completed Flat67/68 references are reused; each run checks its OWN
seed's initialization/config/split/seed/evaluation masks and all150training
mask/update-count/LR traces. Evaluation banks intentionally remain seed66,
matching the original three-seed Flat suite; training RNG uses actual run seed.

The shared worker/report accept explicit approved seeds; their defaults still
allow only66. The original direct single-seed entry is unchanged. Existing
seed66 and MOSEI source snapshots are not edited. Formal67/68 source hash will
differ in experiment orchestration only; all tracked numerical model/sampler
hashes must match seed66 before the three-seed result is published.

The persistent queue waits for successful completion of the existing seed66
QUEUE.json, performs one short real GPU smoke of the unchanged algorithm, then
admits at most two formal jobs concurrently on healthy GPU0. Memory gate:
1.5x measured peak per job plus2GiB/card and>=64GiB MemAvailable. No peer kills,
batch/precision changes or implicit retries. Existing full last checkpoints
and all8BEST with matched predictions are preserved separately per seed.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosi_source_role_direct_3seed_20261020.run \
 --output /data2/yb/remote_experiments/osram_mosi_source_role_direct_3seed_20261020/execution_v1 \
 --source-commit SOURCE_SHA --gpus GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

Complete-only report recomputes all24selected prediction metrics, verifies
same-seed Flat masks/cohorts and identical model/sampler source hashes across
the direct runs. Outputs per_seed reports,24-row METRICS.csv,SUMMARY.json and
RESULT.md. Reports individual seed/rate ACC/W-F1 and three-seed mean/sample
std(ddof1). High missing first averages.5/.6/.7 within each seed, then computes
across-seed statistics. Do not average the three rate standard deviations.
No scores are claimed before the corresponding runs/artifacts complete.

Pre-launch verification:4new tests failed before implementation, then the
68-test focused source-role/TASR/scalar/checkpoint suite passed. Compileall
and diff check passed. Independent read-only review found no blocking issue;
its separate21-test subset passed. Existing optional torch-sparse import
warning does not affect these no-graph-sparse training paths. No numerical
model/sampler changes relative to the completed seed66 source bd2e12e.
