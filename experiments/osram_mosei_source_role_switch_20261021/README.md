# MOSEI66 frozen Source-Role switch

INTERNAL DIAGNOSTIC ONLY. No training,weight updates,new mask sampling,
new checkpoint selection,new architecture or threshold search.

Reuse the completed original direct-source MOSEI66 run and each rate's
already-selected BEST checkpoint/exact stored test availability. Selection
epochs differ across rates;within each rate,learned versus all-active-gamma1
shares precisely the same checkpoint,Local,causal Memory trajectory and head.
Gamma1 retains sources,NOT gamma0 source deletion or original Flat retraining.

Extend the existing tested neutral-readout helper only with an opt-in switch
to skip12unrequested single-slot resets;its original default is unchanged.
Capture12gamma values,strict-past trajectory and source reads from one scan;
replay only the original Flat/head. Verify learned predictions reproduce
saved predictions exactly,frozen tensors/RNG stay unchanged,active masks,
source reconstruction errors,protected artifact and numerical source hashes.

Report ACC/W-F1/MSE,paired corrections/harms,continuous prediction changes,
active-only gamma mean/population variance/quantiles/extrema,by sources/roles.
Text groups partition utterances:currentT visible;currentT missing/pastT seen;
currentT missing/pastT unseen. Also report all seven current combinations.
Counts across eight rates are repeated exposures,not unique utterances.
Eight-rate/high metrics are equal rate means;empty groups are N/A.

Original s0 environment,batch32,official MOSEI features/split,GCNet masks.
One bounded inference job,biggpu healthyGPU7,neverGPU4,RAM>=32GiB/disk>=4GiB.
Immutable source and independent output;do not overwrite any old checkpoint,
mask,prediction or audit. Large NPZ caches stay remote;small stats go to Git.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosei_source_role_switch_20261021.run \
 --output /data2/yb/remote_experiments/osram_mosei_source_role_switch_20261021/execution_v1 \
 --source-commit SOURCE_SHA \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Interpretation is descriptive:near-unit gamma/near-zero output effect supports
weak direct calibration;nonunit gamma with little output change suggests low
decision sensitivity but does not prove Flat absorption. Mixed condition gains
motivate a hypothesis,not an automatically authorized new module or training.
