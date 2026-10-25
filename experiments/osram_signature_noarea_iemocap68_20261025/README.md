# Seed68 matched training: Signature without directional area

User-approved finite control: IEMOCAP-6 seed68,all five LOSO folds and eight
rates,40independent per-rate branches. Original complete Signature is reused as
the matched comparator; it is NOT retrained or overwritten. No other seeds,
datasets,losses or hyperparameter searches are included.

This is new training from the SAME random initialization,not the earlier frozen
intervention on already-trained normal Signature weights. The original Flat
encoder/CHM/queries/LocalSkip/Norm/head remain frozen. Only the same281856
Signature parameters train. Parent and masks are the exact same own-rate original
Flat checkpoint and saved test masks used by the completed normal Signature.

## Only changed factor

For each modality's degree-two path descriptor,keep first level S1 and replace
second level S2 by `0.5 * S1 outer S1`,its symmetric component. This removes the
antisymmetric oriented-area component at TRAIN and TEST. Keep the endpoint,
time coordinate,counts,availability,currentLocal,network widths,zero outlet,
state-dict keys and initialization unchanged. The original32-bit numerical path
is retained elsewhere. Effective input degrees of freedom are reduced by design;
equal nominal parameter counts do not claim identical function classes.

The original Signature implementation and all previous results remain unchanged.
`NoAreaSignatureBlock` overrides only descriptor construction; it inherits the
original short-window8,padding and no-history correction rules.

## Matched training protocol

- cmerc-repro Python on biggpu; unchanged numerical source21e46f2.
- 100epochs,constant branchLR3e-4,Adam,WD1e-5,batch32,original CE and clipping.
- Original cyclic Random Missing0.0–0.7 over training batches.
- Test only each branch's designated rate with exact original GCNet masks.
- Earliest strict Test-best ACC; UA/WF1/class recall from that SAME epoch/mask.
- Original normal Signature branch config,parent,initialization and each epoch's
  mask trace,sample count,rate schedule and optimizer update count must match.
- No MSE metric or objective,validation reselection,paired views or auxiliary loss.
- Full last checkpoint includes block,Adam,epoch,history,best and RNG.

Internal per-rate Test-oracle exploration,not a single deployable checkpoint or
independent generalization result. Main reporting uses equal fold/rate means;
pooled prediction metrics must stay separately labelled.

## Execution and provenance

Run a committed independent namespace-package overlay from the original
`/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2`
working directory. Never edit that immutable numerical snapshot. The new overlay
hashes and original numerical hashes are recorded independently. The original
reference lives under `execution_v1/train/IEMOCAPSix_68_foldF_R`.

The finite scheduler gates admission on a real two-update smoke (fold5,rate0.7).
Smoke compares actual masks to a same-RNG replay of the original two-update
Signature; truncated smoke masks are not incorrectly compared to full-epoch
historical hashes. Formal training compares each whole epoch to its original
completed history. This checks implementation and pairing,not effectiveness.

Only physicalGPU0/1/2 are authorized. Start up to6workers percard (18total),
subject to measured VRAM/RAM/disk reserves and staggered admission. The previous
36-worker trial had no throughput advantage,so idle VRAM is not treated as proof
that more concurrency helps. Do not stop or modify unrelated tasks. Worker
sessions are detached from the controller to preserve them across terminal exit.

```bash
PYTHONPATH=/path/to/new/source_overlay \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_signature_noarea_iemocap68_20261025.scheduler \
 --root /path/to/new/execution_v1 --source-commit AUDIT_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 6
```

After all40jobs,the scheduler validates saved predictions/selection/hashes and
automatically compares ACC AND UA against both original Flat and normal
Signature. If no-area matches or exceeds normal,or the two metrics disagree,
report that result rather than declaring path direction essential. A normal
Signature advantage would support only this particular controlled comparison,
not establish that all improvements arise from order or that Signature is novel.

## CPU test backend boundary

Local Torch1.13 CPU MKLDNN intermittently produced nonfinite backward gradients
for both the original and NoArea Signature in repeated random-seed probes.
Disabling MKLDNN for the scoped CPU gradient-connectivity test eliminated these
failures in a20-seed probe. The test restores the backend flag; no numerical
training source,remote environment or GPU training setting is changed. Actual
CUDA gradients are checked in the real-data smoke and every training update.
