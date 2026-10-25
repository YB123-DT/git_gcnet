# MOSI66: recent NoArea parallel/serial, joint/frozen

Approved four configurations. **NoArea retains S1 and symmetric S2**,not
first-order-only. No CHM/Query/write/order/missing changes. Serial updates
Base/activeGap forward reads BEFORE the original Flat;parallel retains the exact
previous1600-dimensional pre-Norm residual. Local and LocalSkip remain original.
The frozen and joint training stages are separate experimental questions.

## Actual architecture

Both variants reuse the strict-past8-utterance modality paths,projecting each
real observed256d latent to8d and appending fixed-window time. Missing positions
are not synthetic zero observations. NoArea second level is0.5*S1 outer S1;
antisymmetric directional area is absent at both TRAIN and TEST.

Parallel is the exact existing NoAreaSignatureBlock,281856parameters,including
currentLocal/availability/count conditioning and zero-initialized1600exit.

Serial history = signature270 + counts3. Project273->64 withoutbias+GELU;
Local256->64,shared read512->64,role4x8. Shared203->128->64 nonlinear interaction
takes currentLocal,one original read,history,currentavailability and role.
Subtract the same interaction with projectedhistory=0. Four zero-initialized
bias-free64->512decoders map the difference to Base/Gap corrections. Only valid
recent-history Base and currently missing Gap roles update. Backward512 stays
unchangedzero. Return modifiedBase/Gap to the ORIGINAL4352input Flat;no global
1600serial correction,no new head,Gate,query or auxiliary loss. Serial238798
parameters:these are not capacity-matched models or a pure wiring-only ablation.

Temporary hooks capture the already-computed real latents and original reads.
Only one Encoder and causalMemory scan runs per model forward. Hooks are removed
even on exceptions. In frozen mode,parent weights stay frozen/eval,but the task
gradient still passes through frozenFlat/Norm/head to the new block. No labels
enter any block input. No forward caches are shared across conversations/views.

## Finite training scope

| Stage | Configurations | New training jobs | Budget |
| --- | --- | ---: | --- |
| Stage1 joint | parallel,serial |2|150epochs each,one cyclic model evaluates8rates|
| Stage2 frozen | parallel,serial |16|8own-rate parents per mode,150branch epochs each|

All18formaljobs are MOSIseed66 only. Stage2 has8branches per mode to retain exact
own-rate frozen Flat parents,not8new random seeds. Oldweights/results remain.
The previously completed stage2parallel run used literal3e-5 in the final50epochs;
this round reruns pairedparallel explicitly with exact3e-4/10 for both modes,
rather than silently treating the adjacent floats as identical provenance.

biggpu,cmerc-repro,officialdata/split/feature hashes,batch32,AdamWD1e-5,
originalMSEincludinggold0,originalgradientclip,cyclicGCNet-random-mask-v1.
LR3e-4 for100epochs then3e-4/10 for50. OriginalLocalSkip,Flat1600,CHM8heads64d,
jointAVTwrite_step.6,causalread-before-write unchanged. Same acceptedFlat exact
testmasks. Per-rate earliest strict Test-bestWF1;ACC from sameepoch/predictions.
All figures are INTERNAL TEST-ORACLE exploration,not independent test evidence.

Stage1 all parent/new params train from original matched random initialization.
Stage2 loads each accepted same-rate Flat BEST,freezes every originalparameter,
trains only newblock. Additional budgets and differentparent selection prohibit
pooling stage1/stage2 as one ranking. Eight-rate means can combine BEST epochs;
stage2 also combines8differentparent/branch pairs,not one deployment checkpoint.

Both originalFlat adapter and serialdecoder initialize with zero exits. Serial
can legitimately have exactlyzero taskgradients on the FIRST joint update;
subsequent updates must open the path and produce actual nonzerotaskgradients
and parameter changes. Tests check this explicitly;zero is never mislabeled as
disconnected or reported as proof of successful learning by itself.

## Execution and validation

Numerical source e0ca94c4c33de0efda2c2f4786fd289067e1ab2c stays immutable at
`/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
New experiment and sharedNoArea block use a separate committed namespace overlay.
Do not edit old code/checkpoints or install/change packages. Optionaltorch-sparse
import warnings are inherited;the plainOSRAM path does not use that graph op.

4real GPU smokes (stage xmode) precede formal jobs. GPU0/1/2UUIDs only;GPU4
forbidden;max6workers/card subject to measuredVRAM/RAM/disk reserve and stagger.
No otherusers' tasks are signalled. Persistent tmux queue records all commands,
PIDs,logs,status,sourceSHA and actual resource usage. Fullmodel/block,Adam,RNG,
epoch/history/selected state supports explicit compatible resume. Admissions
stop on failure while live jobs and incomplete artifacts are preserved.

```bash
PYTHONPATH=/path/to/immutable/overlay \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_recent_noarea_serial_20261025.scheduler \
 --root /path/to/new/execution_v1 --source-commit ACTUAL_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 6
```

Final RESULT.md/JSON/CSV automatically validates saved selected hashes,
predictions/labels/masks,earliestWF1 epochs,parametercounts and unchanged
protocol. Report all8rates and high0.5–0.7 separately bystage andmode,with
branch-off sensitivities,corrections/harms,parameter/time/VRAM and failedjobs.
Neither smallsmoke success nor a score gain establishes novelty or proves why
recent descriptors help. Unchanged/mixed/negative results must be retained.
