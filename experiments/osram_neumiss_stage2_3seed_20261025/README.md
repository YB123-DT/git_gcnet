# NeuMiss MOSI three-seed completion

INTERNAL TEST-ORACLE SCREEN. Complete only the previously accepted **fixed-rate
frozen-parent NeuMiss** comparison. Do not add Signature,cyclic NeuMiss,HPO,new
datasets or a retrained Flat baseline.

## Scope and unchanged protocol

Reuse the eight completed MOSI66 NeuMiss records in
`/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1/train/neumiss_RATE`.
Its eight-rate ACC is79.6303 and W-F1 is79.6122 versus its frozen Flat79.4604 WF1.
Add ONLY seeds67/68,eachwith8rates. NeuMiss is hard-zero when ATV is fullyvisible,
so rate0.0 is an audited epoch-zero replay,not150fake optimizer updates. There
are16 new logical conditions:14 actual150epoch trainings and2 zero-rate no-ops.
Finalthree-seed report has24 logical seed/rate rows and preserves all negatives.

For eachseed/rate,strictly restore its own accepted original Flat BEST from
the cmerc unified MOSI runs,freeze Encoder/OSRAM/Flat/Norm/head,and train only
the unchanged245984parameter NeuMissBlock. Train ONLY the current missing rate
and evaluate ONLY that rate,using its parent's saved exact mask. Not cyclic,
not persistent,not a one-stage end-to-end run.

Stage2 settings remain150epochs,LR3e-4 forfirst100 then3e-4/10 for50,batch32,
Adam WD1e-5,and original MSE includinggold0. Binary ACC/W-F1 omit gold0 only
during evaluation. Original `gcnet-random-mask-v1`,host/guest,padding,CHM joint
write/Query/read-before-write,Local Skip and data splits stay unchanged.
Select earliest strict current-rate TEST W-F1; ACC/MSE come from the same epoch
and mask. This additional stage2/Test-oracle budget is exploratory,not an
independent generalization claim or one-checkpoint deployment mean.

The reused66 model and all eight artifact/parent/hash/mask/order/metric records
were revalidated before admitting new jobs. Everynew parent uses the preceding
audited source/config/environment bridge and saved original prediction replay.
Initial zero output must exactly reproduce its parent. Frozen Norm/head still
transmit input gradients to NeuMiss; original weights receive none. Selected
branch replay,finite/nonzero gradients,source hashes and frozen-parent digest
are checked. No layer widths,NeuMiss recurrence,initialization or loss change.

## Latest resource/Signature override

The user requested all ouractiveexperiments on onlythree cards,then explicitly
paused Signature to run only NeuMiss. IEMOCAPSignature's controller/workers were
therefore stopped after full checkpoint verification,not deleted or restarted.
PAUSE_SIGNATURE.json records3completed,23paused(withmodel/Adam/RNG/history)
and94unstarted. All23 paused checkpoints were reloaded after worker exit and
selected/parent hashes verified. No owned Signature process remains and no
unrelated process was signalled. Future Signature resume requires explicit user
direction; NeuMiss completion does not automatically restart it.

NeuMiss uses only biggpu physicalGPU0/1/2,with realUUIDbinding and measured
memory/RAM/disk admission. GPU3/7 are not additional spillover cards; GPU4 is
forbidden. Existing unrelatedGPU5/6/7 tasks are untouched. Eachhealthy card
permits atmost6 ofthisqueue's workers;if capacity is insufficient,wait without
changing batch/model/protocol. One real seed67/miss.7 two-update smoke gates
the14formaltrainings and2 no-op replays. No extra baseline training is required.

## Persistent launch

Use an immutable committed source snapshot and cmerc-repro:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_neumiss_stage2_3seed_20261025.scheduler \
 --root /data2/yb/remote_experiments/osram_neumiss_stage2_3seed_20261025/execution_v1 \
 --source-commit ACTUAL_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 6
```

Controller lives in tmux; QUEUE.json retains exactcommands,PIDs,UUIDs,logs and
statuses. New outputs are independent and preserve alloldweights/Mask/history.
Full last.pt contains branch/Adam/epoch/RNG/identity,including epoch-zero no-ops.
Explicit `--resume` adopts exactliveworkers or restores compatible fullstates,
never duplicates them or silently changes source.

Final RESULT.md/JSON/CSV and GROUPS.csv automatically include24 verifiedrows,
perseed/perrate/eight-rate/highmissing means,ownparent pairedchanges and
three-seed sampleSD. Selected metrics are recomputed from saved predictions;
incomplete or invalid results do not form a completed comparison. Allraw
weights,predictions,caches and datasets stay remote and outofGit.
