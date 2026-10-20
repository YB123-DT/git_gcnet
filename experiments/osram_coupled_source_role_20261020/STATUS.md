# Launch verified — final performance pending

INTERNAL TEST-ORACLE SCREEN

Launch snapshot2026-10-20T11:51:43Z, not a live-progress claim:

| Dataset | Formal worker | Progress then | Budget |
|---|---:|---|---:|
| MOSI |2379605|9epochs complete|150|
| MOSEI |2381857|model/masks/config validated; first epoch in progress|60|

Both on biggpu physicalGPU7, UUID`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`;
logicalcuda:0. Persistent controllers2376718/2376721 under tmux socket
`osram_source_role`, sessions`source_role_mosi_f6bc52e` and
`source_role_mosei_f6bc52e`. GPU4 excluded. Existing2e-5 work was not modified.

Frozen source`f6bc52e5e4f83ff13be642595cfa4775ed996e01`. Both effective configs:
source-roleTrue,TASRFalse,originalFlat1600,GCNetrandom-mask-v1,batch32,wd1e-5,
originalMSE,per-rateTest-bestW-F1. One-stage all original weights and96,487new
parameters train jointly; total13,606,280. MOSI LR1e-3/1e-4 at100, MOSEI
constant1e-3. No actual Router despite the queue/metrics legacy aliases.

Both real-data GPU smokes passed exact initial predictions versus the original
Flat, finite/nonzero source-role gradients, actual added and original parameter
updates, full model/Adam/RNG reload and exact selected-prediction replay.
MOSI smoke had14connected but initially only2nonzero added gradient tensors;
by formalepoch9 all14were nonzero. MOSEI smoke had14nonzero tensors. Zero
initialization is not misreported as all new parameters initially learning.

Initial decomposition maximum absolute/relative errors:

| Dataset | SharedMemory=sum sources | Base=sum source reads | Gap=sum source reads |
|---|---|---|---|
| MOSI |4.7684e-7 /2.6533e-7|3.5763e-7 /3.5266e-7|2.9802e-7 /4.5761e-7|
| MOSEI |4.7684e-7 /2.4454e-7|2.3842e-7 /3.4331e-7|1.7881e-7 /3.6794e-7|

Smoke peak allocated MiB:MOSI3126.087,MOSEI1549.257. These are correctness
checks, not sentiment performance or isolated speed benchmarks.
52focused local tests, compileall, CLI flag and whitespace checks passed.
An existing unused optional torch-sparse/libcusparse import warning was
nonfatal. Spec and separate code-quality reviews passed; a final-checkpoint
resume reporting edge case was fixed and regression-tested before deployment.

Both datasets matched their ORIGINAL Flat initialization, split, masks and
seed protocol; only the source-role config switch changed. MOSI had verified
actual training-mask/update/LR parity through9epochs at this observation;
MOSEI full-epoch parity was still pending. Each following epoch checks it.

Output root:
`/data2/yb/remote_experiments/osram_coupled_source_role_20261020/execution_v1`.
Per-dataset queues automatically retain8BEST and matched selected predictions/
masks plus full last.pt, then recompute and publish paired ACC/W-F1 per-rate,
mean andhigh reports ONLY after their complete budget finishes. No old files
were overwritten, no extra seeds or HPO were submitted. LAUNCH.json records
source archives/config/reference hashes, numerical checks and exact scope.
