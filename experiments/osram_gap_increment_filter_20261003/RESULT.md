# Gap-increment Filter

INTERNAL DIAGNOSTIC ONLY

Status: implementation verified; seed66 training launched on biggpu GPU5,
PID 725686, implementation commit 22fdbdf. No performance claim.
Remote log: /data2/yb/remote_experiments/osram_gap_increment_filter_20261003/train.log.
The five recorded implementation source hashes matched the committed local
files before launch. Snapshot is isolated from subsequent working-tree edits.

Measured parameters: Flat 13,509,793; Filter 13,926,434; added 416,641.
All 13,926,434 model parameters remain trainable.

The original Flat anchor is preserved. With the same Local, Base, Gap and one
Memory scan, shared Adapter calls produce uF (full) and uB (Gap zeroed), using
identical dropout. The output is LN(uF + (g-1)(uF-uB)), where
g = 1 + tanh(MLP([LN(uB), LN(uF-uB), availability])). The scalar MLP has
128 hidden units and a zero-initialized final layer. Initialization therefore
recovers Flat exactly. Memory/query/write and the task head are unchanged;
there is no auxiliary loss. All original parameters train jointly.

Reference: original cfg84 full/seed_66 in
/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920.
Implementation parent: 6901379. Original config is copied with only
osram_gap_increment_filter=True. MOSI seed66, 100 epochs, original cyclic
0.0–0.7 missing rates and per-rate BEST Test-oracle selection. Results are
internal screening, not validation-selected paper results.

Verification: 22 tests passed on biggpu V100 GPU5 (verification.log), including
actual cfg84 CUDA identity/RNG, independent readout dropout replay, one scan,
inactive Gap/padding safety, finite updates, checkpoint loading, diagnostics,
runner protocol and existing Relation/dual-readout regressions. Local tests
also verify default-off against the pre-change source. A pre-existing local
GPU exact-equality failure in the old Relation test reproduces before this
change; that test passes on the target V100. Old Relation was not modified.

Run from the isolated remote code snapshot:

```sh
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u experiments/osram_gap_increment_filter_20261003/run.py --output /data2/yb/remote_experiments/osram_gap_increment_filter_20261003/seed_66 --gpu 5 --commit IMPLEMENTATION_COMMIT
```

PROVENANCE.json records source hashes, reference hashes, environment, GPU UUID
and effective config. PARAMETERS.json records measured parameter counts.
Completion requires 100 epochs, eight BEST checkpoints and unchanged evaluation
mask hashes. No automatic multi-seed expansion.
