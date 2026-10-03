# Gap-increment Filter

INTERNAL DIAGNOSTIC ONLY

Status: all three seeds completed 100 epochs. Eight BEST checkpoints per seed
and exact baseline evaluation-mask hashes verified. This variant did not improve
the baseline. Seed66 implementation commit: 22fdbdf, PID 725686 on GPU5.
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
mask hashes.

## User-authorized three-seed extension

User explicitly requested seeds 66/67/68 after the initial launch. Seed66 remains
on GPU5 with its immutable original snapshot. Seeds67/68 use commit 38883d0
and code_multiseed, with identical model files and only seed-aware launcher and
GPU whitelist changes. Each loads its corresponding original Flat configuration.

| Seed | Host GPU | PID | Log |
|---|---|---|---|
| 66 | 5 | 725686 | train.log |
| 67 | 0 | 1335471 | seed_67_gpu0.log |
| 68 | 0 | 1369299 | seed_68_gpu0.log |

Logs live under the remote experiment root above. All runs are from scratch,
100 epochs, with separate checkpoints; no loss/hyperparameter change. GPU0 UUID
is GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45. Seed67's first GPU5 attempt was
blocked before training by the free-memory guard (4091 MiB); that failed launch
log is retained as seed_67.log. The two additional runs were moved to free GPU0
on the same server, not to a different machine. Three launcher tests passed.
The table records the original launch PIDs; completion is verified in the
archived results/seed_*/PROVENANCE.json and metrics.json.

## Completed results

W-F1 (%); differences are percentage points. Per-rate BEST Test-oracle,
INTERNAL DIAGNOSTIC ONLY. Arithmetic mean across rates within each seed,
then equal-weight mean across seeds. High missing means rates .5/.6/.7.

| Seed | Flat 8-rate | Filter 8-rate | Delta | Flat high | Filter high | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 66 | 81.068 | 79.659 | -1.409 | 76.352 | 74.716 | -1.636 |
| 67 | 80.556 | 79.726 | -0.830 | 75.990 | 75.477 | -0.512 |
| 68 | 80.053 | 79.351 | -0.702 | 74.440 | 73.904 | -0.536 |
| Mean | 80.559 | 79.579 | -0.980 | 75.594 | 74.699 | -0.895 |

Sample SD across seeds (ddof=1): 8-rate Flat 0.507, Filter 0.200;
high-missing Flat 1.016, Filter 0.787. No significance claim with these three seeds.

| Missing rate | Flat mean | Filter mean | Delta |
|---|---:|---:|---:|
| .0 | 88.419 | 87.084 | -1.335 |
| .1 | 85.845 | 85.033 | -0.812 |
| .2 | 83.431 | 82.259 | -1.172 |
| .3 | 80.999 | 80.216 | -0.783 |
| .4 | 78.996 | 77.941 | -1.055 |
| .5 | 77.327 | 75.490 | -1.837 |
| .6 | 75.848 | 75.395 | -0.453 |
| .7 | 73.607 | 73.213 | -0.394 |

Observed gate behavior at selected checkpoints, rates .1–.7: mean g ranges
0.108–0.269 (seed66), 0.015–0.656 (seed67), 0.002–0.011 (seed68).
For seed68 every valid utterance has g<0.1 at all seven selected evaluations.
These aggregates include tokens without active Gap; they are not an active-Gap-only
statistic. The gate tends to suppress the learned Gap increment, most clearly
in seed68. This is an observed failure pattern, not proof that gate saturation
caused the entire performance loss.

At rate .0 the measured Gap increment and modulation are exactly zero, yet
performance drops. Thus joint training also changed the original main path;
the total loss cannot be attributed solely to direct inference-time filtering.
Do not interpret suppression as evidence that original Gap is useless.
Recommendation: do not expand this configuration; retain original Flat baseline.
