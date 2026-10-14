# MELD write/retention diagnostic status

**INTERNAL DIAGNOSTIC ONLY — both real-data diagnostics COMPLETE.**

Original weights were unavailable; the authorized reproduction used original
commit `b4a3138` and the original seed66/50-epoch config. Audit code `27ae943`
ran on biggpu GPU0. Each of eight rates covered all 2,610 test utterances;
4,082,240 source/head records were written. Replayed classes and labels
matched the reproduction's own saved artifacts exactly at every rate, and
model parameter hashes were unchanged. Normal fork state parity was checked
on every replay step. All eight best checkpoints are retained.

## Findings

Numbers below are macro means over the eight rate-specific checkpoints;
within a rate, scores average source/head pairs. A/T/V are source modalities.
Relative error is `norm(M @ source_key - source_value) / norm(source_value)`.
Centered cosine subtracts the fixed four-candidate mean, not a train mean.

### Immediate joint-write effect

| Source | Relative error before | Relative error after | Candidate-centered cosine after | Four-candidate top1 after |
|---|---:|---:|---:|---:|
| A | 0.5840 | 0.2343 | 0.7605 | 87.38% |
| T | 0.9490 | 0.3808 | 0.9252 | 96.35% |
| V | 0.5057 | 0.2029 | 0.5346 | 69.40% |

These results show that real joint writes change the state in a direction
that supports reconstructing the inserted Value with its own Key. This is
consistent with the block-delta update objective and does not demonstrate
emotion usefulness. Partial reconstruction is expected at write_step=0.6.

### Four future branches, distance4

The four columns use exactly the same source/target pairs at this distance.
Measurements are after future retention and before the target's write.

| Source | Normal error | Decay-only error | Write-only error | Held-state error |
|---|---:|---:|---:|---:|
| A | 0.4085 | 0.2968 | 0.4014 | 0.2432 |
| T | 0.7259 | 0.4322 | 0.7107 | 0.3826 |
| V | 0.3350 | 0.2750 | 0.3258 | 0.2148 |

For Text, paired error increases from the same source's immediate post-write
state are normal +0.3433, decay-only +0.0496, write-only +0.3281, hold +0.
Normal exceeds decay-only, and write-only exceeds hold, at every rate for
Text at distances2/4/8. These effects are not additive; no percentage of
causal task error is attributed to either mechanism.

| Text, distance4 | Normal | Decay-only | Write-only | Hold |
|---|---:|---:|---:|---:|
| Candidate-centered cosine | 0.4089 | 0.8722 | 0.4379 | 0.9251 |
| Four-candidate top1 | 55.26% | 92.17% | 56.93% | 96.00% |

At distance8, Text top1 is 45.51/87.04/47.95/95.84% respectively.
Compare branches at the same distance. The source cohort changes across
distances due to conversation boundaries; held-state mean changes therefore
reflect cohort composition, not state evolution.

## Interpretation boundary

- The result does not support a blanket claim that observations never enter
  Memory. Immediate source-addressed reconstruction improves for all modalities.
- Later real writes substantially change older Text associations, even when
  retention is disabled. This is a concrete storage-interference observation.
- It does not prove that overwriting is harmful to emotion prediction. Old
  utterances may be irrelevant or outdated; the task does not require perfect
  source recall. No task prediction was recomputed under these state forks.
- Original Base/Gap Query quality and Flat decision use remain unlocalized.
- Four-candidate metrics only cover sources with three preceding observed
  same-modality utterances. At Text distance4 there are 33,728 eligible
  source/head records over rates (4,216 source instances), versus 65,808
  total records (8,226 source instances). These are repeated observations,
  not independent samples or a multi-seed performance study.
- Wrong-Key is the preceding candidate Key. It is not random, and correlated
  candidate Values can make raw error misleading. Candidate-centered cosine
  and retrieval are reported alongside error, with fractional tie handling.

## Reproduction and processing verification

| Internal selection metric | Historical seed66 | Reproduced seed66 | Delta (pp) |
|---|---:|---:|---:|
| Eight-rate ACC | 60.637 | 60.675 | +0.038 |
| Eight-rate W-F1 | 57.087 | 56.807 | -0.280 |

Saved masks and labels match historical artifacts at all eight rates; saved
predictions differ. This is a new reproduction, not restoration of the exact
deleted model. Per-rate selected epochs are 11/11/5/21/5/34/24/10.

Training took 1,084.80s; the audit process took 511.85s. The remote analysis
completed `AGGREGATE.csv` but then failed because its environment lacks
matplotlib. Its original failure record is preserved in `PIPELINE.json`.
Rendering was completed locally with existing matplotlib3.10.9 via the CSV
rendering entrypoint; no new inference or dependency installation was needed.

Artifacts: `AGGREGATE.csv`, `RUN_SUMMARY.json`,
`REPRODUCTION_COMPARISON.json`, `PIPELINE.json`, and `figures/` in this directory.
Full source/head rows remain at the remote output below. Plot exports include
immediate write, absolute reconstruction error, paired error increase,
candidate-centered cosine and retrieval, in PNG/PDF. Five correctness tests
passed before execution; a read-only review found no concrete state-order bug.

## Initial missing-weight blocker (historical record)

The following records describe the initial blocker, subsequently resolved
through the user's authorized baseline reproduction.

The original seed66 `gcnet-random-mask-v1` Flat source directory on biggpu
contains config, history, metrics and all eight saved prediction/mask files,
but no `best_miss_0p0.pt` through `best_miss_0p7.pt`.

Its `PROVENANCE.json` explicitly records `discarded_checkpoints.count=8`,
`discarded_checkpoints.bytes=425530104`. Read-only backup searches under
`/data2/yb` found old-mask Flat and canonical no-Local-skip weights but no
matching original canonical Flat weights. No alternative model was used.

Source:
`/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66`

Implemented both authorized diagnostics, runtime checks, streaming rows,
CSV aggregation and plotting entrypoint. Synthetic tests check exact original
forward parity, normal state replay parity, analytic decay, constant hold,
strict-past candidate pools and capture restoration. These do not establish
MELD performance or diagnose its learned weights.

Verification on biggpu with the established PyTorch 1.13.1+cu117 environment:
`python -m pytest tests/test_meld_write_retention.py -q` — **4 passed (1.98s)**.
Local synthetic checks also passed under PyTorch 1.12.0+cu113. Syntax checks
passed for audit, runner and analysis. The actual eight-rate runner preflight
was attempted on biggpu and failed with the eight explicit missing checkpoint
paths, before GPU allocation, dataset loading or result-directory creation.

Diagnostic code was staged separately at
`/data2/yb/remote_experiments/osram_meld_write_retention_20261014/code`.
No model training, real-data inference, source checkpoint mutation or deletion
was performed by this task. No performance or learned-state findings exist.

To complete: recover the original eight checkpoints. Saved predictions and
Local/Base/Gap caches cannot reconstruct the Key/Value projectors or full
Memory state. No retraining is authorized in this task.

## Subsequent authorization and recovery plan

The user subsequently authorized searching for backups and reproducing the
original baseline if none exists. Local MELD checkpoint search and a remote
configuration scan under `/data2/yb` found only the canonical original seed66
Flat r0/r1 directories with **zero .pt files**. Available retained MELD
weights belong to different missing-mask protocols or no-Local-skip models.

Implemented `reproduce.py` to train exactly the historical `b4a3138` code and
source config: MELD seed66, 50 epochs, batch32, Adam lr1e-3, cyclic GCNet
random mask, original Flat/Local skip, per-rate Test-oracle. Keep eight best
checkpoints, compare historical/new results, then automatically run both
audits and analysis. This remains a reproduction for internal diagnostics.

Reconstruction statistics were vectorized across source/head pairs to avoid
per-row GPU synchronization. A fifth regression test verifies equivalence to
the scalar metric calculation. Actual job launch/status is recorded below;
no learned-state conclusion is available yet.

## Running reproduction and automatic audit

- Server: `ssh biggpu`, host `user22`, physical GPU0.
- GPU UUID: `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`.
- Started UTC: `2026-10-14T10:30:38`.
- Pipeline PID: `4018218`; baseline training PID: `4018230`.
- Original training code: detached Git worktree at `b4a3138`.
- Diagnostic/pipeline code: detached Git worktree at `27ae943`.
- Output: `/data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138`.
- Status file: `PIPELINE.json`; training log: `training.log`.
- Source config SHA256: `ce88eb4166a18f4bd1539326035c8a4f706fe5a4b6c42115f5b8b6599cccd58d`.
- Startup confirmed epoch001 completed, config/provenance created and CUDA
  execution on GPU0. Tests after vectorization: **5 passed on biggpu (1.98s)**.

This is a new reproduction. Once training succeeds, the same persistent job
runs the audit and analysis without another user command. No observed
write/retention/overwrite conclusion is reported while these stages are pending.
