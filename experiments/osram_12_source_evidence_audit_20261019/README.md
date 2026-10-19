# OSRAM 12-Source Evidence Audit

INTERNAL DIAGNOSTIC ONLY. User-approved finite frozen analysis, not training.

## Fixed design and execution plan

1. Reuse protected original Flat seed66 miss0.3 anchors: MELD lr1e-5/wd1e-5
   and IEMOCAPSix fold5. Strict-load current code; verify exact historical
   miss0.3 logits before any capture. No TASR Adapter or other second Block.
2. Extend existing joint Value-slot source tracking with audit-only Base reads
   and decayed/prewrite source-state norms. Shared Memory remains authoritative;
   do not change K/V, Query, joint solve, forgetting, write or read order.
3. Replay saved original GCNet masks for eight random rates; separately apply
   seven Persistent patterns. Same anchor across all15 conditions per dataset.
4. One original scan per batch/condition. Capture actual Flat Local and input.
   Recompute only original Flat + norm + classifier for12 source removals and
   eight A/T/V source coalitions (full output already available).
5. Validate reconstructed reads, identity/logit/RNG parity, padding/inactive
   slots, exact Shapley efficiency and negative-control zero sources before run.
6. Save source reads, per-sample logits/CE/identities/masks/trajectory/state
   norms remotely. Commit only code, configuration and small summary tables.

Rows: Base, Gap-A, Gap-T, Gap-V. Columns: historical Value sources A,T,V.
These are algebraic joint-write Value-slot contributions, not pure independent
modality memories or semantic causal effects.

## Analysis definitions

- Read strength and removal CE/ACC/UA/W-F1 use the same active-row cohort.
  A currently observed target's Gap is N/A, not zero importance. Empty strata
  retain N=0 and null metrics. Base includes first utterances with zero history.
- Removal delta CE = CE_removed - CE_full. Positive means beneficial presence
  under this frozen intervention. Corrections/harms describe removal versus full.
- Cancellation =1 - norm(sum sources)/(sum source norms+epsilon), computed only
  for active rows with nonzero total source norm; not an emotion-conflict score.
- Exact3-player Shapley uses per-sample negative CE, source groups shared across
  all four rows, eight subsets with bitA=1,T=2,V=4. Efficiency target is
  CE_empty - CE_full. Independent12-slot ablations are not additive.
- Strata: current observed pattern, historical ever-seen pattern, per-source
  coverage/ever-seen, and outer evaluation condition/rate. No threshold search.
- No p-values across pooled utterances: single seed, one IEMOCAP fold, correlated
  dialogue samples and reused Test-oracle backbone. Vector norm is not utility.
- Readout zeroing/subtraction can be distribution-shifting. Shapley describes
  this fixed classifier/game, not natural causal modality importance.

## Resource and artifact contract

Run on biggpu only, cmerc-repro Python, healthy UUID binding; physicalGPU4 banned.
No optimizer/backward/training, no checkpoint writes or deletion. Two finite jobs
may share one healthy GPU subject to preflight memory. Remote immutable source
snapshot, data/checkpoint/source hashes, mask hashes, exact ordering and timing
recorded per job. Resume only complete per-condition outputs with same identity.

Root runner and analysis modules are separate from the historical experiment.
The new runtime capture flag defaults off, adds no parameters and does not change
legacy checkpoint loading or default RNG. Existing TASR code remains opt-in.

## Commands

In the immutable source snapshot on biggpu (after health/space checks):

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_12_source_evidence_audit_20261019.launch \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
  --output /home/qinjinghui/remote_experiments/osram_12_source_evidence_audit_20261019/runs_v1
```

This launches only two frozen evaluation jobs, not training. The launcher
checks that this UUID is not physicalGPU4 and has at least4GiB free; both jobs
strictly validate their protected checkpoint/data/mask registry. A failed job
does not trigger retraining or protocol changes. `launcher.lock` prevents
duplicate launchers. Use tmux to retain it after disconnect. Per-condition
atomic JSONs support identity/hash-checked recovery, not new model selection.

Local verification (no real data or performance conclusions):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest \
  tests/test_source_evidence_trace.py tests/test_source_evidence_analysis.py \
  tests/test_source_evidence_runner.py tests/test_source_evidence_report.py \
  tests/test_source_evidence_launch.py tests/test_tasr_primitives.py \
  tests/test_tasr_integration.py tests/test_tasr_protocol.py \
  tests/test_tasr_runner.py tests/test_gcnet_legacy_random_mask.py -q
```

Verified74 tests passed. Existing optional torch-sparse/libcusparse warning
remains; these paths do not require sparse operations. Tests include exact
default output/RNG/legacy state dict parity, strict past/no-history/padding,
source reconstruction, source-only final readout edits, loss sign, N/A handling,
nonlinear Shapley efficiency and finite-job/report integrity. A separate bounded
read-only reviewer also exercised actual small OSRAM source capture through the
readout API with a seven-class head (exact full-logit reproduction).

These are implementation checks, not a substitute for real MELD/IEMOCAP runs.
