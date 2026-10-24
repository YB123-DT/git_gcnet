# Verified corrected launch:2026-10-24

INTERNAL TEST-ORACLE SCREEN. No completed60epoch performance claim.

All eight s0 workers and the s0 controller were stopped after exact command,
output and source-directory ownership checks. No files were deleted. Exact
targets and original queue are preserved in CANCELLED_S0.json. New experiments
are random initialization, not s0 continuation or weight transfer.

- Server: biggpu, physical GPUs0/1, four workers each.
- Python actually verified in ALL eight SUMMARY files:
  `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Actual config LR in ALL eight files:0.0001, constant for60epochs; no local LR
  multiplier, warmup or scheduler. Adam/WD1e-5,batch32,MSE,newGCNetmissing unchanged.
- UUID CUDA RNG repair actually verified in ALL eight:runtime_count1,
  original_python_count0,original_random_count0,random_alias_repaired=True.
- Runner SHA:`16dc02075b5b25862725b96366289d9822c3e932`.
- Numerical base:`0867fb061c965326e95ec06b29a6239c867cc70a`, the completed cmerc
  Nested screen snapshot.130numerical files unchanged; only registry gains the
  existing dim-parametric Nested256 option. The training loop is reused through
  explicitly default-disabled setup/scope callbacks.
- Immutable source:
  `/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024/source_v1`.
- Output: same root,`execution_v1`; controller log:`controller_v1.log`.
- Parent SOURCE_MANIFEST.json records1484Python hashes; source archive's
  SOURCE_SNAPSHOT.json separately records the immutable composite and base SHA.
- tmux:`mosei_capacity_cmerc_20261024`; controller PID1437620.

At12:15:05UTC all eight were alive, past preflight and allocated on their
specified GPUs, with no error. First formal epochs were still in progress.

| Physical GPU | Model | Actual PID |
|---:|---|---:|
| 0 | Flat128 | 1437648 |
| 0 | Flat384 | 1437813 |
| 0 | Flat768 | 1438425 |
| 0 | Flat256+Nested64 | 1438928 |
| 1 | Flat256 | 1439077 |
| 1 | Flat256+Nested128 | 1439307 |
| 1 | Flat512+Nested128 | 1440181 |
| 1 | Flat512+Nested256 | 1440337 |

GPU0:`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`;
GPU1:`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
No other worker or GPU4 was touched. All checkpoints/Masks/predictions retained.

Validation:50tests passed both locally and in actual remote cmerc; syntax/diff
checks passed. Model source parity and actual per-worker environment/LR/RNG
were checked. Do not equate startup validation with improved task performance.
The existing torch-sparse optional-import warning is logged; no dependency or
precision changes were made.

Only compare against the matched repaired cmerc Flat1600 control in
`osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.
Do not mix older s0 or unrepaired cmerc scores into this table. The512 Nested
arms do not yet have a same-width cmerc plain512 in this finite budget; that
delta stays N/A. There is no automatic ninth job or extra seed.
