# Authorized relocation to GPU1

INTERNAL TEST-ORACLE SCREEN; no new performance claim.

The user authorized GPU1 to reduce contention on GPU2. The scope stays exactly
the existing four MOSEI seed66 runs; no extra model, seed, Flat, or HPO trial.

- Server: `biggpu`, unchanged.
- Numerical source: `0867fb061c965326e95ec06b29a6239c867cc70a`, unchanged immutable
  `source_v1`; manifest SHA256
  `375662cb6e7088b8eaa801df7d9f350ac7e832af8aec18a7ebf03cda3007106f`.
- Orchestration implementation: `5208507` (`migrate_gpu1.py`, deployed outside
  the immutable numerical source).
- Target: physical GPU1, `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`;
  each worker sees only `cuda:0`. GPU4 remains forbidden.
- Environment: `/data2/yb/conda_envs/cmerc-repro/bin/python`, unchanged.
- Persistent session: `mosei_nested_local_gpu1_20261024`.
- New controller PID at launch: `501279`.

Remote root:
`/data2/yb/remote_experiments/osram_mosei_nested_local_20261024`.
New output: `execution_gpu1_v1`; original `execution_v1` is retained.
Controller log: `migration_gpu1_v1.log`.

## Verified transfer

| Variant | Recovery | Saved epoch | New PID at launch |
|---|---|---:|---:|
| nested32_lr5e5 | Full epoch recovery | 14 | 503359 |
| nested32_lr2e4 | Full epoch recovery | 12 | 503860 |
| nested32_depth2 | Full epoch recovery | 13 | 503966 |
| nested32_groups4 | One fresh retry after pre-epoch CUDA OOM | N/A | 504430 |

All three copies passed exact last-checkpoint SHA256, selected checkpoint and
prediction hashes, epoch/history length, nonempty Adam state, and single CUDA
RNG-state checks before any original process was terminated. Exact owned PID,
command line, and source directory are checked before signals. All original
files are preserved. Original queue metadata is copied to
`execution_gpu1_v1/ORIGINAL_QUEUE.json`; the old controller ledger is marked
`migrated_to_gpu1`, not successful completion.

`execution_gpu1_v1/RECOVERY.json` contains these last-checkpoint SHA256 values:

- lr5e5: `ff1149305d470d2e043bb14e5499a0a81724d1a3b837e12ba39da2a7e9f0582c`
- lr2e4: `18fba0f3e87be7a9b25826e6a0f130c17b9b8f56d099538bd6b63fc14810bd8e`
- depth2: `ecb38e1dd6166ff9f72e1d7145bd95ed7740f52c9b64d9a4eeaff6eac2ce1393`

The three workers use the unchanged worker's strict `--resume` path: model,
Adam, epoch, history, selection ledger, and CPU/CUDA RNG are restored. Work in
an unfinished epoch is discarded. Both cards are V100-SXM2-32GB, but cross-card
bitwise continuation is not claimed. Recovered run `seconds` fields measure
the new segment only; original elapsed times remain in the preserved original
SUMMARY files. Do not use those segment times alone for total-cost comparisons.

Training/mask/optimizer/numerical source settings are not modified. Maximum
concurrency is four with 6GiB per worker plus 2GiB global margin, actual external
allocations charged to admission, 64GiB RAM and 25GiB disk minimums. Foreign
tasks are untouched. A second failure is not silently retried.

Validation: 30 local migration-plan/PID-ownership/config/report tests passed;
syntax and diff checks passed. At the initial receipt, all four new processes
were alive and loading the shared feature cache. Completed training and new
post-resume epochs must be verified separately from this launch receipt.

When all four finish, the existing strict report writer automatically validates
the four candidates and two read-only controls, then writes RESULT.md/JSON and
48 per-rate CSV rows in `execution_gpu1_v1`. A partial run is not ranked as
completed.
