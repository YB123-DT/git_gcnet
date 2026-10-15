# Replication execution record

INTERNAL TEST-ORACLE THREE-SEED REPLICATION. Not a completion or accuracy claim.

Immutable source: `ab3d7e62` (full SHA saved in each worker provenance).
Source worktree on biggpu:
`/home/qinjinghui/remote_experiments/osram_meld_block2_3seed_code_ab3d7e6`.
The three seed66 sources remain immutable `7abb855`; numerical core source
hashes are verified equal before the queue is admitted.

Persistent nohup controller PID: `3345356`.
Output root:
`/home/qinjinghui/remote_experiments/osram_meld_block2_topacc_3seed_20261015`.

| Job | Host GPU | Initial PID |
|---|---:|---:|
| trial16 seed67 | 2 | 3345360 |
| trial16 seed68 | 3 | 3345436 |
| trial13 seed67 | 0 | 3345492 |
| trial13 seed68 | 2 | 3345584 |
| trial12 seed67 | 3 | 3345642 |
| trial12 seed68 | 0 | 3345738 |

These are observed running queue entries, not pending launch requests. Live
state, birth-checked PIDs, commands, UUIDs and logs are in `queue/QUEUE.json`.
GPU4 and its known bad UUID are excluded. The existing source7abb855 HPO
controller remains running, unchanged.

Verification: five newly written replication tests first failed for the missing
guard/launcher, then passed. Local configuration/queue suite:23 passed,3 skipped,
14 subtests. Remote CPU regression suite:68 passed, zero failures. The one
torch-sparse/libcusparse import warning disables an optional legacy extension;
these OSRAM/B6 jobs do not use it. Original architecture GPU smoke is reused
because no model/trainer/sampler computation was changed.

Models are trained from scratch at seeds67/68; seed66 is reused, not resumed or
trained again. Completion must be confirmed by all six complete worker summaries
and final `RESULT.md`/`SUMMARY.json`, not merely by this startup record.
