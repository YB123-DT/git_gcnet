# Verified finite queue launch

INTERNAL TEST-ORACLE SCREEN. Running; not a final performance result.

- Server biggpu, controller started 2026-10-17T14:00:31.522Z.
- Immutable driver source `e0893a1d4a2d83f30259faab3f9aab9cdfc63eac`.
- Immutable numerical source `b4a313841f098487a6482d2712bdac94aa8e319e`.
- Driver: `/data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/driver_v1`.
- Output: `/data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/runs_v1`.
- tmux: `meld_cosine_3seed_v1`; controller PID3554251.
- Physical GPU5 only: `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`, logical cuda:0.
- Python environment unchanged: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
- Controller log: `/data1/yb/remote_experiments/osram_meld_cosine_step03_3seed_20261017/controller_v1.log`.

## Actual jobs at launch snapshot

| Job | Seed | Write step | State | PID | Committed epoch |
|---|---:|---:|---|---:|---:|
| C67 | 67 | 0.6 | running | 3554441 | 1 |
| C68 | 68 | 0.6 | running | 3554450 | 2 |
| S66 | 66 | 0.3 | pending | — | — |
| S67 | 67 | 0.3 | pending | — | — |
| S68 | 68 | 0.3 | pending | — | — |

The existing completed cosine seed66 is reused from
`/data2/yb/remote_experiments/osram_meld_cosine_20261017/runs_v1/C0`.
It is not retrained. C67/C68 are genuinely running new jobs; the three S jobs
are pending phase1 completion, not falsely reported as running.
The persistent finite controller automatically admits all three S jobs after
both C jobs complete, then stops after all five new jobs end.

## Verification

- 25 focused/regression tests passed locally and in remote cmerc-repro before deployment. Compileall and staged whitespace checks passed.
- Both new cosine seeds completed an actual train/backward/optimizer/eight-rate evaluation/atomic save epoch. Initial hashes differ across seeds as expected; parameters remain13,290,795.
- Actual epoch1 LRs were [.0005,.0005,.0005]; epoch2 LRs [.001,.001,.001] under the unchanged existing schedule.
- All eight selected evaluations contain2610 valid test utterances. Both ACC-best and UA-best plus full last-training state were saved; no added inference or sampler calls.
- C67/C68 do not have fresh same-seed constant-LR controls, so paired initialization/mask flags are intentionally null, not a failed check or a claim of equality across seeds.
- When S jobs start, they will check their initialization and all actual train/test mask hashes against the same-seed cosine reference (C0/C67/C68).
- No failures at the saved snapshot. No other tasks were stopped. Known optional torch-sparse import warning remains; actual OSRAM training ran successfully.
- Same ECERC data, missing protocol,50epochs,batch32,loss,model and environment. New artifacts use /data1 because /data2 was low on space; no old checkpoints were deleted.

`LAUNCH_STATUS.json` is an immutable startup evidence snapshot, not live progress.
The queue and per-job EPOCH_STATUS.json on biggpu are the authoritative live
state. No scores from these initial epochs are treated as method results.
