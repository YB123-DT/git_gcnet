# Launch receipt: 2026-10-26 01:48 UTC

Status **running**, not completed performance evidence. Frozen original Nested,
additional seeds67/68, all four datasets; preserve and later combine completed66.
128 new own-rate/fold branches, 64 protected66 results, 192 total selections.

- Source849079bba363ed048a6a87f409f9466c06ac1d52, uploaded to feature/ecerc-features.
- Serverbiggpu, cmerc-repro Python/Torch1.13.1+cu117/NumPy1.26.4/sklearn1.3.2.
- Immutable overlay `/data2/yb/remote_experiments/osram_frozen_nested_3seed_20261026/source_849079b`.
- Output `/data2/yb/remote_experiments/osram_frozen_nested_3seed_20261026/execution_v1`.
- Persistent tmux `nested_3seed_20261026`, controllerPID1122323.
- GPU0/1/2 UUID-bound, max4workers/card, max12total. GPU4/unrelated jobs untouched.
- At capture:12 formal runs running,116pending, no failures; automatic finite refill.
- Trainable159235-parameter original Nested only; frozen own-seed Flat not retrained.
- Original GCNet-random-mask-v1, own protected evaluation masks and Test-best unchanged.

## Verification actually executed

Local49 focused tests passed. Remote cmerc58 tests passed after adding two
missing test fixtures to the initial overlay before any training (protected MELD
registry JSON and NoArea fixture module); no numerical code change was needed.
One inherited optional torch-sparse/libcusparse.so.11 warning remains. The actual
Nested uses the tested native tensor path; no environment change was made.

Remote CPU preflight verified128/128 own-parent checkpoint/prediction/mask/config/
environment/data identities, zero failures. Full preflight JSON is remote;
path and SHA256 are in LAUNCH.json.

All8 real GPU smokes passed: both new seeds ×fourdatasets, IEMOCAPfold5.
Each completed2 original-task backward/Adam updates, exact initial Flat logits,
protected-mask replay, single live CHM scan, connected finite Nested gradients,
nonzero new-parameter updates, unchanged parent digest/no parent gradients,
selected-checkpoint reload and all-array replay. These checks demonstrate
implementation correctness, not improved accuracy.

At capture MOSI67 reached30/150, MOSI68 reached17/150, IEMOCAP67fold1 reached9/100,
MELD67 reached2/50; these are progress snapshots for their first rate branches,
not complete eight-rate scores. No premature three-seed averages are reported.
MOSEI was still completing its first formal epoch; its concurrency gate remains
closed until a full actual cyclic epoch confirms measured capacity.

## Next automatic actions

Queue drains all128 approved branches without another user launch command.
No seed66 rerun, HPO, additional module, dataset or IEMOCAP4 experiment is added.
Final collector validates every best artifact, own parent and saved mask, then
combines protectedseed66 with67/68. Report per-seed/per-rate and equal-fivefold
ACC/UA/W-F1, three-seed mean/sample std and high-missing results. A failure is
recorded and stops new admission without killing already-running workers.

Use QUEUE.json and each worker's SUMMARY/history/log for fresh progress; the
snapshot LAUNCH.json is not live status. `--resume` requires compatible complete
block/Adam/RNG/history state. Test-oracle results remain internal exploration.
