# MOSI head48 three-seed confirmation

INTERNAL TEST-ORACLE SCREEN

User-authorized finite supplement: reuse completed head48 seed66; train only
seeds67/68 from scratch. No repeat66, no new head-width/write-strength search,
no module, new loss, persistent view, or change to the accepted MOSI environment.

## Fixed protocol

- Original cfg84 no-JEPA Flat;8 heads, Key=Value=48 per head, write_step0.6.
- Forward read384, full causal interface768 (backward half remains zero),
  Flat input3328, hidden/output1600. Local and Local Skip unchanged.
- Parameters11245857; this changes Memory AND Base/Gap/Flat input dimensions,
  not a parameter-matched pure Memory-storage ablation.
- MOSI official split, s0 Torch2.2.2+cu121 / NumPy1.24.3 / sklearn1.3.2.
- Adam,batch32,WD1e-5,original dropout,MSE;150epochs.
- LR3e-4 in epochs1–100,3e-5 in101–150.
- `gcnet-random-mask-v1`, original host/guest padded rectangle sampling,
  cyclic rates0.0–0.7; same original test-mask banks across candidates/seeds.
- Per-rate TEST W-F1 best, strict improvement/earliest tie. Every rate retains
  BEST plus matching predictions/availability; full last.pt retains Adam/RNG/history.
- Exact training mask/update/LR ledger checked against each seed's original
  completed Flat. Common-shaped parameters match that seed's fresh UNTRAINED
  original Flat; shape-changed parameters remain fresh. No trained weight initialization.

Seed66 read-only candidate:
`/data2/yb/remote_experiments/osram_mosi_memory_grid_20261022/execution_v1/runs/head_48`.
Its eight-rate/high W-F1 are79.496914/73.893646. Reuse does not create another run.
Original head64 controls reuse the completed same-seed accepted3e-4→3e-5 Flats;
no additional control training is authorized.

The original Flat ledgers omitted `mixed_rate.py` from source hashes. Before
launch, `git show` of their recorded commits5877678/55a5771 and the current
file all yielded SHA256 `0d3e9af0b3c35c0fde4a148851390cf47850a50d9bdb0737a6997dc9438b22c1`.
Only these two explicit Git-object recoveries are permitted and reported;
unknown missing hashes fail. Actual training Mask/update/LR traces must still
match the original same-seed150-epoch ledger. No old records are overwritten.

## Execution

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosi_head48_3seed_20261022.run \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
  --source-commit SOURCE_SHA --output /path/to/fresh/execution
```

Run on biggpu from the immutable recorded source, inside persistent tmux.
Reuse the existing two-job resource-admitted controller: max2 own concurrent
jobs on one healthy GPU; conservative4GiB/job plus2GiB card margin,64GiB host
RAM and12GiB disk headroom. Recheck before each launch; never use bad GPU4,
kill peers, shrink batch, or retry a failure without diagnosis.

The exact head48 architecture already passed a real MOSI two-epoch/four-step
GPU smoke in the completed nine-arm grid (peak921.844MiB, optimizer/RNG/prediction
restore exact). Only seed routing/reference selection is extended; no repeated
GPU smoke or extra experiment is added. Unit checks guard permitted seeds,
configuration, original default behavior and queue scope. Formal first-epoch
logs additionally verify actual mask/gradient/update paths.

`--resume` on the controller is explicit complete-state recovery only; never
silently start another copy. Existing checkpoints and source snapshots remain
untouched. The finite queue automatically writes per-seed/per-rate ACC/W-F1/MSE,
high-missing means, same-seed head64 differences and sample-SD three-seed means
after both new150-epoch jobs complete. Partial results are never labeled3seeds.

The candidate was selected by seed66 test scores. Test-oracle/candidate-selection
bias remains even after completing67/68; do not call this independent paper evidence.
