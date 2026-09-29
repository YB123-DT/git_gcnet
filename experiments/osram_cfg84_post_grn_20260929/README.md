# Optional post-Flat GRN — implementation verified

Status: **implementation and small-scale verification complete**. The user
subsequently authorized full training ("启动啊", then "继续吧"). The following
launch record supersedes the original implementation-only scope below.

## Authorized full experiment

Three seeds 66/67/68, 100 epochs, biggpu host GPU0, three concurrent jobs.
Only `osram_post_grn=True` changes relative to each completed original cfg84
Flat config; Flat readout, no-JEPA, random cyclic missingness, loss and task
head remain unchanged. Original Flat training is reused, not rerun. The
inherited per-rate Test-oracle protocol is **internal diagnostic only**, not
validation-selected paper evidence. Eight best checkpoints per seed remain
saved; canonical test masks are checked against the original reference.

Launch directory: `/data1/yb/remote_experiments/osram_cfg84_post_grn_20260929`.
The same-server output partition changed because `/data2` was full at the
initial preflight. A later check found space had been released on `/data2`;
the approved `/data1` destination is retained. No old files were deleted.
Its `code/` is copied from the already verified isolated snapshot; only a
launcher was added. Dataset and Python environment stay at the original
`/data2` paths. `runs/launch.json`, `children.json`, per-seed logs and
PROVENANCE.json record process IDs and running/completed/failed states.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_cfg84_post_grn_20260929/run.py --launch --max-tasks-per-gpu 3
```

## Interface and semantics

Keep `--osram-readout-fusion flat`; enable only `--osram-post-grn` (default
off). `PostGRN` is called after Flat's original emotion_norm and before the
unchanged task head. All existing Flat and memory parameters stay trainable.
The prior memory-shift-residual path remains optional and is not enabled or
combined with post-GRN. Non-Flat + post-GRN is rejected.

The condition has latent_dim + 4 * context_dim + 3 components: Local,
already-readout-ablated Base, three fixed missing-only Gap slots, availability.
Padding/inactive slots use torch.where before affine operations. Condition
LayerNorm precedes Linear_c; Linear_x and Linear_c map to128, ELU precedes
Linear2(128,128), then Dropout uses the existing backbone dropout probability.
Gate and value project to actual osram.output_dim (cfg84:1600), not500.
Only value weight/bias are zero-initialized; gate bias is zero. The branch
adds sigmoid(gate) * value to x with no further LayerNorm.

Creating the optional module preserves existing parameter initialization via
fork_rng. With the flag off, state keys, forward/backward and RNG behavior
remain unchanged. With the flag on, initial outputs are exactly equal to
Flat for matched upstream randomness; the added training Dropout normally
consumes RNG. This is not a claim of identical subsequent training RNG.

Diagnostics in training history and per-rate metrics, under `post_grn`:

- `gate_mean`: mean over valid utterances and output channels.
- `gate_saturation_fraction`: fraction <=0.05 or >=0.95 on valid channels.
- `gated_residual_flat_norm_ratio`: mean per-valid-utterance L2 residual norm
  divided by max(L2 flat_hidden norm,1e-8).
- `valid_count`: weighting count; all-padding batches do not bias aggregates.

## Verification evidence

94 remote tests passed (one pre-existing PyG deprecation warning). Checks
include flag-off historical Flat behavior, exact zero-init train/eval output
at actual cfg84 CUDA dimensions, inactive Gap/NaN/padding isolation, readout
ablation routing, no freezes, formula/diagnostic correctness, and finite
gradients plus parameter updates over several steps. Byte compilation and
git diff --check also passed.

One-epoch seed66 smoke reused the original cfg84 config, changing only
osram_post_grn=True and the explicitly shortened epochs=1 verification budget.
Completed with 8 checkpoint artifacts, matching canonical test masks, JEPA
loss and target count both0. The new value weight became nonzero and finite
(204800 nonzero elements), confirming the branch is updated by the actual
training optimizer. All diagnostics covered1284 train /686 test utterances.
These smoke metrics are **not evidence of performance improvement**.

## Provenance and artifacts

- Model server: `ssh biggpu`, hostname user22; no server migration.
- Host GPU0 UUID: GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45.
  CUDA_VISIBLE_DEVICES=0 maps it to logical cuda:0. Host GPU4 is forbidden.
- Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`,
  Torch2.2.2+cu121.
- Source snapshot: `/data2/yb/remote_experiments/osram_cfg84_post_grn_20260929/code`.
- Smoke: `/data2/yb/remote_experiments/osram_cfg84_post_grn_20260929/smoke/seed_66`.
- Baseline: `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66`.
- Config/source hashes and status: `smoke/seed_66/VERIFICATION.json`.
- Local source branch: feature/osram-uniform-forced-text at4eef3f5; dirty
  changes preserved, no branch switch. `source-dirty.patch` records tracked
  differences including pre-existing edits, not solely this task's diff.

Verification command, run inside the isolated snapshot with the project
dataset root and CUDA_VISIBLE_DEVICES=0:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_cfg84_post_grn_20260929/verify.py
```

This verification script refuses to overwrite the completed smoke directory.
The full run uses the separate `run.py` and `/data1` output root above.
