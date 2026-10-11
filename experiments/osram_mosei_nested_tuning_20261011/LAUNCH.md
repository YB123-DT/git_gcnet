# MOSEI Nested capacity screen launch

INTERNAL DIAGNOSTIC ONLY. Per-rate Test-oracle screening; not a formal paper result.

## Scope

Six CMU-MOSEI seed66 runs, 100 epochs, cfg84 no-JEPA, cyclic missing0.0–0.7. The existing Flat, OSRAM Memory/query/write, task loss and evaluation masks are unchanged. Only the registered Nested configuration differs.

| Host GPU | UUID | Methods | Training PIDs |
|---:|---|---|---|
| 2 | `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4` | `mosei_nested_dim32`, `mosei_nested_depth2`, `mosei_nested_dim32_depth2` | 66054, 66055, 66056 |
| 3 | `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a` | `mosei_nested_mlp128`, `mosei_nested_groups4`, `mosei_nested_dim32_depth2_mlp64` | 66057, 66058, 66059 |

Server: `biggpu` (`user22`). Launch time: 2026-10-11 14:12 UTC. GPU4 was not used. No existing process was terminated.

## Immutable execution inputs

- Code commit: `068457d`.
- Sealed source: `/data1/yb/remote_experiments/osram_mosei_nested_tuning_20261011/source`.
- Existing pinned input manifest: `/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json`.
- Same-seed Flat reference: `/data2/yb/remote_experiments/osram_mosei_cfg84_nojepa_20260919/seed_66`.
- Run root: `/data1/yb/remote_experiments/osram_mosei_nested_tuning_20261011`.
- Persistent tmux socket/session: `gcnet_mosei_nested_tuning_20261011:dispatcher`.
- Dispatcher PID: 64955.

## Verification at launch

- Sealed-source tests: `12 passed` (`tests/test_mosei_nested_tuning.py` and `tests/test_nested_sweep.py`).
- Prelaunch GPU2/GPU3 free memory: 32491 MiB each.
- After all six models loaded, GPU2 used7890 MiB and GPU3 used7990 MiB; utilization was89%/99% at the observation.
- All six child PIDs were alive, all six `PROVENANCE.json` files existed, and `DISPATCH.json` recorded each run as `running`.
- `/data1` had40 GiB free and `/data2` had95 GiB free before launch. The dispatcher requires at least30 GiB on its run filesystem.

## Reproduction command

```bash
cd /data1/yb/remote_experiments/osram_mosei_nested_tuning_20261011/source
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_mosei_nested_tuning_20261011.run \
  --root /data1/yb/remote_experiments/osram_mosei_nested_tuning_20261011 \
  --inputs /data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json
```

This command must not be run again while `DISPATCH.json` exists. Completion requires process exit plus verified artifacts; launch is not a performance result.
