# Authorized full launch

Code32c6897, branch feature/osram-uniform-forced-text, pushed to github remote.
2026-10-01T10:07:30UTC; biggpu GPU6 TeslaV10032GB,
UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153; coordinatorPID2490138.
Two runs control/contrastive, seed66,100epochs, same paired masks, fixed
drop.2,temperature.1,lambda0/.1. Baseline seed66 reused.

Snapshot `/data1/yb/remote_experiments/osram_paired_history_views_20261001/code`.
Outputs `/data1/yb/remote_experiments/osram_paired_history_views_20261001/runs`.
Committed archive overlaid on isolated environment, no unrelated dirty changes.
Python `/data2/yb/reproduction_workspace/envs/s0/bin/python` (existing torch2.2.2).
Dataset `/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.

```bash
cd /data1/yb/remote_experiments/osram_paired_history_views_20261001/code
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_paired_history_views_20261001/run.py --launch
```

Final regression33passed, one existing PyG warning. Both real-data smoke arms
complete with all8best checkpoints/matching canonical evaluation masks.
Raw launch/preflight files retained locally; per-run config, source hashes,
timestamps, masks and epoch statistics retained remotely.
No early stop, retries or tuning authorized based on score. Preserve all results.
Status/children/logs identify runtime progress; launch is not completion evidence.
Per-rate Test-oracle protocol remains INTERNAL DIAGNOSTIC only.
