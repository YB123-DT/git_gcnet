# Launch record

Code commit b51b927, pushed to github/feature/osram-uniform-forced-text.
Server biggpu, host GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
Coordinator PID3524185. Seeds66/67/68, 100epochs each, up to3concurrent jobs.

Snapshot: `/data1/yb/remote_experiments/osram_memory_only_adapter_20261001/code`
Outputs: `/data1/yb/remote_experiments/osram_memory_only_adapter_20261001/runs`

```bash
cd /data1/yb/remote_experiments/osram_memory_only_adapter_20261001/code
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_memory_only_adapter_20261001/run.py --launch --gpus 6 --max-tasks-per-gpu 3
```

Snapshot created from committed code, excluding unrelated dirty gcnet/model.py.
Environment reused unchanged: Python3.10, torch2.2.2 CUDA12.1.
Dataset root `/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.
PREFLIGHT and per-seed PROVENANCE retain original config/metric hashes,
effective configs, source hashes, seed, GPU and timestamps. Separate run paths
preserve previous experiments. No baseline rerun and no checkpoint loading.

Real-data seed66 one-epoch smoke completed successfully before launch,
including all8rate evaluations, checkpoint saves, canonical mask comparison.
Smoke outputs remain isolated in sibling `smoke`, not part of reported scores.
Training uses original emotion-only MSE; no extra loss. Compare W-F1 as requested.
Keep eight per-rate best checkpoints perseed, historical Test-oracle INTERNAL
diagnostics only. Launch verification does not establish final performance.
