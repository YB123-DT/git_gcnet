# Paired diagnostic launch — 2026-09-30

Code64e473f. biggpu hostGPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
Python PID4166485 (nohup wrapper4166484). Isolated snapshot:
`/data1/yb/remote_experiments/osram_readout_paths_20260930/code`.
Log: `/data1/yb/remote_experiments/osram_readout_paths_20260930/run.log`.
Output: `/data1/yb/remote_experiments/osram_readout_paths_20260930/results`.

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 /data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_readout_paths_20260930/run.py
```

Three seeds sequentially, eight rates, train/validation separately,13 fixed
readout settings. One original Memory forward per batch, no optimization.
Seed66 rate0 train1284 and validation229 utterances verified immediately after
launch with exact identity and unchanged model-state hashes. Per-seed status
and provenance retain progress and effective source/checkpoint/data hashes.

This launch does not include IEMOCAP: its existing official validation aliases
test. No additional training or test-label use was authorized/performed.
