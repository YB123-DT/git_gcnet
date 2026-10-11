# Extra capacity launch

INTERNAL DIAGNOSTIC ONLY. Shared immutable tracked source commit7939317.
Remote root `/data1/yb/remote_experiments/osram_nested_direct_capacity_more_20261011`.
Healthy biggpu GPU0/1, two runs per card; GPU4 forbidden. Both cards initially
32495MiB free and0%utilization; no unrelated processes killed or protocol changes.
tmux socket nested_capacity_more_20261011, sessions gpu0/gpu1.
Lane0 dispatcher1591312, first MLP128 process1591660.
Lane1 dispatcher1591494, first dim128/MLP512 process1591675.
Remaining second process in each lane admitted automatically after resource
check; exact per-run PIDs/commands in lane{0,1}/DISPATCH.json.

Effective initial RAW_CONFIG: correct direct method, output1600, adapter width0
(legacy1600), seed66,100epochs. Remote CPU9 tests passed (eight interface/gradient
cases and all-eight config protocol check). Local10passed/1skipped including
root Probe regression; skip is absent local torch_geometric config import.

```bash
cd /data1/yb/remote_experiments/osram_nested_direct_capacity_more_20261011/source
# Launch once per lane, 0 and 1, separate --root lane0/lane1:
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_nested_direct_capacity_more_20261011.dispatch --lane 0 \
--root /data1/yb/remote_experiments/osram_nested_direct_capacity_more_20261011/lane0 \
--reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66/config.json \
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Lane logs lane0.log/lane1.log. Training logs lane{lane}/runs/{method}/seed_66/train.log.
No final results at launch; exit0+100epochs+outputs_verified required. Preserve
all outcomes, no automatic extra seeds or third capacity batch.
