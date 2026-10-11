# Launch and verification

INTERNAL DIAGNOSTIC ONLY. Server biggpu physical GPU7, UUID
GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e. GPU4 forbidden.
Sealed tracked source commit c08682c, root:
`/data1/yb/remote_experiments/osram_nested_direct_capacity_20261011`.
Dispatcher PID1566207, tmux socket nested_direct_capacity_20261011/session capacity.
Two concurrent runs; four total. MLP256 first PID1566300, other admissions
recorded in DISPATCH.json. No existing processes terminated.

Verification: four interface/finite-gradient/no-bypass tests failed RED on absent
variants before implementation. Four passed locally; the additional full config
test initially could not import local torch_geometric (environment missing).
All FIVE tests passed on remote CPU using existing torch_geometric environment.
No GPU smoke. Local config test now skips when dependency absent; no training
source edits after snapshot sealing. Existing residual builder defaults retained.

```bash
cd /data1/yb/remote_experiments/osram_nested_direct_capacity_20261011/source
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_nested_direct_capacity_20261011.dispatch \
--root /data1/yb/remote_experiments/osram_nested_direct_capacity_20261011 \
--reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66/config.json \
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Logs: dispatch.log and runs/{method}/seed_66/train.log. Config/provenance/history,
full recovery and eight per-rate BEST/predictions retained per run. Completion
requires exit0 and outputs_verified, not merely a live process. No final scores
claimed at launch; no automatic multi-seed expansion or follow-up capacity sweep.
