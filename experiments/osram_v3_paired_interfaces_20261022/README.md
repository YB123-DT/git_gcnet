# MOSI V3 paired interface screen

INTERNAL TEST-ORACLE SCREEN. Six seed66 MOSI runs, grouped into three pre-registered A/B pairs.

| Pair | A | B | Only changed factor |
|---|---|---|---|
| 1 | topology-full | topology-diagonal | Layer 2 source Self-Attention allowed edges |
| 2 | local-adapter | local-shared | Whether Local Skip receives the corrected Local |
| 3 | task-pre-norm | task-post-norm | Whether the relation increment passes through the final original LayerNorm |

All runs retain original CHM, Local/Memory inputs, Flat1600, task head and MSE. The paired variants
initialize every parameter tensor identically, checked by the real-data setup before training.
Diagonal attention retains V/O transformations, residual and original normalization; Q/K gradients
can be structurally zero and are not required to become nonzero as a false success condition.
The complete model is checked against the fresh Flat initial predictions at every test rate.

MOSI protocol: 150 epochs, Adam, batch32, WD1e-5, LR3e-4 for epochs1–100, LR3e-5 for101–150.
GCNet random-mask-v1, cyclic0.0–0.7, exact accepted Flat66 test masks and training mask traces.
No new Flat run, additional seeds, extra losses or parameter search. Eight per-rate Test-best W-F1
checkpoints are retained with the predictions/masks from their selected epochs. The eight-rate mean
can combine different epochs and is not a single deployed checkpoint score.

Accepted reference:
`/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003`.
The old scalar setup validates its stored dataset manifest and file hashes. A source archive must
retain the `dataset` link used by prior immutable source snapshots (for example `source_d00d062/dataset`)
after checking its target and exact manifest; copying only Python source is insufficient.

Run from an immutable source snapshot with the existing s0 environment:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -m experiments.osram_v3_paired_interfaces_20261022.run \
  --output /data2/yb/remote_experiments/osram_v3_paired_interfaces_20261022/execution_v1 \
  --source-commit ACTUAL_COMMIT \
  --gpu-uuids HEALTHY_PHYSICAL_GPU2_UUID HEALTHY_PHYSICAL_GPU3_UUID --per-gpu 3
```

The controller first runs four real optimizer updates for each variant, checking finite gradients,
source-read connectivity, exactly one Memory scan per batch, original parameter updates, initialization,
and full model/optimizer/RNG checkpoint reload. Successful smoke checks are not scores.
Training uses persistent QUEUE.json and per-job logs. GPU UUID binding, physical GPU4 rejection,
RAM/disk checks and measured memory peaks limit admission; six jobs is a total limit, not a promise
that every resource configuration can execute all six simultaneously. Ongoing measured peaks update
the admission budget. Other users' jobs are never signaled.

For an interrupted controller, rerun the identical command with `--resume`; live owned jobs are
reattached using PID/start-time/command identity, incomplete formal jobs require complete `last.pt`.
There is no implicit OOM retry or hyperparameter change. Source snapshots must not change in place.

Outputs: each run's `SUMMARY.json`, `history.json`, exact `fixed_masks.pt`, eight BEST weights and
matching predictions, plus full `last.pt`. Aggregate `REPORT.json`, `PER_RATE.csv`, and `RESULT.md`
contain only completed formal runs. Pair differences are always B minus A, per-rate and mean/high.
Each model is also compared with the accepted Flat66 reference, explicitly separate from the paired
factor comparison. Checkpoint and feature arrays stay remote and are not Git artifacts.
