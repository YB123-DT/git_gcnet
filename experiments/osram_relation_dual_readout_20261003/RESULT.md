# Same-trajectory dual-readout supervision

INTERNAL DIAGNOSTIC ONLY

Status: completed, 100 epochs; provenance completion time 2026-10-03 07:18:38 UTC.

## Verified results

Seed66, W-F1 (%), original per-rate BEST Test-oracle selection:

|Model|8-rate mean|High missing (.5/.6/.7)|
|---|---:|---:|
|Original Flat|81.068|76.352|
|Relation, full-only task loss|80.576|75.685|
|Relation, 0.5 base + 0.5 full|80.649|75.716|
|Dual minus Flat|-0.419|-0.636|
|Dual minus single-loss Relation|+0.073|+0.031|

The dual objective does not recover baseline performance in this seed. Only rate
0.1 improves over Flat (+0.065 points); seven rates decline. This does not establish
a mechanism or multi-seed significance. No further runs are automatically launched.

All eight saved prediction arrays were checked against original labels and
availability, with W-F1 recomputed using label != 0 and prediction > 0. All eight
mask hashes match the prior Relation run. The only changed saved configuration
field versus that run is osram_relation_dual_readout=True. Total parameters remain
13,789,441 (279,648 above Flat); the new training objective adds no parameters.
History contains 100 epochs; remote provenance is complete and eight BEST files
remain on biggpu. Local artifacts exclude model weights.

Per-rate scores, corrections/harms and the unchanged four-cell audit are in
[analysis/COMPARISON.md](analysis/COMPARISON.md). Its `Relation full` column refers
to this dual-supervised model, not the earlier full-only-loss run.

Fixed four-cell changes vs Flat: near/same -2.139, near/opposite +16.529,
far/same -1.659, far/opposite +0.691 percentage points. Overall corrections/harms
are 256/282 across 5,248 rate-exposures, not independent utterances. Improved
near/opposite performance does not offset the aggregate decline. Groups continue
to use original Flat Local-only predictions at threshold 0.25 and unchanged
adjacent polarity labels; neither is supplied to the model.

Reproduce the offline analysis (new output directory required):

```bash
python experiments/osram_current_history_relation_20261003/analyze.py \
  --run experiments/osram_relation_dual_readout_20261003/results/seed_66 \
  --output experiments/osram_relation_dual_readout_20261003/analysis --seed 66
```

## Launch record

- Server: biggpu, host GPU5, UUID GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
- PID: 1718929. Code commit: c624672; implementation: 46c1ceb.
- MOSI seed66, 100 epochs, original cfg84 random cyclic missing rates 0.0–0.7.
- From scratch; no frozen modules or checkpoint warm start.
- Loss: 0.5 task(base) + 0.5 task(full). Base contains Memory.
- One shared Memory trajectory and Flat anchor; shared emotion_norm/task head.
- Evaluation: full readout only, original per-rate BEST Test-oracle protocol.
- No second view, JEPA, completion, contrastive loss, or new parameters.
- Baseline reference: /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66.
- Remote root: /data2/yb/remote_experiments/osram_relation_dual_readout_20261003.
- Isolated source: code/; log: train.log; run outputs: seed_66/.
- launch.json records the command/PID; seed_66/PROVENANCE.json records configuration, hashes and status.

Command (within the isolated remote code directory):

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
PYTHONPATH=/data2/yb/remote_experiments/osram_relation_dual_readout_20261003/code \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_current_history_relation_20261003/run.py \
--output /data2/yb/remote_experiments/osram_relation_dual_readout_20261003/seed_66 \
--gpu 5 --commit c624672 --dual-readout
```

Launch verified by live process, loaded data dimensions and saved config/provenance.
Previous implementation verification passed 18 V100 correctness tests; this is not
evidence of dataset performance. Do not infer completion from this launch record.
