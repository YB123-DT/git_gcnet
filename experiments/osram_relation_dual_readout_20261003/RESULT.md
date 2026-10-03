# Same-trajectory dual-readout supervision

INTERNAL DIAGNOSTIC ONLY

Status: launched, results pending (2026-10-03 07:05 UTC).

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
