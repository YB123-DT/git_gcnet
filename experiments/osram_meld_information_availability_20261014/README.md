# MELD Information Availability Probe

**INTERNAL DIAGNOSTIC ONLY.** This experiment freezes the retained MELD
seed66 cfg84 no-JEPA causal Flat checkpoints. It fits small offline classifiers
to current Local, learned OSRAM Memory, strict-past Local summaries, and
dimension-matched cross-conversation donor controls. It does not retrain or
modify OSRAM.

The source checkpoint and each probe epoch are Test-oracle selected. Results
therefore diagnose the retained representations and are not deployable or
independent paper performance.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" python -m \
  experiments.osram_meld_information_availability_20261014.run \
  --source /path/to/flat_seed66 \
  --dataset-root /path/to/dataset \
  --meld-pickle /path/to/meld.pkl \
  --output /path/to/new/output \
  --gpu-uuid "$GPU_UUID"
```

Use `--check-only --rates 0.0` for the mandatory real-checkpoint smoke.
