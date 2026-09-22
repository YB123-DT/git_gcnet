# Frozen joint-pretraining reinjection (MOSI)

This is the L2 diagnostic requested after the three-dataset utterance-level
M3 pretraining. Each seed loads its matching joint checkpoint and freezes:

- `ObservedSetEncoder.projectors.*` (`P_A/P_T/P_V`);
- `SourceOnlyM3Predictor` (`MMoE`);
- the unused EMA teacher.

The predicted missing slots pass through `CompletedReadFusion` and the single
causal OSRAM read/write path. The observed-set fusion, CompletedReadFusion,
OSRAM and emotion classifier remain trainable. No JEPA loss is added in this
L2 run; this isolates whether reinjected predictions help the task path.

The run uses the existing cfg84 causal OSRAM family (1600 output, 8 heads,
64-d keys/values, write step 0.6), cyclic training masks, and independent
per-rate Test-oracle checkpoint selection. It is an internal diagnostic, not a
formal paper result.

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  experiments/osram_joint_frozen_reinjection_20260922/run.py --launch
```
