# MOSEI66 Direct: Local / Base / Gap readout audit

INTERNAL DIAGNOSTIC ONLY. No training, new masks or checkpoint selection.

Reuses the original completed Source-Role Direct seed66 MOSEI run (60 epochs,
constant LR1e-3), its eight W-F1-selected BEST checkpoints and exact saved test
availability. This is not the separately trained Flat baseline or the newer
constant1e-4 Flat67/68 runs. Original source SHA is checked from its SUMMARY.

Four classifier-input conditions, sharing one causal scan per batch:

- L: original Local input and original Local Skip; Base/Gap zero.
- LB: original Local and calibrated Base; Gap zero.
- LG: original Local and calibrated active Gap; Base zero.
- Full: unchanged original Direct model prediction.

The learned 2*sigmoid source coefficients remain active. No replacement with
unweighted source sums, no CHM write/query intervention and no Local Skip ablation.
All parameters frozen; exact Full reconstruction, saved predictions, sample IDs,
labels, mask, model tensors and RNG are checked. Padding/inactive Gap/backward
half checks reject unexpected evidence. Zeroing final readouts can be OOD; this
is not a training ablation or a causal localization of writing versus reading.

Original s0 interpreter/Torch/NumPy/sklearn; healthy GPU7 by UUID, neverGPU4.
Bounded eight-rate inference only; no optimizer. Large NPZ stays remote.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -m pytest -q \
 experiments/osram_mosei_direct_role_readout_20261021/test_readout.py \
 tests/test_mosei_source_role_switch.py tests/test_source_role_neutral_readout.py \
 tests/test_source_role_neutral_analysis.py

/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosei_direct_role_readout_20261021.run \
 --output /data2/yb/remote_experiments/osram_mosei_direct_role_readout_20261021/execution_v1 \
 --source-commit SOURCE_SHA \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Report per-rate ACC/W-F1/MSE, equal eight-rate and high(.5/.6/.7) means, paired
L→LB / L→LG / LB→Full / LG→Full / L→Full corrections and harms.
Polarity excludes true label0 and uses prediction>0. Counts summed across rates
are repeated test exposures, not independent unique utterances. No multi-seed or
independent-test significance claim. Small reports/CSV/provenance go to Git.

Implementation reuses the existing protected-anchor/environment/mask loader via
optional collector/analyzer/reporter hooks; its original default path is unchanged.
Targeted tests:3new + original switch/neutralization regression suite. Red stage
failed because the new module was absent; green stage executed remotely in s0.
