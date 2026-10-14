# MELD Local-shortcut-only frozen ablation

**INTERNAL DIAGNOSTIC ONLY.** Evaluation-only intervention on retained per-rate
Test-oracle checkpoints; this is not a retrained no-shortcut model.

Only `local_skip(Local)` immediately before the final LayerNorm is removed. The
Local prefix inside `emotion_adapter([Local,Base,Gap])` remains unchanged;
Memory is scanned once and is unchanged.

## Results

All values are percent. High mean covers missing rates 0.5/0.6/0.7.

| Metric | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | high mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACC original | 65.977 | 63.870 | 62.605 | 59.847 | 58.774 | 57.011 | 55.326 | 53.678 | 59.636 | 55.338 |
| ACC shortcut-off | 15.287 | 15.287 | 12.146 | 11.571 | 12.299 | 13.678 | 15.019 | 13.870 | 13.645 | 14.189 |
| W-F1 original | 63.204 | 61.038 | 58.628 | 55.291 | 53.872 | 50.893 | 48.119 | 45.190 | 54.530 | 48.068 |
| W-F1 shortcut-off | 9.727 | 9.922 | 5.621 | 5.797 | 6.970 | 9.623 | 12.342 | 10.872 | 8.859 | 10.946 |

Eight-rate changes:

- ACC: 59.636 → 13.645 (**−45.991 pp**).
- W-F1: 54.530 → 8.859 (**−45.670 pp**).
- Across the eight repeated test evaluations: 1,687 corrections, 11,290 harms,
  and 18,747 changed predictions.

## Verification and interpretation

- Original replay exactly reproduced the saved source ACC/W-F1 at all eight
  rates; maximum replay-logit errors were between `9.06e-6` and `1.34e-5`.
- Each batch performed one unchanged causal Memory scan. No training, second
  view, altered mask, or Memory intervention was used.
- Source: MELD causal no-JEPA Flat seed66; implementation commit `f952ae7`.
- Full predictions and hashes:
  `/data2/yb/remote_experiments/osram_meld_local_skip_ablation_20261014/seed66_f952ae7`.

The frozen model therefore depends strongly on the final Local shortcut. The
Adapter still receives Local/Base/Gap, but it was trained as a contextual
residual around that anchor and cannot independently sustain the classifier
after the anchor is removed. This does **not** prove that Memory lacks emotion
information (the frozen Probe experiment shows otherwise), nor does it define
the achievable performance of a model retrained without the shortcut.
