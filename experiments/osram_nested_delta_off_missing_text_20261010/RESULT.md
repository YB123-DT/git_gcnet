# Frozen Nested residual ON/OFF, seed66, all eight missing rates

INTERNAL DIAGNOSTIC ONLY. Completed on biggpu GPU7. Evaluation-only intervention,
no training. Source original100 Nested per-rate Test-oracle BEST checkpoints:
`/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/nested_seed66`.
Executed code commit `1b71fca`; original baseline lineage/config in source PROVENANCE.json.

OFF bypasses Nested adapter-input Local/Base/Gap residuals, **not OSRAM delta write**.
Memory scan/query/write, original Local Skip, Flat/norm/head, weights and test masks
remain unchanged. ON/OFF reuse one scan per batch. OFF is not Local-only, and not
a separately trained Flat model. This isolates direct branch effect; it does not
measure how training Nested changed the backbone.

## Main all-test result

| Cohort | OFF W-F1 (%) | ON W-F1 (%) | ON−OFF (pp) | Corrections / harms |
|---|---:|---:|---:|---:|
| Eight-rate mean | 80.919 | 80.992 | +0.073 | 30 / 24 |
| High missing .5/.6/.7 | 75.950 | 76.077 | +0.127 | 19 / 15 |
| Current no-Text, nonempty-rate mean | 68.691 | 68.394 | −0.298 | 20 / 19 |
| Current no-Text, high missing | 65.790 | 65.662 | −0.128 | 15 / 14 |

## Current Text-missing samples, per rate

| Miss | Nonneutral N | OFF W-F1 | ON W-F1 | ON−OFF pp | Corrections / harms |
|---|---:|---:|---:|---:|---:|
| .0 | 0 | N/A | N/A | N/A | 0 / 0 |
| .1 | 69 | 72.878 | 71.192 | −1.685 | 0 / 1 |
| .2 | 144 | 70.920 | 70.460 | −0.460 | 2 / 2 |
| .3 | 202 | 66.967 | 67.443 | +0.476 | 2 / 1 |
| .4 | 252 | 72.705 | 72.674 | −0.032 | 1 / 1 |
| .5 | 301 | 67.794 | 66.702 | −1.092 | 4 / 6 |
| .6 | 348 | 62.337 | 63.260 | +0.923 | 7 / 3 |
| .7 | 368 | 67.240 | 67.025 | −0.215 | 4 / 5 |

## No-Text pattern split: mean over nonempty rates

| Pattern | OFF W-F1 | ON W-F1 | ON−OFF pp | Corrections / harms |
|---|---:|---:|---:|---:|
| A | 73.299 | 73.406 | +0.106 | 8 / 7 |
| V | 64.814 | 63.111 | −1.703 | 5 / 5 |
| AV | 66.799 | 66.709 | −0.091 | 7 / 7 |

These are current random-missing utterance groups, **not whole-conversation fixed
missing** results. Low-rate pattern cohorts can be small. Rates are macro-averaged;
counts sum repeated rate/utterance pairs, not independent unique samples.
Weighted F1 depends on which class flips, so corrections minus harms alone does
not determine its change. Nonzero labels only; binary prediction threshold >0.
No cohort or label is used to alter inference.

## Verification and artifacts

- All eight strict checkpoint loads and frozen state/file hashes passed.
- Full ON matches original archived labels/masks exactly and predictions within
  4.77e-7; original per-rate W-F1 reproduced.
- ON readout replay matches normal forward within 4.77e-7.
- CPU readout regression test: 1 passed; Python compilation passed. Inactive
  NaN Gap inputs safely masked and padding predictions strictly zero.
- Attempt1 stopped on archive ordering mismatch. Attempt2 corrected only collection
  order (conversation-major); failure logs retained remotely. No weights changed.
- `results/utterances.csv`: every valid sample's ID/rate/label/pattern/ON/OFF.
- `results/per_rate.csv`, `results/macro.csv`, `results/SUMMARY.json`,
  `results/STATUS.json`: full scores, paired counts, hashes, epochs, completed status.

Conclusion limited to seed66: direct residual effect is small overall and does not
consistently improve current Text-missing samples. No mechanism or multi-seed claim.
