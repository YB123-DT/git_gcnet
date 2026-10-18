# B1/B2 frozen diagnostic

INTERNAL DIAGNOSTIC ONLY — Test-oracle seed66.

No training, new seeds or U. Original GCNet masks replayed exactly; source files read-only.
Off is the SAME trained checkpoint without its new residual, not original Flat weights.

## 1. Whole-branch contribution

| Model | condition | ACC | UA | W-F1 | CE |
|---|---|---:|---:|---:|---:|
| B1 | on | 61.795977 | 38.501536 | 58.935928 | 1.194127 |
| B1 | off | 61.728927 | 38.247149 | 58.770872 | 1.192649 |

B1 off→on: corrections55, harms41, changed predictions141; counts sum8rate evaluations, not unique utterances.
Mean norm statistics: `{"anchor_norm": 8.475032448768616, "residual_anchor_ratio": 0.0025527855614200234, "residual_norm": 0.021643830463290215}`.

| Model | condition | ACC | UA | W-F1 | CE |
|---|---|---:|---:|---:|---:|
| B2 | on | 61.757663 | 38.005000 | 58.808538 | 1.192670 |
| B2 | off | 61.757663 | 37.850650 | 58.718606 | 1.191512 |
| B2 | half | 61.762452 | 38.009442 | 58.813009 | 1.192666 |

B2 off→on: corrections44, harms44, changed predictions128; counts sum8rate evaluations, not unique utterances.
Mean norm statistics: `{"anchor_norm": 8.483839809894562, "half_residual_norm": 0.020515949930995703, "residual_anchor_ratio": 0.002404453989583999, "residual_norm": 0.020593287190422416}`.

## 2. Fixed epoch and mask (rate0.1)

Common mask: B1 rate0.1 selected epoch13. B2 epoch13 weights recovered read-only from
its rate0.5 BEST and verified against final full-state trajectory references.
| Readout | epoch | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|---:|
| B1_epoch13 | 13 | 65.555556 | 45.739314 | 64.311837 | 1.102531 |
| B2_epoch10 | 10 | 65.555556 | 42.834889 | 63.776552 | 1.100823 |
| B2_epoch13 | 13 | 65.517241 | 45.697906 | 64.280179 | 1.102078 |

model_at_epoch13: `{"N": 2610, "both_correct": 1710, "both_wrong": 899, "class_disagreement_count": 4, "corrections": 0, "harms": 1, "mean_delta_ce": -0.0004528114464436618, "metric_deltas": {"ACC": -0.03831417624520839, "CE": -0.00045281144644371274, "UA": -0.04140786749483283, "WF1": -0.031657391091826526}, "prediction_changes": 4}`.

B2_epoch_selection_10_to_13: `{"N": 2610, "both_correct": 1669, "both_wrong": 858, "class_disagreement_count": 136, "corrections": 41, "harms": 42, "mean_delta_ce": 0.0012555204224533895, "metric_deltas": {"ACC": -0.03831417624520839, "CE": 0.00125552042245336, "UA": 2.863017042623369, "WF1": 0.5036268271701232}, "prediction_changes": 136}`.

B1_epoch13_to_B2_selected10: `{"N": 2610, "both_correct": 1669, "both_wrong": 857, "class_disagreement_count": 135, "corrections": 42, "harms": 42, "mean_delta_ce": -0.0017083318688970513, "metric_deltas": {"ACC": 0.0, "CE": -0.0017083318688970728, "UA": -2.904424910118202, "WF1": -0.5352842182619497}, "prediction_changes": 135}`.

## 3. Learned S and S=0.5

S metrics use only valid strict-past8 pairs, same-class labels scored OFFLINE.
All relation scores below are fractions0–1. No threshold/calibration fitting.
| rate | pairs | same prevalence | mean S | balanced ACC | AUROC | AUPRC | BCE | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 12142 | 0.359002 | 0.429964 | 0.500000 | 0.665006 | 0.558616 | 0.658312 | 0.232713 | 0.071769 |
| 0.1 | 12142 | 0.359002 | 0.422494 | 0.500000 | 0.628222 | 0.469506 | 0.656183 | 0.231688 | 0.063492 |
| 0.2 | 12142 | 0.359002 | 0.421838 | 0.500000 | 0.596121 | 0.449903 | 0.656417 | 0.231806 | 0.062836 |
| 0.3 | 12142 | 0.359002 | 0.394157 | 0.500000 | 0.608733 | 0.477415 | 0.646861 | 0.227247 | 0.061144 |
| 0.4 | 12142 | 0.359002 | 0.387273 | 0.507258 | 0.612135 | 0.500209 | 0.640932 | 0.224419 | 0.050640 |
| 0.5 | 12142 | 0.359002 | 0.414333 | 0.500000 | 0.552330 | 0.427839 | 0.656049 | 0.231631 | 0.055332 |
| 0.6 | 12142 | 0.359002 | 0.383734 | 0.499936 | 0.583841 | 0.456539 | 0.646367 | 0.227028 | 0.041057 |
| 0.7 | 12142 | 0.359002 | 0.371861 | 0.515962 | 0.597500 | 0.470577 | 0.639489 | 0.223831 | 0.025407 |
| pooled-rate-evaluations | 97136 | 0.359002 | 0.403207 | 0.502895 | 0.584116 | 0.460492 | 0.650076 | 0.228795 | 0.045792 |

Full distributions, fixed-bin calibration, S=.5 baseline and retrieval-weighted
secondary scores are in RESULTS.json; individual pair/prediction NPZ files stay remote.
S=.5 replay sets D=0,rho=.5 for nonempty history ONLY; E,R,Local,Base,Gap and CHM unchanged.
On-vs-half task metrics and corrections/harms are listed above/in per-rate JSON.

## Boundaries

- Whole-branch on/off is not an isolation of E or attentionR: its MLP also sees Local/Base/Gap.
- S=.5 changes an internal training-time distribution: sensitivity analysis, NOT a new model score.
- A learned relation classifier and useful task evidence are different claims; do not equate them.
- Pooled pairs repeat utterance identities across rates and share conversations; no independence/significance claim.
- Test labels enter offline metrics only, never an inference rule or a trained Gate.
- No automatic multi-seed, U, threshold search or new training is authorized by this result.
