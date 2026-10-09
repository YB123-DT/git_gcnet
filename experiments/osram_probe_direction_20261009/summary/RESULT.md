# TEST-ORACLE INTERNAL DIAGNOSTIC

Source status: complete; rows: 25734; missing rate/seed/arm groups: 0.

Frozen oldOSRAM and the existing preceding3_mean ProbeB are evaluated without training or new checkpoint/configuration selection. Inherited test-oracle selection remains a limitation. Probe 66 is primary; 67 and 68 are robustness checks. All are reported, with no best-seed selection. Three probe seeds are not three independently trained backbones.

Full uses actual historical observed-text deletion. Hist projects its real raw-R displacement onto the normalized local probe gradient; other is the complementary displacement. The basis uses raw active-forward R coordinates, so coordinate scaling determines this geometry. This is a rank-one local probe-sensitive direction, not a pure emotion direction. The orthogonal complement may still contain the same decodable information because D is nonlinear; other does not mean non-emotion. Hist/other states can be off-trajectory synthetic states. Nonlinear outputs need not add; nonadditivity is reported explicitly.

Deltas are intervention minus base. Positive delta MSE means worse. Sentiment MSE includes neutral targets; ACC and weighted F1 exclude y=0. Reconstruction failure counts use absolute error >1e-5. Negligible gradients use the producer valid_gradient flag.

Association tables include signed and absolute Pearson/Spearman, and residual correlations adjusting both variables for log1p(displacement norm), lag, and one-hot availability within each rate, probe seed, and arm. Partial Spearman ranks the two outcomes before this same adjustment. These are statistical adjustments, not causal identification. Constant or insufficient-residual-degree groups have null correlations.

Four quadrants use fixed within-rate/seed/arm medians of absolute full deltas. Equality is small. Thresholds and all quadrants are reported descriptively; they are not selection criteria.

Macro statistics weight rates equally within each probe seed before descriptive averaging across probe seeds. Duplicate observations across seeds are never pooled as independent samples. Missing rates and unequal coverage are explicit. Bootstrap intervals use 500 conversation-cluster resamples by default within each rate and arm for primary probe 66 only; one-conversation intervals are marked unidentifiable.

| Primary probe 66, equal-rate macro | ΔMSE full | ΔMSE hist | ΔMSE other |
|---|---:|---:|---:|
| control (8/8 rates) | -0.0018131 | -1.0092e-05 | -0.00181811 |
| delete (8/8 rates) | 0.0361882 | -3.99266e-05 | 0.036087 |

Detailed tables: per_rate.csv, associations.csv, quadrants.csv, macro_rates.csv, probe_seed_descriptions.csv, cluster_bootstrap.csv. SUMMARY.json retains source STATUS and completeness metadata.
