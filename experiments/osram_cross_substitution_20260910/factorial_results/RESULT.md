# Saved-output 2×2 factorial decomposition

**INTERNAL DIAGNOSTIC ONLY. No model execution or training.**
MOSI five frozen Flat checkpoints; exactly-one-missing AT/AV/TV and nonzero labels.
Y = sign(label) × raw sentiment regression score. Higher is better. This is a signed decision margin, not calibrated class-logit margin or NLL.

Cells: BB=Base-only; BG=Move B→Gap; GB=Move G→Base; GG=Gap-only. Local is unchanged.
Content effect C=((BB+BG)−(GB+GG))/2; positive favors Base content.
Slot effect S=((BG+GG)−(BB+GB))/2; positive favors Gap slot.
Interaction I=(BG−BB)−(GG−GB).
Exact reconstruction: Y(c,s)=grand+c*C/2+s*S/2+c*s*I/4, with c=+1 for B and s=+1 for Gap slot.
Thus the sample-wise residual of the best additive 2×2 representation is ±I/4.

Means below weight rates .1–.7 equally within each seed and then seeds equally. Rate0 has no eligible samples.

| Effect | Signed mean | Mean absolute | Median absolute (group-averaged) | P90 absolute (group-averaged) |
|---|---:|---:|---:|---:|
| content | +0.037879 | 0.112707 | 0.096202 | 0.236507 |
| slot | +0.005511 | 0.054452 | 0.042061 | 0.121697 |
| interaction | -0.010484 | 0.043189 | 0.028924 | 0.113268 |

| Rate | C mean | S mean | I mean | Mean abs I | Additive sign disagreement |
|---|---:|---:|---:|---:|---:|
| 0.1 | +0.03414 | +0.00896 | -0.01086 | 0.04501 | 0.27% |
| 0.2 | +0.04293 | +0.00517 | -0.01363 | 0.04477 | 0.20% |
| 0.3 | +0.04303 | +0.01096 | -0.01048 | 0.04357 | 0.26% |
| 0.4 | +0.03686 | +0.00362 | -0.01121 | 0.04485 | 0.46% |
| 0.5 | +0.03300 | +0.00598 | -0.00772 | 0.04211 | 0.24% |
| 0.6 | +0.03710 | -0.00173 | -0.01143 | 0.04254 | 0.30% |
| 0.7 | +0.03810 | +0.00561 | -0.00806 | 0.03947 | 0.27% |

| Seed | I mean | Mean abs I |
|---|---:|---:|
| 66 | -0.00858 | 0.02141 |
| 67 | -0.01386 | 0.04614 |
| 68 | -0.01404 | 0.04967 |
| 69 | -0.01829 | 0.05057 |
| 70 | +0.00235 | 0.04815 |

Interaction means positive in 5/35 seed/rate groups.
Additive approximation margin MAE: 0.010797.
Interaction share of within-four-cell factorial energy (group-averaged): 3.64%.
Ignoring interaction changes decision sign in 0.29% of four-cell cases (group-averaged). This is not an F1 delta.

Energy uses E[I²]/E[4C²+4S²+I²]; it is descriptive, not population explained variance.
No equivalence threshold was prespecified. Small signed mean alone cannot establish negligible interaction.
Regression-score scale, frozen checkpoint selection and repeated samples limit inference; no significance/independence claims are made.
See per_sample.csv for full paired effects, per_seed_rate_pattern.csv for AT/AV/TV and all rates, and metadata.json for all input hashes.
