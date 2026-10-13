# MELD Flat frozen history-scale diagnostic

**INTERNAL DIAGNOSTIC ONLY.** This is a post-hoc sweep over test predictions,
not a trained or deployable model comparison. The source checkpoints use the
existing per-rate Test-oracle selection protocol.

## Intervention

For one unchanged causal scan and one fixed checkpoint:

\[
h_\alpha=\operatorname{LN}\left[
W_LL+F(L,\alpha B,\alpha G_A,\alpha G_T,\alpha G_V)
\right],
\qquad \alpha\in\{0,0.5,1,1.5,2\}.
\]

Local, masks, Memory trajectory, Flat weights and classifier remain fixed.
Gap inputs retain their original availability mask. Seeds 66/67/68 and missing
rates 0.0--0.7 use the same `gcnet-legacy-fixed-v1` masks and checkpoints as
the preceding source-combination audit.

## Three-seed aggregate

All values are percentages. `High` averages rates 0.5/0.6/0.7.

| History scale | ACC 8-rate | ACC High | UA 8-rate | UA High | W-F1 8-rate | W-F1 High | W-F1 delta vs 1 | High delta vs 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 59.294 | 55.045 | 31.021 | 24.947 | 54.046 | 47.626 | -0.362 | -0.297 |
| 0.5 | 59.459 | 55.160 | 31.302 | 25.107 | 54.313 | 47.817 | -0.095 | -0.106 |
| **1.0 (original)** | 59.502 | 55.198 | 31.398 | 25.162 | 54.408 | 47.923 | 0 | 0 |
| 1.5 | **59.524** | 55.262 | **31.501** | 25.313 | **54.481** | 48.060 | **+0.073** | +0.137 |
| 2.0 | 59.465 | **55.317** | 31.486 | **25.419** | 54.447 | **48.165** | +0.039 | **+0.242** |

## Per-rate W-F1

| Scale | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 62.486 | 60.331 | 58.064 | 55.913 | 52.698 | 50.250 | 47.627 | 45.001 |
| 0.5 | 62.965 | 60.829 | 58.331 | 56.013 | 52.913 | 50.449 | 47.983 | 45.019 |
| **1.0** | 63.087 | 60.897 | 58.335 | 56.111 | **53.061** | 50.686 | 48.038 | 45.045 |
| 1.5 | **63.186** | **61.003** | **58.370** | **56.133** | 52.972 | **50.840** | 48.245 | 45.095 |
| 2.0 | 63.065 | 60.911 | 58.255 | 55.996 | 52.855 | 50.811 | **48.282** | **45.401** |

## Seed-level W-F1

| Seed | Scale 0 | Scale .5 | Scale 1 | Scale 1.5 | Scale 2 |
|---:|---:|---:|---:|---:|---:|
| 66, 8-rate | 54.456 | 54.509 | 54.530 | 54.665 | **54.705** |
| 67, 8-rate | 53.869 | 54.327 | **54.433** | 54.379 | 54.331 |
| 68, 8-rate | 53.814 | 54.102 | 54.260 | **54.398** | 54.304 |
| 66, High | 48.011 | 48.031 | 48.068 | 48.245 | **48.355** |
| 67, High | 46.652 | 47.028 | 47.140 | 47.032 | **47.206** |
| 68, High | 48.215 | 48.392 | 48.562 | 48.903 | **48.933** |

## Checks and conclusion

- Scale 1 reproduced the original classes for all 24 seed/rate cells. Maximum
  absolute replay logit error was `1.765e-5`.
- Scaling history upward produces a small response: scale 1.5 is best on the
  eight-rate W-F1 mean (`+0.073` pp), while scale 2 is best on High W-F1
  (`+0.242` pp).
- The gain is not seed-consistent for the eight-rate mean: seed 67 is best at
  the original scale, while seeds 66 and 68 prefer stronger history.
- Therefore simple global under-scaling is not the main explanation for the
  weak MELD Memory contribution. The sweep does not justify selecting 1.5 or
  2 on test data, nor does it show that increasing Flat parameter count would
  help.

Raw per-seed summaries are stored under `results/`.
