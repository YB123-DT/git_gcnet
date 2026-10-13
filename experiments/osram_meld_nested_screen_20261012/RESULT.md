# MELD Nested capacity screens

**INTERNAL DIAGNOSTIC ONLY.** All values use the repository's per-rate
Test-oracle protocol. For MELD, each missing rate selects its epoch by test
accuracy; W-F1 is taken from that same epoch. Checkpoints were discarded after
metrics and predictions were saved.

## Three-seed anchors (50 epochs)

| Model | 8-rate ACC | High-missing ACC | 8-rate W-F1 | High-missing W-F1 |
|---|---:|---:|---:|---:|
| Flat | 60.996 | 57.561 | 57.486 | 52.764 |
| Default Nested | 60.889 | 57.590 | 57.625 | 52.791 |

## Top-seven seed-66 candidates completed over three seeds

| Configuration | 8-rate ACC | High ACC | 8-rate W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| d32-l4-g2-m64 | **61.153** | 57.769 | 57.623 | 52.911 |
| d64-l4-g4-m256 | 61.137 | **57.871** | 57.691 | **53.070** |
| d256-l3-g1-m128 | 61.135 | 57.710 | 57.532 | 52.634 |
| d32-l4-g4-m128 | 61.106 | 57.616 | 57.524 | 52.736 |
| d32-l1-g8-m512 | 61.082 | 57.595 | **57.696** | 52.670 |
| d128-l4-g8-m64 | 61.057 | 57.569 | 57.517 | 52.383 |
| d32-l3-g8-m128 | 61.028 | 57.642 | 57.667 | 52.819 |

The best mean-ACC configuration improves over the three-seed Flat anchor by
0.157 percentage points. The d64-l4-g4-m256 configuration has the strongest
high-missing and W-F1 aggregate, but its improvements remain small. These
screens do not establish statistical significance.

For `d32-l4-g2-m64`, recomputed three-seed UA is 36.445 over all rates and
31.495 over rates 0.5--0.7. The matched 50-epoch causal ComP screen obtains
33.742 and 27.768, respectively, so the Nested configuration is higher by
2.702 and 3.727 percentage points. The per-seed mean-UA advantages are +4.330,
+3.053, and +0.724 points, so the direction is consistent but its magnitude is
seed-dependent.

The per-seed breakdown of the best mean-ACC configuration (`d32-l4-g2-m64`)
is:

| Seed | 8-rate ACC | High ACC | 8-rate W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| 66 | 61.255 | 58.072 | 58.086 | 53.707 |
| 67 | 61.207 | 57.497 | 57.729 | 52.502 |
| 68 | 60.996 | 57.739 | 57.053 | 52.524 |

Relative to the matched Flat seeds, its mean-ACC changes are +0.340, +0.264,
and -0.134 percentage points. The improvement is therefore not seed-consistent.

Across all seven follow-up configurations, `d256-l3-g1-m128` was the only one
whose mean ACC exceeded its matched Flat result for every seed (+0.249, +0.153,
and +0.014 points). `d64-l4-g4-m256` was the only configuration with a clear
high-missing ACC improvement for every seed (+0.230, +0.370, and +0.332
points), although its seed-68 overall mean was 0.110 points below Flat. Thus
the strongest overall-ACC and high-missing patterns occur in different
configurations, and neither constitutes a large uniform gain.

## One-seed compact local refinement

All 14 configurations completed. The leading new configuration was
`d32-l3-g2-m64` with 61.140 mean ACC, 57.586 high-missing ACC, and 58.185
mean W-F1. It did not exceed the broad-screen best mean ACC of 61.255.

| Local configuration (seed 66) | Mean ACC | High ACC | Mean W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| d32-l3-g2-m64 | **61.140** | 57.586 | **58.185** | 53.484 |
| d32-l4-g4-m64 | 60.987 | **57.918** | 57.429 | 52.861 |
| d48-l3-g4-m64 | 60.987 | 57.701 | 58.025 | 53.236 |
| d32-l4-g2-m128 | 60.958 | 57.458 | 57.698 | 52.185 |
| d64-l3-g2-m128 | 60.915 | 57.573 | 57.740 | 52.591 |
| d64-l4-g4-m64 | 60.905 | 57.676 | 57.903 | 53.149 |
| d64-l3-g4-m64 | 60.900 | 57.752 | 57.230 | 53.078 |
| d32-l3-g4-m128 | 60.867 | 57.599 | 57.348 | 51.622 |
| d32-l3-g2-m128 | 60.852 | 57.637 | 56.911 | 51.273 |
| d64-l4-g2-m64 | 60.790 | 57.484 | 58.104 | **53.581** |
| d48-l4-g2-m64 | 60.738 | 57.548 | 57.563 | 52.773 |
| d48-l3-g2-m64 | 60.728 | 57.625 | 57.603 | 53.105 |
| d64-l4-g4-m128 | 60.723 | 57.433 | 57.330 | 51.875 |
| d48-l4-g4-m64 | 60.656 | 57.382 | 57.323 | 52.918 |

UA was recomputed as seven-class macro recall from the saved per-utterance
predictions because it is not stored directly in `metrics.json`. The seed-66
Flat anchor obtains 35.327 mean UA and 30.249 high-missing UA; default Nested
obtains 38.226 and 32.746. Within the local14 screen, `d64-l4-g4-m64` has the
best mean UA (38.050), while `d64-l4-g2-m64` has the best high-missing UA
(33.668). The local mean-ACC winner `d32-l3-g2-m64` reaches 37.827 mean UA and
32.466 high-missing UA. These one-seed results suggest a class-balance effect,
but they do not establish cross-seed stability.

Across the broad 64-run seed-66 screen, larger capacity was not beneficial:
parameter count had a descriptive correlation of -0.286 with mean ACC. The
best factor-level means favored dimension 64, depth 4, group count 4, and MLP
width 64, but every factor-level difference was small and the winning
configurations were heterogeneous.
