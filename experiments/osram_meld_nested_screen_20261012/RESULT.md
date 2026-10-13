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

The per-seed breakdown of the best mean-ACC configuration (`d32-l4-g2-m64`)
is:

| Seed | 8-rate ACC | High ACC | 8-rate W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| 66 | 61.255 | 58.072 | 58.086 | 53.707 |
| 67 | 61.207 | 57.497 | 57.729 | 52.502 |
| 68 | 60.996 | 57.739 | 57.053 | 52.524 |

Relative to the matched Flat seeds, its mean-ACC changes are +0.340, +0.264,
and -0.134 percentage points. The improvement is therefore not seed-consistent.

## One-seed compact local refinement

All 14 configurations completed. The leading new configuration was
`d32-l3-g2-m64` with 61.140 mean ACC, 57.586 high-missing ACC, and 58.185
mean W-F1. It did not exceed the broad-screen best mean ACC of 61.255.

Across the broad 64-run seed-66 screen, larger capacity was not beneficial:
parameter count had a descriptive correlation of -0.286 with mean ACC. The
best factor-level means favored dimension 64, depth 4, group count 4, and MLP
width 64, but every factor-level difference was small and the winning
configurations were heterogeneous.
