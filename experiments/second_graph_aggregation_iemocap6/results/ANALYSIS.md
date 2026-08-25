# Second-Graph Mechanism Analysis

Chinese version: [ANALYSIS.zh.md](ANALYSIS.zh.md). English mirror: [ANALYSIS.en.md](ANALYSIS.en.md).

## Evidence boundary and setup

All numeric entries below were recomputed from [summary.json](summary.json); no value was inferred from filenames or copied from a best run. The protocol and provenance boundary are recorded in [EXPERIMENT.md](../EXPERIMENT.md), with Chinese and English mirrors at [EXPERIMENT.zh.md](../EXPERIMENT.zh.md) and [EXPERIMENT.en.md](../EXPERIMENT.en.md).

The comparison uses IEMOCAPSix, fold 5, missing rates `0.0` and `0.7`, and formal seeds `66`, `67`, and `68`. Each candidate task is paired with a read-only inherited Original archive by rate, seed, fold, and mask SHA256. Thus, each rate-level summary contains three paired runs, and each candidate has six paired differences. Values are weighted F1 mean ± **sample SD** unless stated otherwise. Original was not retrained.

## Weighted F1 by missing rate

| Arm | Missing 0.0 | Missing 0.7 |
|---|---:|---:|
| Original | 0.627564 ± 0.004537 | 0.616087 ± 0.017411 |
| GenAgg | 0.476453 ± 0.235104 | 0.391534 ± 0.133442 |
| Scaled Soft Medoid | 0.626670 ± 0.017790 | 0.608371 ± 0.024686 |
| SSMA Conv2 | 0.623863 ± 0.021238 | 0.605441 ± 0.017788 |
| RTDR | 0.632265 ± 0.012190 | 0.616468 ± 0.034939 |

The paired candidate-minus-Original differences are:

| Arm | Delta at missing 0.0 | Delta at missing 0.7 | Across-rate macro delta |
|---|---:|---:|---:|
| GenAgg | -0.151111 ± 0.237563 | -0.224553 ± 0.138094 | -0.187832 |
| Scaled Soft Medoid | -0.000893 ± 0.020555 | -0.007715 ± 0.031507 | -0.004304 |
| SSMA Conv2 | -0.003701 ± 0.021211 | -0.010645 ± 0.016442 | -0.007173 |
| RTDR | +0.004701 ± 0.007659 | +0.000381 ± 0.045653 | +0.002541 |

## Seed stability

Each entry below is the mean of that seed's paired deltas at missing rates 0.0 and 0.7.

| Arm | Seed 66 | Seed 67 | Seed 68 | Positive seeds |
|---|---:|---:|---:|---:|
| GenAgg | -0.036149 | -0.377294 | -0.150053 | 0/3 |
| Scaled Soft Medoid | +0.018302 | -0.022715 | -0.008500 | 1/3 |
| SSMA Conv2 | -0.001779 | -0.008314 | -0.011427 | 0/3 |
| RTDR | +0.029813 | -0.001006 | -0.021183 | 1/3 |

RTDR is therefore only slightly positive after averaging all six cells; the sign does not replicate across seeds. Its high-missing rate mean is especially uncertain relative to its magnitude: +0.000381 mean delta with a 0.045653 sample SD.

## Parameters and runtime

The current candidate archives record both `parameter_count` (the instantiated current model) and `selected_path_parameter_count` (the selected experimental path). The historical Original archive stores only the selected-path count with a comparable meaning; its legacy `parameter_count` field must not be interpreted as a current-model total or compared across the Total column. Runtime is summarized across the six tasks for each arm; Original's six unique inherited tasks are counted once.

| Arm | Total parameters | Selected-path parameters | Runtime, seconds |
|---|---:|---:|---:|
| Original | N/A (legacy archive stored selected path only) | 34,140,166 | 362.439 ± 53.831 |
| GenAgg | 36,419,934 | 34,140,284 | 697.742 ± 160.192 |
| Scaled Soft Medoid | 36,419,816 | 34,140,166 | 447.302 ± 114.032 |
| SSMA Conv2 | 37,015,216 | 34,735,566 | 501.393 ± 109.075 |
| RTDR | 36,419,816 | 34,140,166 | 371.226 ± 145.975 |

These wall-clock values describe the completed task records, not isolated operator latency, and concurrent scheduling can contribute to their variance. They should not be interpreted as a controlled speed benchmark.

## Collapse audit

Exactly one candidate task was marked collapsed by the preregistered class-coverage criterion:

| Arm | Missing rate | Seed | Class coverage | Dominant prediction ratio | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| GenAgg | 0.0 | 67 | 4/6 | 0.498460 | 0.205430 |

No Soft Medoid, SSMA, or RTDR task lost class coverage. GenAgg's failure is strongly affected by the collapsed cell (paired delta -0.425189) and by two high-missing cells at seeds 67 and 68 (deltas -0.329398 and -0.276176). It is not explained by the single collapse alone: all six GenAgg paired deltas are negative.

## Exploratory paired tests

For transparency, a two-sided one-sample paired-difference t test and a two-sided Wilcoxon signed-rank test were applied to each candidate's six paired deltas. Bonferroni correction is applied **within each test family**, not across one pooled eight-test family: the four t tests form one family and the four Wilcoxon tests form a separate family. Each family therefore uses a threshold of 0.05 / 4 = 0.0125, and its adjusted p values are `min(4p, 1)`.

| Arm | t(5) | Raw t p | Bonferroni t p | Wilcoxon W | Raw Wilcoxon p | Bonferroni Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| GenAgg | -2.5792 | 0.0495 | 0.1979 | 0 | 0.0313 | 0.1250 |
| Scaled Soft Medoid | -0.4378 | 0.6798 | 1.0000 | 8 | 0.6875 | 1.0000 |
| SSMA Conv2 | -1.0101 | 0.3588 | 1.0000 | 5 | 0.3125 | 1.0000 |
| RTDR | +0.2119 | 0.8405 | 1.0000 | 10 | 1.0000 | 1.0000 |

These tests are **exploratory and low-powered** (`n=6`). More importantly, the six cells are not fully independent: the same three seeds are repeated at two missing rates, and all candidates reuse the same inherited Original cells. Consequently, the table is not a confirmatory inferential analysis. Neither a small raw p value nor a large p value should be read as proof of a mechanism effect or of equivalence. No comparison crosses the Bonferroni-adjusted threshold.

## Mechanism-consistent interpretation

- **GenAgg:** [GenAgg](https://arxiv.org/abs/2306.13826) replaces a fixed sum with a learnable generalized aggregation. In this adaptation, that extra flexibility did not produce stable GCNet behavior: every paired cell declined, one complete-modality run covered only four classes, and the two largest high-missing declines persisted without a class-coverage collapse. The evidence supports stopping this adaptation; it does not identify which internal GenAgg parameter caused the instability.
- **Scaled Soft Medoid:** [robust soft-medoid aggregation](https://arxiv.org/abs/2010.15651) is intended to limit isolated outlier influence. Its mean effect here was small and negative, with mixed per-cell signs and only one positive seed macro. This suggests that robust centering of second-layer messages did not offer a reproducible advantage under this graph/missingness protocol; it does not show that outliers are absent.
- **SSMA Conv2:** [SSMA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html) adds cross-neighbor interaction before compression. All three seed macros were slightly negative, so the richer interaction was not beneficial on this gate. Because the selected path also adds 595,400 parameters, these negative results do not motivate the preregistered parameter-matched follow-up that would only have been required after a positive gate.
- **RTDR:** RTDR is a custom diagonal relation-transition routing hypothesis, not a transfer claim from MrMP; [the inspected MrMP source](https://arxiv.org/abs/2202.04844) mixes relations within a layer. RTDR produced a +0.002541 macro delta and positive rate means, but only seed 66 improved across rates. The conservative conclusion is a small, seed-unstable signal, not a validated improvement.

## Preregistered decision

The locked gate required, for each candidate, positive paired means at both rates, a positive across-rate macro delta, at least two of three positive seed macros, finite outputs, and no collapsed run. GenAgg failed the rate, seed, and collapse conditions; Soft Medoid and SSMA failed the rate and seed conditions; RTDR failed the two-positive-seed condition. Therefore all four candidates stop at this first wave.

The remaining missing rates are not run. Extending the grid after seeing that the preregistered gate failed would weaken the stopping rule, increase post-hoc selection freedom, and spend computation on mechanisms that did not meet the minimum stability requirement. This is a protocol decision, not a claim that the untested rates must be negative.
