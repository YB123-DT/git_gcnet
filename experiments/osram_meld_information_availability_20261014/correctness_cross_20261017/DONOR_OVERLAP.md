# Real/Donor overlap on Local-error cohorts

INTERNAL DIAGNOSTIC ONLY. Read saved predictions; no training, model loading,
inference or GPU use. gcnet-legacy-fixed-v1, backbone seed66, three offline
probe initializations66/67/68. This is not the current random-mask baseline.

## Seed66, missing0.7

Local gets1209/2610 samples wrong. All following comparisons use that same
fixed Local-error cohort, not a cohort selected by Real being correct.

| Correctness on Local-error cohort | Count |
|---|---:|
| Real Memory correct |100|
| MemoryDonor correct |103|
| Both correct |47|
| Real correct / Donor wrong |53|
| Donor correct / Real wrong |56|

Within the53 Real-only correct predictions, LocalMemory gets23 correct and30
wrong; the original cached full Flat gets3 correct and50 wrong. This is an
intersection of independently trained predictions, not a deployable selector
or a same-model causal effect of correct history. Original Flat cache logits
were checked finite and7-class; prediction IDs follow the checked cache order.

## Class breakdown within that fixed Local-error cohort

| True class | Local-error N | Real Memory correct | Donor correct |
|---|---:|---:|---:|
| Neutral | 104 | 47 | 102 |
| Surprise | 228 | 6 | 0 |
| Fear | 50 | 0 | 0 |
| Sadness | 187 | 4 | 0 |
| Joy | 302 | 18 | 0 |
| Disgust | 68 | 0 | 0 |
| Anger | 270 | 25 | 1 |

Real gets53 non-Neutral errors correct; Donor gets1. Conversely Donor gets102
Neutral errors correct versus Real's47. Thus overall error-correction counts
hide a pronounced class tradeoff. This does not establish speaker or temporal
history mechanisms. The class labels are offline strata, never model inputs.

## Each rate, mean counts across the three probe initializations

| Rate | Real correct on Local errors | Donor correct on Local errors | Real-only correct | Donor-only correct | ΔACC within Local-error cohort, pp |
|---|---:|---:|---:|---:|---:|
| 0.0 | 171.667 | 224.667 | 82.333 | 135.333 | -5.969 |
| 0.1 | 197.000 | 217.000 | 81.667 | 101.667 | -2.142 |
| 0.2 | 168.000 | 190.000 | 59.000 | 81.000 | -2.333 |
| 0.3 | 181.667 | 180.333 | 76.333 | 75.000 | 0.131 |
| 0.4 | 191.333 | 190.667 | 76.667 | 76.000 | 0.062 |
| 0.5 | 163.000 | 158.000 | 66.333 | 61.333 | 0.445 |
| 0.6 | 153.000 | 120.000 | 90.000 | 57.000 | 2.873 |
| 0.7 | 119.333 | 115.667 | 54.667 | 51.000 | 0.303 |

These are averages over repeated rate/probe evaluations, not counts of distinct
new utterances. Error-subset ACC uses only Local-error denominators and must not
be confused with full-test ACC. Local-error selection changes class balance.

## Implications for the latest A/B/C hypotheses

These letters refer to the user's latest taxonomy, not the older availability
probe report's differently defined cases.

- A: not supported as a global absence of useful real Memory. Full-cohort Real Memory exceeds MemoryDonor by5.280pp mean ACC across rates/probe seeds. But the earlier100 complementary correct samples cannot all be credited to history:47 are also Donor-correct. Aggregate Real/Donor proximity on Local errors is compatible with class-prior effects and is not proof of faulty writing/storage.
- B: there is unrealized complementarity in the53 Real-only samples, including50 that the original Flat misses. However Real/Donor and joint predictions use separately fitted heads, and all probes are test-selected. These overlaps do not identify Query versus Decision as the causal bottleneck or prove a new fusion block can exploit them.
- C: descriptive evidence of class- and missing-rate-dependent utility is present. Real helps more non-Neutral Local errors in the shown cell; on average its Local-error ACC advantage is negative at low missing and positive at0.6. Speaker/history-length hypotheses remain untested in this analysis.

Do not select a structural change solely from these old-protocol intersections.
Current cosine/step0.3 three-seed jobs remain unchanged; no extra runs are
started here. No gold-defined useful-sample switch is implemented.

## Reproduction and checks

The saved donor_overlap.py is the exact read-only Python program executed via
SSH on biggpu with cmerc-repro and CUDA_VISIBLE_DEVICES empty. It imports the
previously tested correctness-cross utility from the recorded remote analysis
directory. It reads all24 rate/seed cells, verifies original per-probe metric
parity, argmax/labels and donor inclusion alignment, and asserts all overlap
partitions. DONOR_OVERLAP.json records input hashes; the eight feature-cache
hashes and finite original Flat logits were checked separately. Donor probe
weights and original cache files were not altered. No causal/significance
claim is made from these descriptive counts.
