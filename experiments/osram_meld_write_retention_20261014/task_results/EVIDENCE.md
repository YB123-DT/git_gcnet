# Source Text association recovery does not imply task improvement

INTERNAL DIAGNOSTIC ONLY. Original large Flat seed66, frozen weights and
target Local/queries/observed keys, original saved GCNet masks, eight rates.
No model training or new mask trajectory was used in this diagnostic.

Below, each row pools the existing eight rate-conditioned **source-Text / target
pairs** at that fixed distance. A target can appear at different rates; N is a
count of paired diagnostic observations, not independent unique utterances.
Means are N-weighted over these observations, not eight-rate mean model scores.
All four branches use identical eligible pairs within each rate/distance.

Δassociation error = branch error − Normal error; negative means better source
Key→Value recovery. ΔCE = branch CE − Normal CE; positive means task harm.

| Distance | Branch | N | Δassociation error | ΔCE | Corrections | Harms |
|---:|---|---:|---:|---:|---:|---:|
| 2 | decay-only | 10776 | -0.156142 | +0.004665 | 64 | 82 |
| 2 | write-only | 10776 | -0.013442 | -0.000219 | 10 | 4 |
| 2 | hold | 10776 | -0.179765 | +0.004557 | 69 | 86 |
| 4 | decay-only | 8226 | -0.316937 | +0.009281 | 76 | 100 |
| 4 | write-only | 8226 | -0.012426 | -0.000532 | 11 | 3 |
| 4 | hold | 8226 | -0.364983 | +0.008906 | 79 | 100 |
| 8 | decay-only | 4395 | -0.380847 | +0.018690 | 42 | 58 |
| 8 | write-only | 4395 | -0.012961 | -0.000100 | 8 | 4 |
| 8 | hold | 4395 | -0.477449 | +0.019972 | 47 | 65 |

This is also present when current Text is absent (A/V/AV combined, no label-based
selection):

| Distance | N | Decay-only ΔCE | Corrections / harms | Hold ΔCE | Corrections / harms |
|---:|---:|---:|---:|---:|---:|
| 2 | 2912 | +0.008908 | 27 / 42 | +0.009506 | 32 / 44 |
| 4 | 2209 | +0.011599 | 40 / 50 | +0.012786 | 41 / 54 |
| 8 | 1121 | +0.028882 | 18 / 32 | +0.036331 | 20 / 37 |

Write-only on these no-current-Text cohorts gives small positive ΔCE
(+0.000306/+0.000312/+0.002677) despite more corrections than harms
(7/3, 7/2, 6/1). CE and discrete classification need not change together;
neither metric should be silently substituted for the other.

## What this supports

Suppressing intervening writes (decay-only/hold) improves source Text
association recovery, yet worsens mean task CE and causes more classification
harms than corrections on these paired cohorts. Thus stronger preservation
of old associations alone is not a justified optimization objective here.

The interventions affect **all intervening joint writes**, not isolated Text
writes, and can remove useful newer information. The result does not show
that historical Text is intrinsically harmful, that forgetting is globally
optimal, or that the original Query/Flat is optimal. Write-only gives small
paired improvements here, not evidence of a trained-method performance gain.

Full per-rate, fixed-distance, current-availability tables, ACC/UA/W-F1 deltas,
class recalls/supports, source-association-degraded strata and paired direction
counts are retained in the CSV and compressed JSON. No cross-distance whole
test score or model-selection claim is made. Query retrieval and classifier
utilization remain unseparated by this fixed-Query state intervention.
