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
