# Distance

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | history_query_count | distance_mean | distance_median | distance_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 76 | 1.11842 | 1 | 1.5 |
| CMUMOSI | 0.1 | text | 71 | 1.11268 | 1 | 1 |
| CMUMOSI | 0.1 | visual | 60 | 1.13333 | 1 | 2 |
| CMUMOSI | 0.3 | audio | 153 | 1.27451 | 1 | 2 |
| CMUMOSI | 0.3 | text | 195 | 1.40513 | 1 | 2 |
| CMUMOSI | 0.3 | visual | 185 | 1.28108 | 1 | 2 |
| CMUMOSI | 0.5 | audio | 319 | 1.80251 | 1 | 3 |
| CMUMOSI | 0.5 | text | 294 | 1.63605 | 1 | 3 |
| CMUMOSI | 0.5 | visual | 278 | 1.86691 | 1 | 4 |
| CMUMOSI | 0.7 | audio | 362 | 2.25967 | 2 | 4 |
| CMUMOSI | 0.7 | text | 331 | 2.09366 | 2 | 4 |
| CMUMOSI | 0.7 | visual | 384 | 2.39844 | 2 | 5 |
