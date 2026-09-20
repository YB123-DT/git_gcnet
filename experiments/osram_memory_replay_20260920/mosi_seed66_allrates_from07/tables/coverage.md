# Coverage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | missing_count | no_history_count | no_history_ratio |
| --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 77 | 1 | 0.012987 |
| CMUMOSI | 0.1 | text | 73 | 2 | 0.0273973 |
| CMUMOSI | 0.1 | visual | 65 | 5 | 0.0769231 |
| CMUMOSI | 0.3 | audio | 164 | 11 | 0.0670732 |
| CMUMOSI | 0.3 | text | 211 | 16 | 0.0758294 |
| CMUMOSI | 0.3 | visual | 196 | 11 | 0.0561224 |
| CMUMOSI | 0.5 | audio | 345 | 26 | 0.0753623 |
| CMUMOSI | 0.5 | text | 309 | 15 | 0.0485437 |
| CMUMOSI | 0.5 | visual | 311 | 33 | 0.106109 |
| CMUMOSI | 0.7 | audio | 423 | 61 | 0.144208 |
| CMUMOSI | 0.7 | text | 382 | 51 | 0.133508 |
| CMUMOSI | 0.7 | visual | 408 | 24 | 0.0588235 |
