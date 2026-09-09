# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | [0,.1) | 2077 | 0.0507732 | 0.0502553 | 0.011117 | 0.00165407 | 0.0146933 |
| CMUMOSI | 0.7 | [.1,.2) | 2283 | 0.148993 | 0.145169 | 0.0389642 | 0.0161435 | 0.0138123 |
| CMUMOSI | 0.7 | [.2,.4) | 2468 | 0.286543 | 0.272767 | 0.0900545 | 0.0490353 | 0.0123781 |
| CMUMOSI | 0.7 | [.4,.6) | 1124 | 0.490879 | 0.452278 | 0.206558 | 0.149928 | 0.0123302 |
| CMUMOSI | 0.7 | [.6,1] | 664 | 0.641264 | 0.605355 | 0.2763 | 0.267588 | 0.00986901 |
