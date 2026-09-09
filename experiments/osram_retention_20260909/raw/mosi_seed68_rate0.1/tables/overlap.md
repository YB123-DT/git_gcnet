# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 195 | 0.0601201 | 0.0488088 | 0.0214882 | 0.00550253 | 0.0195943 |
| CMUMOSI | 0.1 | [.1,.2) | 378 | 0.136196 | 0.107066 | 0.0363114 | 0.0178638 | 0.0196342 |
| CMUMOSI | 0.1 | [.2,.4) | 457 | 0.326012 | 0.255158 | 0.112042 | 0.0691651 | 0.0183094 |
| CMUMOSI | 0.1 | [.4,.6) | 460 | 0.458666 | 0.345748 | 0.184461 | 0.102902 | 0.0184689 |
| CMUMOSI | 0.1 | [.6,1] | 70 | 0.623304 | 0.454285 | 0.351538 | 0.15662 | 0.0187932 |
