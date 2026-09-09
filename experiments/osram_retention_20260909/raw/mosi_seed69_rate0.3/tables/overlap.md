# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | [0,.1) | 665 | 0.0557182 | 0.0521349 | 0.0147721 | 0.00299378 | 0.0180879 |
| CMUMOSI | 0.3 | [.1,.2) | 1172 | 0.154748 | 0.136593 | 0.0474027 | 0.019311 | 0.0176995 |
| CMUMOSI | 0.3 | [.2,.4) | 1454 | 0.285845 | 0.238557 | 0.0929752 | 0.0550854 | 0.0163767 |
| CMUMOSI | 0.3 | [.4,.6) | 781 | 0.493336 | 0.371581 | 0.20324 | 0.118162 | 0.0159335 |
| CMUMOSI | 0.3 | [.6,1] | 328 | 0.629852 | 0.508787 | 0.223161 | 0.186663 | 0.0140301 |
