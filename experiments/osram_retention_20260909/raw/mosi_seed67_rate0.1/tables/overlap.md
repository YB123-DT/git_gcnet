# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 261 | 0.0531606 | 0.0447371 | 0.0215341 | 0.00501119 | 0.0196145 |
| CMUMOSI | 0.1 | [.1,.2) | 461 | 0.153749 | 0.121042 | 0.0467381 | 0.0228501 | 0.0197277 |
| CMUMOSI | 0.1 | [.2,.4) | 406 | 0.29427 | 0.218096 | 0.114976 | 0.0511684 | 0.019551 |
| CMUMOSI | 0.1 | [.4,.6) | 479 | 0.470434 | 0.360338 | 0.185804 | 0.0981696 | 0.0185533 |
| CMUMOSI | 0.1 | [.6,1] | 17 | 0.609094 | 0.574745 | 0.184345 | 0.107752 | 0.0179124 |
