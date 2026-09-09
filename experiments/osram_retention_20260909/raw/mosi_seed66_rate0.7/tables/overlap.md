# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | [0,.1) | 2941 | 0.0517127 | 0.0508297 | 0.00753926 | 0.00165182 | 0.0156399 |
| CMUMOSI | 0.7 | [.1,.2) | 1537 | 0.143088 | 0.13919 | 0.0377285 | 0.0157336 | 0.0143725 |
| CMUMOSI | 0.7 | [.2,.4) | 3228 | 0.291195 | 0.278697 | 0.0930877 | 0.0510683 | 0.0121818 |
| CMUMOSI | 0.7 | [.4,.6) | 883 | 0.469465 | 0.446336 | 0.206644 | 0.125098 | 0.0105414 |
| CMUMOSI | 0.7 | [.6,1] | 27 | 0.622271 | 0.561509 | 0.351533 | 0.211933 | 0.00833348 |
