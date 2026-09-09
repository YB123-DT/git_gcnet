# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 133 | 0.0556697 | 0.0490317 | 0.0209645 | 0.0055549 | 0.0201417 |
| CMUMOSI | 0.1 | [.1,.2) | 421 | 0.157439 | 0.130344 | 0.0678751 | 0.0309312 | 0.020003 |
| CMUMOSI | 0.1 | [.2,.4) | 516 | 0.290276 | 0.222338 | 0.114912 | 0.0728377 | 0.0184388 |
| CMUMOSI | 0.1 | [.4,.6) | 358 | 0.487128 | 0.331005 | 0.20925 | 0.129314 | 0.0189601 |
| CMUMOSI | 0.1 | [.6,1] | 116 | 0.628001 | 0.456893 | 0.244219 | 0.184273 | 0.0179599 |
