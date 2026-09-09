# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 1 | 536 | 0.000430314 | 0.0207722 | 0.12998 | 0.0203419 | 0.109208 | 1 | 1 | 0.984101 | 0.999874 | 0.979234 | 0.992923 | 0.0204714 | 0.0529769 |
| CMUMOSI | 0.1 | audio | 2-3 | 80 | 0.143982 | 0.149576 | 0.168243 | 0.00559407 | 0.0186673 | 0.980644 | 0.980644 | 0.975359 | 0.992413 | 0.971928 | 0.954365 | 0.0060403 | 0.0109602 |
| CMUMOSI | 0.1 | text | 1 | 440 | 0.0019182 | 0.0219386 | 0.0817585 | 0.0200204 | 0.0598199 | 0.999999 | 0.999999 | 0.99578 | 0.998728 | 0.978111 | 0.963273 | 0.0202552 | 0.0427377 |
| CMUMOSI | 0.1 | text | 2-3 | 40 | 0.0939568 | 0.105122 | 0.130698 | 0.0111648 | 0.0255766 | 0.988583 | 0.988583 | 0.983956 | 0.964867 | 0.944952 | 0.939657 | 0.013861 | 0.00646283 |
| CMUMOSI | 0.1 | visual | 1 | 416 | 0.000893933 | 0.0209004 | 0.23324 | 0.0200064 | 0.212339 | 0.999999 | 0.999999 | 0.955691 | 0.999768 | 0.97913 | 1.01676 | 0.0202316 | 0.103348 |
| CMUMOSI | 0.1 | visual | 2-3 | 48 | 0.127102 | 0.137357 | 0.197208 | 0.0102548 | 0.0598514 | 0.984538 | 0.984538 | 0.973079 | 0.975594 | 0.955455 | 1.00442 | 0.0129961 | -0.000517238 |
