# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | audio | 1 | 1080 | 0.000527433 | 0.0209096 | 0.126832 | 0.0203822 | 0.105922 | 1 | 1 | 0.983149 | 0.999733 | 0.979096 | 0.987728 | 0.0204804 | 0.0478946 |
| CMUMOSI | 0.3 | audio | 2-3 | 432 | 0.146307 | 0.153255 | 0.176106 | 0.00694788 | 0.0228511 | 0.978255 | 0.978255 | 0.973759 | 0.98936 | 0.968938 | 0.966608 | 0.00890261 | 0.00380639 |
| CMUMOSI | 0.3 | audio | 4-7 | 8 | 0.277582 | 0.27848 | 0.172411 | 0.000897422 | -0.106068 | 0.956672 | 0.956672 | 0.983125 | 0.998125 | 0.977526 | 0.918347 | 0.000980414 | -0.123932 |
| CMUMOSI | 0.3 | text | 1 | 1032 | 0.001715 | 0.0218057 | 0.0860753 | 0.0200907 | 0.0642696 | 0.999999 | 0.999999 | 0.994524 | 0.99885 | 0.978231 | 0.953148 | 0.0202434 | 0.0338275 |
| CMUMOSI | 0.3 | text | 2-3 | 392 | 0.0984875 | 0.110364 | 0.124324 | 0.0118766 | 0.0139599 | 0.992232 | 0.992232 | 0.990897 | 0.949753 | 0.930149 | 0.928672 | 0.0131992 | 0.00360208 |
| CMUMOSI | 0.3 | visual | 1 | 1128 | 0.00101985 | 0.0210007 | 0.213787 | 0.0199808 | 0.192786 | 0.999999 | 0.999999 | 0.957779 | 0.999669 | 0.979034 | 1.01215 | 0.0202006 | 0.0836127 |
| CMUMOSI | 0.3 | visual | 2-3 | 424 | 0.204581 | 0.210651 | 0.276928 | 0.00606969 | 0.0662774 | 0.961106 | 0.961106 | 0.943072 | 1.00017 | 0.979521 | 0.990165 | 0.00897381 | 0.00916592 |
| CMUMOSI | 0.3 | visual | 4-7 | 16 | 0.345088 | 0.347316 | 0.397948 | 0.00222817 | 0.0506314 | 0.933896 | 0.933896 | 0.918425 | 1.01075 | 0.989876 | 1.00015 | 0.00637302 | 0.0280312 |
