# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | audio | 1 | 1008 | 0.000608095 | 0.021385 | 0.132549 | 0.0207769 | 0.111164 | 1 | 1 | 0.982098 | 0.999749 | 0.978625 | 0.993556 | 0.0208572 | 0.0529881 |
| CMUMOSI | 0.3 | audio | 2-3 | 488 | 0.136368 | 0.144143 | 0.1795 | 0.00777493 | 0.0353571 | 0.981312 | 0.981312 | 0.973472 | 0.993001 | 0.97202 | 0.978656 | 0.00929902 | 0.00969676 |
| CMUMOSI | 0.3 | audio | 4-7 | 8 | 0.239355 | 0.245959 | 0.195751 | 0.00660357 | -0.0502077 | 0.957824 | 0.957824 | 0.972467 | 0.957366 | 0.937131 | 0.935662 | 0.00920573 | -0.0423946 |
| CMUMOSI | 0.3 | text | 1 | 1184 | 0.00183416 | 0.0223924 | 0.0669183 | 0.0205583 | 0.0445259 | 0.999999 | 0.999999 | 0.996622 | 0.998747 | 0.977644 | 0.964287 | 0.0206092 | 0.019332 |
| CMUMOSI | 0.3 | text | 2-3 | 328 | 0.0793765 | 0.0938124 | 0.107045 | 0.0144359 | 0.0132326 | 0.994983 | 0.994983 | 0.993617 | 0.953891 | 0.933734 | 0.930157 | 0.0164531 | 0.00180305 |
| CMUMOSI | 0.3 | text | 4-7 | 48 | 0.133645 | 0.150017 | 0.152218 | 0.0163719 | 0.00220095 | 0.994479 | 0.994479 | 0.993945 | 0.88777 | 0.869008 | 0.867673 | 0.0179081 | 0.000634685 |
| CMUMOSI | 0.3 | visual | 1 | 1152 | 0.000974503 | 0.0214701 | 0.181234 | 0.0204956 | 0.159764 | 0.999999 | 0.999999 | 0.968152 | 0.999677 | 0.978555 | 1.01845 | 0.0206692 | 0.0649446 |
| CMUMOSI | 0.3 | visual | 2-3 | 416 | 0.188421 | 0.192775 | 0.215452 | 0.00435332 | 0.0226769 | 0.967966 | 0.967966 | 0.961537 | 1.01982 | 0.998271 | 1.00033 | 0.00580074 | 0.00484106 |
| CMUMOSI | 0.3 | visual | 4-7 | 8 | 0.11294 | 0.123571 | 0.136004 | 0.0106311 | 0.0124332 | 0.994336 | 0.994336 | 0.991438 | 0.945854 | 0.925866 | 0.933427 | 0.010716 | 0.000812013 |
