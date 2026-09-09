# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 1 | 448 | 0.00064353 | 0.0216373 | 0.181488 | 0.0209938 | 0.159851 | 1 | 1 | 0.973725 | 0.999778 | 0.97837 | 0.997521 | 0.0210717 | 0.10601 |
| CMUMOSI | 0.1 | audio | 2-3 | 64 | 0.278465 | 0.277988 | 0.319863 | -0.000477705 | 0.0418751 | 0.940959 | 0.940959 | 0.930353 | 1.0016 | 0.98015 | 0.998876 | -0.000780039 | 0.0280541 |
| CMUMOSI | 0.1 | text | 1 | 480 | 0.00168015 | 0.0224233 | 0.112359 | 0.0207431 | 0.0899359 | 0.999999 | 0.999999 | 0.9897 | 0.999013 | 0.977622 | 0.964614 | 0.0207695 | 0.0479972 |
| CMUMOSI | 0.1 | text | 2-3 | 72 | 0.100088 | 0.111084 | 0.111491 | 0.0109961 | 0.000406726 | 0.991694 | 0.991694 | 0.993503 | 0.969024 | 0.948272 | 0.954808 | 0.013428 | 0.00220557 |
| CMUMOSI | 0.1 | visual | 1 | 432 | 0.00105283 | 0.0217577 | 0.190274 | 0.0207049 | 0.168517 | 0.999999 | 0.999999 | 0.966124 | 0.999679 | 0.978274 | 1.00893 | 0.0208144 | 0.0765957 |
| CMUMOSI | 0.1 | visual | 2-3 | 48 | 0.113461 | 0.122171 | 0.197347 | 0.0087094 | 0.0751759 | 0.989473 | 0.989473 | 0.97214 | 0.985293 | 0.964193 | 0.970249 | 0.0099237 | 0.0321245 |
