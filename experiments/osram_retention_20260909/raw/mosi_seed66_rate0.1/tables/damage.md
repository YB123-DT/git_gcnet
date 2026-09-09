# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 1 | 544 | 0.000593519 | 0.021502 | 0.151582 | 0.0209085 | 0.13008 | 1 | 1 | 0.975244 | 0.999764 | 0.978504 | 0.978778 | 0.0210219 | 0.0677025 |
| CMUMOSI | 0.1 | audio | 2-3 | 64 | 0.218895 | 0.22325 | 0.255063 | 0.00435483 | 0.031813 | 0.938552 | 0.938552 | 0.926395 | 0.984006 | 0.963087 | 0.952709 | 0.00459095 | 0.0124796 |
| CMUMOSI | 0.1 | text | 1 | 512 | 0.00183491 | 0.0223915 | 0.0884615 | 0.0205565 | 0.0660701 | 0.999999 | 0.999999 | 0.993639 | 0.998896 | 0.977654 | 0.978455 | 0.020636 | 0.0390332 |
| CMUMOSI | 0.1 | text | 2-3 | 56 | 0.0789923 | 0.0892104 | 0.105543 | 0.0102181 | 0.0163323 | 0.995909 | 0.995909 | 0.993994 | 0.979313 | 0.958489 | 0.960452 | 0.0112181 | 0.00523613 |
| CMUMOSI | 0.1 | visual | 1 | 424 | 0.000867966 | 0.0214786 | 0.198225 | 0.0206107 | 0.176746 | 0.999999 | 0.999999 | 0.965604 | 0.999807 | 0.978546 | 1.02201 | 0.0207067 | 0.0887313 |
| CMUMOSI | 0.1 | visual | 2-3 | 56 | 0.150842 | 0.156421 | 0.195083 | 0.00557898 | 0.038662 | 0.981104 | 0.981104 | 0.973644 | 0.993383 | 0.972253 | 0.988935 | 0.00581072 | 0.0174355 |
