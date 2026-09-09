# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 1 | 488 | 0.000996701 | 0.0214671 | 0.313776 | 0.0204704 | 0.292309 | 0.999999 | 0.999999 | 0.932267 | 0.999792 | 0.978581 | 1.04673 | 0.0207331 | 0.174847 |
| CMUMOSI | 0.1 | audio | 2-3 | 80 | 0.223651 | 0.227693 | 0.301058 | 0.00404211 | 0.0733654 | 0.959767 | 0.959767 | 0.934195 | 0.98601 | 0.96509 | 0.991125 | 0.00452703 | 0.01249 |
| CMUMOSI | 0.1 | text | 1 | 472 | 0.00200789 | 0.0225062 | 0.126893 | 0.0204983 | 0.104387 | 0.999999 | 0.999999 | 0.987107 | 0.998733 | 0.977546 | 0.948098 | 0.0205192 | 0.0661423 |
| CMUMOSI | 0.1 | text | 2-3 | 64 | 0.102734 | 0.113446 | 0.13568 | 0.0107117 | 0.0222338 | 0.991596 | 0.991596 | 0.985608 | 0.950659 | 0.930491 | 0.935189 | 0.0122902 | 0.000981491 |
| CMUMOSI | 0.1 | visual | 1 | 496 | 0.00100148 | 0.0215588 | 0.254189 | 0.0205574 | 0.23263 | 0.999999 | 0.999999 | 0.954508 | 0.999674 | 0.978466 | 1.01992 | 0.0206384 | 0.154948 |
| CMUMOSI | 0.1 | visual | 2-3 | 16 | 0.136921 | 0.142097 | 0.109606 | 0.00517661 | -0.032491 | 0.99044 | 0.99044 | 0.994674 | 0.964729 | 0.944259 | 0.951283 | 0.00452945 | -0.0255167 |
