# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | [0,.1) | 2697 | 0.0524172 | 0.0516946 | 0.0134481 | 0.00248172 | 0.0172301 |
| CMUMOSI | 0.7 | [.1,.2) | 2147 | 0.145882 | 0.141539 | 0.0361187 | 0.0135485 | 0.014385 |
| CMUMOSI | 0.7 | [.2,.4) | 1820 | 0.299279 | 0.282058 | 0.0786944 | 0.0276316 | 0.0132591 |
| CMUMOSI | 0.7 | [.4,.6) | 1730 | 0.462213 | 0.440166 | 0.136014 | 0.0607904 | 0.0114826 |
| CMUMOSI | 0.7 | [.6,1] | 102 | 0.614595 | 0.61195 | 0.258833 | 0.178845 | 0.012614 |
