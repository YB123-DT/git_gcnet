# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | [0,.1) | 2283 | 0.0507728 | 0.0479937 | 0.00866927 | 0.00207197 | 0.0174273 |
| CMUMOSI | 0.5 | [.1,.2) | 1139 | 0.144003 | 0.133657 | 0.0510622 | 0.0214765 | 0.0158891 |
| CMUMOSI | 0.5 | [.2,.4) | 2889 | 0.294072 | 0.264686 | 0.108105 | 0.0595218 | 0.0141185 |
| CMUMOSI | 0.5 | [.4,.6) | 794 | 0.470977 | 0.41777 | 0.221858 | 0.126119 | 0.0132089 |
| CMUMOSI | 0.5 | [.6,1] | 23 | 0.612163 | 0.496281 | 0.231452 | 0.108977 | 0.0141914 |
