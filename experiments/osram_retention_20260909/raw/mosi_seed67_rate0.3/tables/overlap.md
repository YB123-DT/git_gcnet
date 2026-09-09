# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | [0,.1) | 1126 | 0.0529646 | 0.0489629 | 0.0177185 | 0.00411229 | 0.0187732 |
| CMUMOSI | 0.3 | [.1,.2) | 1231 | 0.148169 | 0.130184 | 0.039241 | 0.0200674 | 0.0178902 |
| CMUMOSI | 0.3 | [.2,.4) | 1106 | 0.298 | 0.246022 | 0.113953 | 0.0445887 | 0.0168008 |
| CMUMOSI | 0.3 | [.4,.6) | 1124 | 0.468265 | 0.392561 | 0.157201 | 0.0843101 | 0.0156141 |
| CMUMOSI | 0.3 | [.6,1] | 53 | 0.615232 | 0.602498 | 0.163572 | 0.118581 | 0.0163325 |
