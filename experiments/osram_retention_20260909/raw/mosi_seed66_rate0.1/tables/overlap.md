# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 400 | 0.0613019 | 0.0488895 | 0.0148475 | 0.00627096 | 0.0203217 |
| CMUMOSI | 0.1 | [.1,.2) | 227 | 0.142954 | 0.112835 | 0.073646 | 0.0363524 | 0.0191756 |
| CMUMOSI | 0.1 | [.2,.4) | 824 | 0.298886 | 0.225775 | 0.14215 | 0.0820928 | 0.0187647 |
| CMUMOSI | 0.1 | [.4,.6) | 201 | 0.472644 | 0.344353 | 0.215717 | 0.140473 | 0.0187752 |
| CMUMOSI | 0.1 | [.6,1] | 4 | 0.6389 | 0.513239 | 0.375149 | 0.322088 | 0.02098 |
