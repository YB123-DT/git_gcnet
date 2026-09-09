# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | [0,.1) | 874 | 0.0574334 | 0.0534517 | 0.0143821 | 0.00288714 | 0.0181776 |
| CMUMOSI | 0.3 | [.1,.2) | 1052 | 0.137787 | 0.122654 | 0.0365394 | 0.0165094 | 0.0178467 |
| CMUMOSI | 0.3 | [.2,.4) | 1285 | 0.318286 | 0.273939 | 0.100222 | 0.0538788 | 0.0160881 |
| CMUMOSI | 0.3 | [.4,.6) | 1139 | 0.463172 | 0.3823 | 0.177804 | 0.0961865 | 0.0157778 |
| CMUMOSI | 0.3 | [.6,1] | 162 | 0.624595 | 0.510362 | 0.365914 | 0.173753 | 0.0139023 |
