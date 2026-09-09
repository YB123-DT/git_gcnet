# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | [0,.1) | 2420 | 0.0622227 | 0.0615618 | 0.00844354 | 0.00151934 | 0.0161231 |
| CMUMOSI | 0.7 | [.1,.2) | 1853 | 0.137019 | 0.133998 | 0.0280455 | 0.00915664 | 0.0158229 |
| CMUMOSI | 0.7 | [.2,.4) | 2308 | 0.303727 | 0.294553 | 0.0725312 | 0.033819 | 0.0125951 |
| CMUMOSI | 0.7 | [.4,.6) | 1787 | 0.461565 | 0.435652 | 0.152931 | 0.0723739 | 0.0131599 |
| CMUMOSI | 0.7 | [.6,1] | 240 | 0.621994 | 0.588221 | 0.327437 | 0.16947 | 0.0103662 |
