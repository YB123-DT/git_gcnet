# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | [0,.1) | 2825 | 0.0510606 | 0.0490569 | 0.0222781 | 0.0127055 | 0.0122054 |
| IEMOCAPFour | 0.5 | [.1,.2) | 2823 | 0.149458 | 0.139757 | 0.0843372 | 0.0740463 | 0.0117725 |
| IEMOCAPFour | 0.5 | [.2,.4) | 4390 | 0.29093 | 0.263393 | 0.184299 | 0.168866 | 0.0114086 |
| IEMOCAPFour | 0.5 | [.4,.6) | 2276 | 0.48335 | 0.427911 | 0.361012 | 0.346727 | 0.0121943 |
| IEMOCAPFour | 0.5 | [.6,1] | 454 | 0.667068 | 0.562367 | 0.539037 | 0.544275 | 0.0118017 |
