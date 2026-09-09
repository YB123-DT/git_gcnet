# Distance

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | history_query_count | distance_mean | distance_median | distance_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 128 | 1.09375 | 1 | 1 |
| IEMOCAPFour | 0.1 | text | 137 | 1.16058 | 1 | 2 |
| IEMOCAPFour | 0.1 | visual | 111 | 1.14414 | 1 | 2 |
