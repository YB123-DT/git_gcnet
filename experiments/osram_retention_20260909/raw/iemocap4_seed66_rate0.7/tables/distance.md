# Distance

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | history_query_count | distance_mean | distance_median | distance_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 662 | 2.36103 | 2 | 5 |
| IEMOCAPFour | 0.7 | text | 662 | 2.14955 | 2 | 4 |
| IEMOCAPFour | 0.7 | visual | 698 | 2.4914 | 2 | 5 |
