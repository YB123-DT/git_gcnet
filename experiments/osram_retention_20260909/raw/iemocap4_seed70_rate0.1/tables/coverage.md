# Coverage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | missing_count | no_history_count | no_history_ratio |
| --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 146 | 9 | 0.0616438 |
| IEMOCAPFour | 0.1 | text | 133 | 4 | 0.0300752 |
| IEMOCAPFour | 0.1 | visual | 117 | 0 | 0 |
