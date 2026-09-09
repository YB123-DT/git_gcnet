# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | [0,.1) | 2950 | 0.0514946 | 0.0501631 | 0.020279 | 0.00977152 | 0.0122976 |
| IEMOCAPFour | 0.5 | [.1,.2) | 3003 | 0.148988 | 0.141578 | 0.075872 | 0.0664796 | 0.0120621 |
| IEMOCAPFour | 0.5 | [.2,.4) | 4546 | 0.291375 | 0.26712 | 0.17735 | 0.160115 | 0.0111136 |
| IEMOCAPFour | 0.5 | [.4,.6) | 2172 | 0.485062 | 0.425191 | 0.340797 | 0.329385 | 0.0108879 |
| IEMOCAPFour | 0.5 | [.6,1] | 641 | 0.660457 | 0.562044 | 0.55402 | 0.531637 | 0.0116866 |
