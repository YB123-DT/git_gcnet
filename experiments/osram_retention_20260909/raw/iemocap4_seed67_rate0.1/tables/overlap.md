# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | [0,.1) | 261 | 0.0601735 | 0.0503991 | 0.0395226 | 0.0277087 | 0.0175582 |
| IEMOCAPFour | 0.1 | [.1,.2) | 603 | 0.153791 | 0.119445 | 0.124709 | 0.11187 | 0.0174229 |
| IEMOCAPFour | 0.1 | [.2,.4) | 1078 | 0.29358 | 0.225523 | 0.256445 | 0.231546 | 0.0177577 |
| IEMOCAPFour | 0.1 | [.4,.6) | 485 | 0.483934 | 0.354776 | 0.436849 | 0.411672 | 0.0173431 |
| IEMOCAPFour | 0.1 | [.6,1] | 213 | 0.675194 | 0.462336 | 0.658757 | 0.629323 | 0.0178017 |
