# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | [0,.1) | 4072 | 0.0492627 | 0.048868 | 0.0146039 | 0.00597677 | 0.010218 |
| IEMOCAPFour | 0.7 | [.1,.2) | 3799 | 0.147644 | 0.144833 | 0.063332 | 0.0519623 | 0.0102659 |
| IEMOCAPFour | 0.7 | [.2,.4) | 4796 | 0.286851 | 0.27668 | 0.144928 | 0.126074 | 0.00956148 |
| IEMOCAPFour | 0.7 | [.4,.6) | 2768 | 0.488063 | 0.462795 | 0.290927 | 0.270996 | 0.00923058 |
| IEMOCAPFour | 0.7 | [.6,1] | 973 | 0.684634 | 0.643136 | 0.492833 | 0.509272 | 0.00874809 |
