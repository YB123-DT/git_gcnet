# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | [0,.1) | 312 | 0.05945 | 0.0504382 | 0.0389162 | 0.0237502 | 0.0166897 |
| IEMOCAPFour | 0.1 | [.1,.2) | 582 | 0.151422 | 0.120115 | 0.120827 | 0.100839 | 0.0168408 |
| IEMOCAPFour | 0.1 | [.2,.4) | 1102 | 0.296899 | 0.224924 | 0.259957 | 0.218845 | 0.0169306 |
| IEMOCAPFour | 0.1 | [.4,.6) | 832 | 0.489673 | 0.360232 | 0.449141 | 0.40949 | 0.0167789 |
| IEMOCAPFour | 0.1 | [.6,1] | 180 | 0.653967 | 0.469842 | 0.662056 | 0.604847 | 0.01754 |
