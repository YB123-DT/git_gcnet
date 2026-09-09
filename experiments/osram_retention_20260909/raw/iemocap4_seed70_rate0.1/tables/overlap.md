# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | [0,.1) | 384 | 0.0589919 | 0.0506275 | 0.0347751 | 0.0250916 | 0.0172125 |
| IEMOCAPFour | 0.1 | [.1,.2) | 616 | 0.151913 | 0.123263 | 0.11557 | 0.100137 | 0.0167578 |
| IEMOCAPFour | 0.1 | [.2,.4) | 1214 | 0.296276 | 0.230552 | 0.252936 | 0.232923 | 0.0168097 |
| IEMOCAPFour | 0.1 | [.4,.6) | 714 | 0.488478 | 0.362974 | 0.448139 | 0.416028 | 0.0171264 |
| IEMOCAPFour | 0.1 | [.6,1] | 136 | 0.661174 | 0.468534 | 0.597132 | 0.577632 | 0.0170377 |
