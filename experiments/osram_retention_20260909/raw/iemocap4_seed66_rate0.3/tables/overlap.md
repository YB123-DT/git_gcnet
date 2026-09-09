# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | [0,.1) | 1327 | 0.0537217 | 0.0500358 | 0.0250001 | 0.0137344 | 0.0145581 |
| IEMOCAPFour | 0.3 | [.1,.2) | 1813 | 0.15201 | 0.133907 | 0.0945001 | 0.0822667 | 0.0142379 |
| IEMOCAPFour | 0.3 | [.2,.4) | 2955 | 0.28973 | 0.240452 | 0.216529 | 0.182597 | 0.0143254 |
| IEMOCAPFour | 0.3 | [.4,.6) | 1971 | 0.488746 | 0.393364 | 0.417691 | 0.377915 | 0.0146059 |
| IEMOCAPFour | 0.3 | [.6,1] | 438 | 0.656802 | 0.521979 | 0.596915 | 0.550721 | 0.0154451 |
