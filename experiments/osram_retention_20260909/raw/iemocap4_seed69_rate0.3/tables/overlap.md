# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | [0,.1) | 1415 | 0.0527101 | 0.049325 | 0.0240294 | 0.0134291 | 0.0137635 |
| IEMOCAPFour | 0.3 | [.1,.2) | 1861 | 0.150105 | 0.130252 | 0.095353 | 0.0782241 | 0.0142309 |
| IEMOCAPFour | 0.3 | [.2,.4) | 2794 | 0.289656 | 0.240061 | 0.216735 | 0.188879 | 0.0140989 |
| IEMOCAPFour | 0.3 | [.4,.6) | 1749 | 0.487074 | 0.387294 | 0.382235 | 0.344819 | 0.0137795 |
| IEMOCAPFour | 0.3 | [.6,1] | 557 | 0.679531 | 0.518742 | 0.607207 | 0.578938 | 0.0144342 |
