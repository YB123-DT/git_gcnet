# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | [0,.1) | 1326 | 0.0561502 | 0.052292 | 0.0292446 | 0.0182712 | 0.0148548 |
| IEMOCAPFour | 0.3 | [.1,.2) | 1830 | 0.150779 | 0.131856 | 0.101656 | 0.0909004 | 0.0145136 |
| IEMOCAPFour | 0.3 | [.2,.4) | 3178 | 0.288535 | 0.244798 | 0.216533 | 0.198902 | 0.0140377 |
| IEMOCAPFour | 0.3 | [.4,.6) | 1352 | 0.483448 | 0.391949 | 0.397433 | 0.372352 | 0.0145968 |
| IEMOCAPFour | 0.3 | [.6,1] | 498 | 0.68105 | 0.526793 | 0.606078 | 0.584683 | 0.0147148 |
