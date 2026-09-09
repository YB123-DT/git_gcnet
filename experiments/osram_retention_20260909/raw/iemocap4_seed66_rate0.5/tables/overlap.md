# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | [0,.1) | 2748 | 0.0512966 | 0.0497659 | 0.0200223 | 0.00898027 | 0.0121396 |
| IEMOCAPFour | 0.5 | [.1,.2) | 2884 | 0.150665 | 0.141908 | 0.0765366 | 0.0641686 | 0.0115592 |
| IEMOCAPFour | 0.5 | [.2,.4) | 4326 | 0.290897 | 0.263412 | 0.181384 | 0.158204 | 0.0118967 |
| IEMOCAPFour | 0.5 | [.4,.6) | 2546 | 0.489699 | 0.435195 | 0.375547 | 0.345465 | 0.0117669 |
| IEMOCAPFour | 0.5 | [.6,1] | 536 | 0.652686 | 0.57601 | 0.559908 | 0.52485 | 0.0120551 |
