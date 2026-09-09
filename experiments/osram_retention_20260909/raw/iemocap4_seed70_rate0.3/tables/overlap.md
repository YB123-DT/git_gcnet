# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | [0,.1) | 1364 | 0.0537155 | 0.0496321 | 0.0302881 | 0.0197831 | 0.0146984 |
| IEMOCAPFour | 0.3 | [.1,.2) | 1667 | 0.150656 | 0.132767 | 0.104695 | 0.0948134 | 0.0142655 |
| IEMOCAPFour | 0.3 | [.2,.4) | 2992 | 0.292704 | 0.24533 | 0.22999 | 0.210152 | 0.0145271 |
| IEMOCAPFour | 0.3 | [.4,.6) | 1576 | 0.48401 | 0.392773 | 0.401617 | 0.382962 | 0.0147736 |
| IEMOCAPFour | 0.3 | [.6,1] | 337 | 0.660314 | 0.512465 | 0.572434 | 0.553157 | 0.0147826 |
