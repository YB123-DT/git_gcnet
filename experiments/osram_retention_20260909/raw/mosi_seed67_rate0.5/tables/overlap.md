# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | [0,.1) | 1989 | 0.0506962 | 0.0487801 | 0.0131167 | 0.00281164 | 0.0181149 |
| CMUMOSI | 0.5 | [.1,.2) | 1759 | 0.146855 | 0.137006 | 0.0373366 | 0.016253 | 0.0165145 |
| CMUMOSI | 0.5 | [.2,.4) | 1540 | 0.295397 | 0.263668 | 0.0920273 | 0.0393661 | 0.0155528 |
| CMUMOSI | 0.5 | [.4,.6) | 1593 | 0.466867 | 0.422634 | 0.153274 | 0.0785361 | 0.0139417 |
| CMUMOSI | 0.5 | [.6,1] | 71 | 0.611453 | 0.607386 | 0.24856 | 0.169213 | 0.0124926 |
