# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | [0,.1) | 301 | 0.0556954 | 0.0550797 | 0.0101412 | 0.00137165 | 0.0152465 |
| CMUMOSI | 0.3 | [.1,.2) | 559 | 0.15189 | 0.146161 | 0.041837 | 0.0211195 | 0.0151675 |
| CMUMOSI | 0.3 | [.2,.4) | 2082 | 0.312755 | 0.264675 | 0.134776 | 0.0711471 | 0.0160158 |
| CMUMOSI | 0.3 | [.4,.6) | 1602 | 0.460293 | 0.380809 | 0.241124 | 0.156483 | 0.0150342 |
| CMUMOSI | 0.3 | [.6,1] | 64 | 0.636815 | 0.462637 | 0.273492 | 0.177784 | 0.0165186 |
