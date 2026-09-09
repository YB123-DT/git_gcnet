# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | [0,.1) | 1707 | 0.0602129 | 0.0583222 | 0.0118748 | 0.00235084 | 0.0172778 |
| CMUMOSI | 0.5 | [.1,.2) | 1514 | 0.135923 | 0.127939 | 0.0363944 | 0.0134619 | 0.0169785 |
| CMUMOSI | 0.5 | [.2,.4) | 1911 | 0.308862 | 0.284827 | 0.0909664 | 0.0470609 | 0.0143027 |
| CMUMOSI | 0.5 | [.4,.6) | 1606 | 0.46405 | 0.409217 | 0.180755 | 0.0938654 | 0.0142737 |
| CMUMOSI | 0.5 | [.6,1] | 206 | 0.623788 | 0.554683 | 0.385398 | 0.218801 | 0.0116874 |
