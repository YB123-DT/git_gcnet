# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | [0,.1) | 594 | 0.0581709 | 0.0581674 | 0.00386022 | 0.000479421 | 0.0141626 |
| CMUMOSI | 0.5 | [.1,.2) | 1087 | 0.151357 | 0.150026 | 0.023321 | 0.00901084 | 0.0132053 |
| CMUMOSI | 0.5 | [.2,.4) | 3149 | 0.311744 | 0.287414 | 0.104292 | 0.0484003 | 0.0145162 |
| CMUMOSI | 0.5 | [.4,.6) | 2228 | 0.45929 | 0.410808 | 0.193767 | 0.122843 | 0.0133769 |
| CMUMOSI | 0.5 | [.6,1] | 102 | 0.636845 | 0.515874 | 0.25159 | 0.167041 | 0.0124197 |
