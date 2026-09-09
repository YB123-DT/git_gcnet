# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | [0,.1) | 769 | 0.0515708 | 0.0515518 | 0.00344812 | 0.000507131 | 0.0133043 |
| CMUMOSI | 0.7 | [.1,.2) | 1523 | 0.15417 | 0.15393 | 0.0227814 | 0.00884113 | 0.0123361 |
| CMUMOSI | 0.7 | [.2,.4) | 3723 | 0.309743 | 0.300026 | 0.102688 | 0.0522017 | 0.0132583 |
| CMUMOSI | 0.7 | [.4,.6) | 2646 | 0.460058 | 0.438858 | 0.194434 | 0.135821 | 0.0123846 |
| CMUMOSI | 0.7 | [.6,1] | 83 | 0.633302 | 0.574806 | 0.373047 | 0.310741 | 0.0133093 |
