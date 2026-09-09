# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | [0,.1) | 1194 | 0.0537776 | 0.0470262 | 0.0106532 | 0.00327174 | 0.0187606 |
| CMUMOSI | 0.3 | [.1,.2) | 635 | 0.144084 | 0.124507 | 0.0622705 | 0.0229072 | 0.0180139 |
| CMUMOSI | 0.3 | [.2,.4) | 1939 | 0.297006 | 0.243654 | 0.126529 | 0.0646332 | 0.0170924 |
| CMUMOSI | 0.3 | [.4,.6) | 492 | 0.474221 | 0.372006 | 0.28051 | 0.149678 | 0.0172438 |
| CMUMOSI | 0.3 | [.6,1] | 4 | 0.611888 | 0.49772 | 0.499394 | 0.213351 | 0.0202641 |
