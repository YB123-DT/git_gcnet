# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | [0,.1) | 1557 | 0.0520403 | 0.050697 | 0.0118107 | 0.00203801 | 0.0159583 |
| CMUMOSI | 0.5 | [.1,.2) | 1916 | 0.149757 | 0.142398 | 0.0406144 | 0.0166903 | 0.0151944 |
| CMUMOSI | 0.5 | [.2,.4) | 2361 | 0.28619 | 0.263062 | 0.0838055 | 0.0471053 | 0.0137242 |
| CMUMOSI | 0.5 | [.4,.6) | 998 | 0.491695 | 0.417989 | 0.208651 | 0.145008 | 0.0143642 |
| CMUMOSI | 0.5 | [.6,1] | 568 | 0.638701 | 0.58003 | 0.271774 | 0.227955 | 0.0126087 |
