# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | [0,.1) | 4168 | 0.0493203 | 0.0488594 | 0.0169709 | 0.00751043 | 0.0100595 |
| IEMOCAPFour | 0.7 | [.1,.2) | 3761 | 0.14929 | 0.146006 | 0.0709135 | 0.0631799 | 0.00995326 |
| IEMOCAPFour | 0.7 | [.2,.4) | 5295 | 0.288884 | 0.278602 | 0.159716 | 0.147909 | 0.00960924 |
| IEMOCAPFour | 0.7 | [.4,.6) | 2671 | 0.488018 | 0.468137 | 0.315105 | 0.30797 | 0.00984548 |
| IEMOCAPFour | 0.7 | [.6,1] | 585 | 0.662775 | 0.620509 | 0.476981 | 0.483346 | 0.00871242 |
