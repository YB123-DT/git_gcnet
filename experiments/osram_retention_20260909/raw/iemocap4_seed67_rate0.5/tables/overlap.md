# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | [0,.1) | 2577 | 0.0511654 | 0.0494865 | 0.0220346 | 0.0117194 | 0.011937 |
| IEMOCAPFour | 0.5 | [.1,.2) | 2878 | 0.150787 | 0.140953 | 0.0820239 | 0.073754 | 0.0120783 |
| IEMOCAPFour | 0.5 | [.2,.4) | 4739 | 0.287845 | 0.261895 | 0.185151 | 0.169901 | 0.0117376 |
| IEMOCAPFour | 0.5 | [.4,.6) | 1981 | 0.483814 | 0.424767 | 0.350375 | 0.344091 | 0.011906 |
| IEMOCAPFour | 0.5 | [.6,1] | 721 | 0.680317 | 0.586523 | 0.563205 | 0.56397 | 0.0118154 |
