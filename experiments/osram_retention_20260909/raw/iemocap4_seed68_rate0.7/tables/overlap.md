# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | [0,.1) | 4022 | 0.0498448 | 0.049346 | 0.0163921 | 0.00794486 | 0.0103289 |
| IEMOCAPFour | 0.7 | [.1,.2) | 3859 | 0.14915 | 0.145618 | 0.0661426 | 0.0572105 | 0.0101205 |
| IEMOCAPFour | 0.7 | [.2,.4) | 5405 | 0.290895 | 0.279235 | 0.154544 | 0.138071 | 0.00964173 |
| IEMOCAPFour | 0.7 | [.4,.6) | 2333 | 0.482715 | 0.454001 | 0.309783 | 0.293024 | 0.00953576 |
| IEMOCAPFour | 0.7 | [.6,1] | 629 | 0.660483 | 0.601683 | 0.493708 | 0.486783 | 0.0100549 |
