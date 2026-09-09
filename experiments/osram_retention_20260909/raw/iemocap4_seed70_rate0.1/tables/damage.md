# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 1 | 968 | 0.00181626 | 0.0208605 | 0.31695 | 0.0190442 | 0.296089 | 0.999999 | 0.999999 | 0.931122 | 0.999147 | 0.9792 | 0.980806 | 0.0191363 | 0.24648 |
| IEMOCAPFour | 0.1 | audio | 2-3 | 128 | 0.348201 | 0.347232 | 0.443885 | -0.000968758 | 0.0966525 | 0.917772 | 0.917772 | 0.879007 | 0.995841 | 0.97595 | 0.977923 | 0.000303902 | 0.063252 |
| IEMOCAPFour | 0.1 | text | 1 | 888 | 0.00258519 | 0.0216818 | 0.269877 | 0.0190967 | 0.248195 | 0.999998 | 0.999998 | 0.944316 | 0.998337 | 0.978407 | 0.945742 | 0.0191965 | 0.196933 |
| IEMOCAPFour | 0.1 | text | 2-3 | 144 | 0.252094 | 0.255835 | 0.360843 | 0.00374091 | 0.105008 | 0.950324 | 0.950324 | 0.911637 | 0.945465 | 0.926586 | 0.913717 | 0.00495601 | 0.0434808 |
| IEMOCAPFour | 0.1 | visual | 1 | 832 | 0.00175242 | 0.0208615 | 0.323532 | 0.0191091 | 0.30267 | 0.999999 | 0.999999 | 0.927779 | 0.999144 | 0.979197 | 0.976397 | 0.0192166 | 0.246869 |
| IEMOCAPFour | 0.1 | visual | 2-3 | 104 | 0.333786 | 0.335504 | 0.397666 | 0.00171798 | 0.0621617 | 0.921828 | 0.921828 | 0.899993 | 0.97224 | 0.952831 | 0.954364 | 0.00156955 | 0.0273261 |
