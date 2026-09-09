# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 1 | 2416 | 0.00169324 | 0.0211693 | 0.253533 | 0.019476 | 0.232364 | 0.999999 | 0.999999 | 0.947135 | 0.999046 | 0.978875 | 0.970101 | 0.0194496 | 0.164317 |
| IEMOCAPFour | 0.7 | audio | 2-3 | 2176 | 0.282536 | 0.285282 | 0.369753 | 0.00274615 | 0.0844711 | 0.938082 | 0.938082 | 0.908124 | 0.962626 | 0.943189 | 0.940227 | 0.00294001 | 0.0327818 |
| IEMOCAPFour | 0.7 | audio | 4-7 | 848 | 0.46777 | 0.468165 | 0.501286 | 0.000395083 | 0.0331215 | 0.86343 | 0.86343 | 0.846436 | 0.90849 | 0.890143 | 0.894296 | 0.00122082 | 0.00621587 |
| IEMOCAPFour | 0.7 | audio | 8+ | 88 | 0.553317 | 0.554925 | 0.591732 | 0.00160834 | 0.0368061 | 0.820091 | 0.820091 | 0.800552 | 0.808146 | 0.791836 | 0.803041 | 0.00313905 | 0.00599325 |
| IEMOCAPFour | 0.7 | text | 1 | 2400 | 0.00234945 | 0.0217913 | 0.26388 | 0.0194418 | 0.242089 | 0.999998 | 0.999998 | 0.940244 | 0.998436 | 0.978278 | 0.941481 | 0.0194396 | 0.169513 |
| IEMOCAPFour | 0.7 | text | 2-3 | 2040 | 0.296604 | 0.300944 | 0.389637 | 0.00434 | 0.0886932 | 0.927882 | 0.927882 | 0.8888 | 0.926469 | 0.907768 | 0.896295 | 0.00479397 | 0.0322485 |
| IEMOCAPFour | 0.7 | text | 4-7 | 736 | 0.48435 | 0.486337 | 0.520248 | 0.0019869 | 0.0339104 | 0.841262 | 0.841262 | 0.822083 | 0.864048 | 0.846611 | 0.847751 | 0.00263377 | 0.00945716 |
| IEMOCAPFour | 0.7 | text | 8+ | 72 | 0.609329 | 0.609817 | 0.613039 | 0.000487944 | 0.00322167 | 0.767392 | 0.767392 | 0.757993 | 0.797486 | 0.781385 | 0.780175 | 0.00104545 | -0.000301957 |
| IEMOCAPFour | 0.7 | visual | 1 | 2408 | 0.00157454 | 0.0211121 | 0.256207 | 0.0195375 | 0.235095 | 0.999999 | 0.999999 | 0.945802 | 0.999099 | 0.978928 | 0.96847 | 0.0195089 | 0.172113 |
| IEMOCAPFour | 0.7 | visual | 2-3 | 2208 | 0.28903 | 0.291796 | 0.385103 | 0.00276663 | 0.0933068 | 0.933491 | 0.933491 | 0.898422 | 0.960665 | 0.941269 | 0.940493 | 0.00311397 | 0.0359037 |
| IEMOCAPFour | 0.7 | visual | 4-7 | 896 | 0.471152 | 0.471203 | 0.508228 | 5.07456e-05 | 0.0370254 | 0.858379 | 0.858379 | 0.840619 | 0.915333 | 0.896852 | 0.904667 | 0.0006852 | 0.00549714 |
| IEMOCAPFour | 0.7 | visual | 8+ | 80 | 0.656255 | 0.653793 | 0.676964 | -0.00246277 | 0.0231709 | 0.736412 | 0.736412 | 0.719122 | 0.900763 | 0.882554 | 0.890578 | -0.0017353 | 0.00372237 |
