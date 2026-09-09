# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | audio | 1 | 2376 | 0.00174731 | 0.0209114 | 0.265055 | 0.0191641 | 0.244144 | 0.999999 | 0.999999 | 0.945529 | 0.999085 | 0.97914 | 0.966323 | 0.0192281 | 0.194063 |
| IEMOCAPFour | 0.5 | audio | 2-3 | 1520 | 0.294057 | 0.296258 | 0.380402 | 0.00220071 | 0.0841441 | 0.935677 | 0.935677 | 0.905819 | 0.959724 | 0.940565 | 0.938858 | 0.00244363 | 0.0355056 |
| IEMOCAPFour | 0.5 | audio | 4-7 | 344 | 0.459544 | 0.458928 | 0.523096 | -0.00061577 | 0.0641686 | 0.869757 | 0.869757 | 0.836513 | 0.92809 | 0.909564 | 0.907462 | 0.000245512 | 0.0306903 |
| IEMOCAPFour | 0.5 | text | 1 | 2384 | 0.00244215 | 0.0216119 | 0.253132 | 0.0191698 | 0.23152 | 0.999998 | 0.999998 | 0.943824 | 0.998396 | 0.978465 | 0.948331 | 0.0192613 | 0.158359 |
| IEMOCAPFour | 0.5 | text | 2-3 | 1376 | 0.289794 | 0.293908 | 0.361663 | 0.00411428 | 0.0677557 | 0.93012 | 0.93012 | 0.906634 | 0.930907 | 0.912321 | 0.902501 | 0.00449383 | 0.0245854 |
| IEMOCAPFour | 0.5 | text | 4-7 | 456 | 0.500529 | 0.501195 | 0.544331 | 0.000666023 | 0.0431357 | 0.831709 | 0.831709 | 0.809691 | 0.888903 | 0.871154 | 0.877417 | 0.00188024 | 0.011072 |
| IEMOCAPFour | 0.5 | text | 8+ | 8 | 0.816436 | 0.810061 | 0.924424 | -0.00637545 | 0.114364 | 0.650943 | 0.650943 | 0.594922 | 0.974188 | 0.954731 | 1.07392 | -0.00766248 | 0.153979 |
| IEMOCAPFour | 0.5 | visual | 1 | 2504 | 0.00179187 | 0.0210246 | 0.270544 | 0.0192327 | 0.249519 | 0.999999 | 0.999999 | 0.942648 | 0.998972 | 0.979029 | 0.973031 | 0.0193329 | 0.186281 |
| IEMOCAPFour | 0.5 | visual | 2-3 | 1520 | 0.303743 | 0.305438 | 0.399772 | 0.00169549 | 0.0943334 | 0.932505 | 0.932505 | 0.899144 | 0.967144 | 0.947837 | 0.949452 | 0.00217827 | 0.0396886 |
| IEMOCAPFour | 0.5 | visual | 4-7 | 280 | 0.455281 | 0.454897 | 0.49945 | -0.00038397 | 0.0445527 | 0.87964 | 0.87964 | 0.850446 | 0.937973 | 0.919252 | 0.900706 | 0.000736445 | 0.0180998 |
