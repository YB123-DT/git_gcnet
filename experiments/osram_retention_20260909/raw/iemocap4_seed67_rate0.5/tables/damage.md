# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | audio | 1 | 2504 | 0.0017224 | 0.021116 | 0.256291 | 0.0193936 | 0.235175 | 0.999999 | 0.999999 | 0.947282 | 0.999106 | 0.978935 | 0.974332 | 0.0194043 | 0.17233 |
| IEMOCAPFour | 0.5 | audio | 2-3 | 1536 | 0.27963 | 0.282009 | 0.375204 | 0.00237873 | 0.0931954 | 0.940788 | 0.940788 | 0.908585 | 0.968511 | 0.948957 | 0.949524 | 0.00265494 | 0.0424627 |
| IEMOCAPFour | 0.5 | audio | 4-7 | 232 | 0.467033 | 0.466339 | 0.540347 | -0.000693251 | 0.0740079 | 0.869086 | 0.869086 | 0.828777 | 0.93868 | 0.919721 | 0.923073 | -0.000110269 | 0.0206478 |
| IEMOCAPFour | 0.5 | text | 1 | 2416 | 0.00240415 | 0.0217934 | 0.279204 | 0.0193893 | 0.25741 | 0.999998 | 0.999998 | 0.934564 | 0.998439 | 0.978281 | 0.94578 | 0.019381 | 0.187215 |
| IEMOCAPFour | 0.5 | text | 2-3 | 1584 | 0.302137 | 0.305815 | 0.390678 | 0.00367816 | 0.0848627 | 0.92584 | 0.92584 | 0.8906 | 0.939143 | 0.920184 | 0.913384 | 0.0043422 | 0.0369421 |
| IEMOCAPFour | 0.5 | text | 4-7 | 352 | 0.457568 | 0.459249 | 0.504662 | 0.00168151 | 0.045413 | 0.859206 | 0.859206 | 0.834191 | 0.893501 | 0.875464 | 0.873356 | 0.00249242 | 0.00925846 |
| IEMOCAPFour | 0.5 | text | 8+ | 8 | 0.541385 | 0.541382 | 0.72593 | -3.10689e-06 | 0.184548 | 0.831844 | 0.831844 | 0.794868 | 0.884583 | 0.866708 | 1.01493 | 0.00225483 | 0.0529274 |
| IEMOCAPFour | 0.5 | visual | 1 | 2368 | 0.00168104 | 0.0211139 | 0.279305 | 0.0194329 | 0.258191 | 0.999999 | 0.999999 | 0.937056 | 0.999106 | 0.978934 | 0.975332 | 0.019418 | 0.187814 |
| IEMOCAPFour | 0.5 | visual | 2-3 | 1528 | 0.313759 | 0.315143 | 0.395408 | 0.00138423 | 0.080265 | 0.923862 | 0.923862 | 0.892229 | 0.96807 | 0.948525 | 0.944471 | 0.00162351 | 0.0293071 |
| IEMOCAPFour | 0.5 | visual | 4-7 | 368 | 0.478332 | 0.477795 | 0.514263 | -0.000536359 | 0.0364673 | 0.849945 | 0.849945 | 0.829009 | 0.934121 | 0.915256 | 0.918009 | 0.000236258 | 0.0128247 |
