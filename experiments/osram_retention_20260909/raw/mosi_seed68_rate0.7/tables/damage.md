# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | audio | 1 | 1224 | 0.00056467 | 0.0210266 | 0.109782 | 0.020462 | 0.0887558 | 1 | 1 | 0.987299 | 0.999612 | 0.978977 | 0.971878 | 0.0205254 | 0.0406042 |
| CMUMOSI | 0.7 | audio | 2-3 | 1056 | 0.12742 | 0.137601 | 0.160221 | 0.0101808 | 0.0226201 | 0.983965 | 0.983965 | 0.977873 | 0.962216 | 0.942353 | 0.939893 | 0.0120105 | 0.00225454 |
| CMUMOSI | 0.7 | audio | 4-7 | 416 | 0.185542 | 0.196821 | 0.207591 | 0.0112798 | 0.0107698 | 0.975822 | 0.975822 | 0.973072 | 0.910401 | 0.891607 | 0.889664 | 0.0133025 | 0.000488713 |
| CMUMOSI | 0.7 | audio | 8+ | 72 | 0.255493 | 0.267771 | 0.274282 | 0.0122788 | 0.00651046 | 0.971515 | 0.971515 | 0.969261 | 0.814883 | 0.79806 | 0.790855 | 0.0146486 | 4.84809e-05 |
| CMUMOSI | 0.7 | text | 1 | 1232 | 0.00177082 | 0.0219225 | 0.111927 | 0.0201517 | 0.0900048 | 0.999999 | 0.999999 | 0.990464 | 0.998727 | 0.978111 | 0.936842 | 0.0202982 | 0.0361531 |
| CMUMOSI | 0.7 | text | 2-3 | 1280 | 0.115505 | 0.128464 | 0.150604 | 0.0129588 | 0.02214 | 0.990228 | 0.990228 | 0.986089 | 0.934362 | 0.915076 | 0.90586 | 0.0147749 | 0.00242282 |
| CMUMOSI | 0.7 | text | 4-7 | 520 | 0.180857 | 0.193664 | 0.202288 | 0.0128071 | 0.00862398 | 0.984464 | 0.984464 | 0.98227 | 0.880531 | 0.862357 | 0.861335 | 0.0142077 | 0.00119841 |
| CMUMOSI | 0.7 | text | 8+ | 72 | 0.274514 | 0.286518 | 0.284706 | 0.0120046 | -0.00181223 | 0.982656 | 0.982656 | 0.983618 | 0.790037 | 0.773731 | 0.768156 | 0.0131758 | 0.000930898 |
| CMUMOSI | 0.7 | visual | 1 | 1296 | 0.000953021 | 0.0211083 | 0.206569 | 0.0201553 | 0.185461 | 1 | 1 | 0.954341 | 0.999547 | 0.978914 | 1.00183 | 0.0202969 | 0.0575945 |
| CMUMOSI | 0.7 | visual | 2-3 | 984 | 0.243655 | 0.249426 | 0.311566 | 0.00577123 | 0.0621401 | 0.945928 | 0.945928 | 0.929267 | 0.997616 | 0.977019 | 0.98658 | 0.00848112 | 0.00716078 |
| CMUMOSI | 0.7 | visual | 4-7 | 400 | 0.394734 | 0.397992 | 0.433136 | 0.00325726 | 0.0351444 | 0.896708 | 0.896708 | 0.882328 | 0.952448 | 0.932782 | 0.946816 | 0.00603554 | 0.00211502 |
| CMUMOSI | 0.7 | visual | 8+ | 56 | 0.536147 | 0.536107 | 0.447964 | -4.01844e-05 | -0.088143 | 0.844483 | 0.844483 | 0.887994 | 0.953619 | 0.933926 | 0.883677 | 0.00246578 | -0.000222459 |
