# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 1 | 2440 | 0.00173987 | 0.0209716 | 0.251601 | 0.0192317 | 0.230629 | 0.999999 | 0.999999 | 0.949572 | 0.99902 | 0.979075 | 0.966161 | 0.0192922 | 0.172443 |
| IEMOCAPFour | 0.7 | audio | 2-3 | 2200 | 0.293813 | 0.296666 | 0.389208 | 0.00285277 | 0.0925422 | 0.935936 | 0.935936 | 0.899933 | 0.952236 | 0.933226 | 0.926949 | 0.00322048 | 0.0359893 |
| IEMOCAPFour | 0.7 | audio | 4-7 | 712 | 0.466649 | 0.467048 | 0.511729 | 0.000398391 | 0.0446811 | 0.860919 | 0.860919 | 0.837882 | 0.901515 | 0.883521 | 0.881715 | 0.000738621 | 0.0110818 |
| IEMOCAPFour | 0.7 | audio | 8+ | 88 | 0.591457 | 0.590947 | 0.605829 | -0.000509353 | 0.0148816 | 0.781302 | 0.781302 | 0.774373 | 0.844602 | 0.827745 | 0.834455 | 0.00127789 | 0.0018336 |
| IEMOCAPFour | 0.7 | text | 1 | 2424 | 0.00238672 | 0.0216539 | 0.237294 | 0.0192672 | 0.21564 | 0.999999 | 0.999999 | 0.949727 | 0.998343 | 0.978412 | 0.942642 | 0.019342 | 0.148906 |
| IEMOCAPFour | 0.7 | text | 2-3 | 1992 | 0.280029 | 0.284364 | 0.363933 | 0.00433562 | 0.0795684 | 0.93491 | 0.93491 | 0.906275 | 0.932367 | 0.913751 | 0.902245 | 0.00486819 | 0.0273765 |
| IEMOCAPFour | 0.7 | text | 4-7 | 792 | 0.445258 | 0.447331 | 0.478037 | 0.00207244 | 0.0307063 | 0.869759 | 0.869759 | 0.852465 | 0.87517 | 0.857695 | 0.856623 | 0.00282584 | 0.00614059 |
| IEMOCAPFour | 0.7 | text | 8+ | 88 | 0.553414 | 0.554528 | 0.582717 | 0.00111483 | 0.0281886 | 0.832077 | 0.832077 | 0.818979 | 0.833042 | 0.81639 | 0.842724 | 0.00202151 | 0.000864655 |
| IEMOCAPFour | 0.7 | visual | 1 | 2304 | 0.00162274 | 0.020969 | 0.254314 | 0.0193462 | 0.233345 | 0.999999 | 0.999999 | 0.947705 | 0.999014 | 0.97907 | 0.964908 | 0.019425 | 0.169407 |
| IEMOCAPFour | 0.7 | visual | 2-3 | 2272 | 0.2892 | 0.292026 | 0.384676 | 0.00282657 | 0.0926498 | 0.936995 | 0.936995 | 0.902588 | 0.953508 | 0.934473 | 0.929005 | 0.00316599 | 0.0379472 |
| IEMOCAPFour | 0.7 | visual | 4-7 | 1056 | 0.481647 | 0.481835 | 0.525418 | 0.000188204 | 0.0435834 | 0.858207 | 0.858207 | 0.834823 | 0.908976 | 0.890832 | 0.895728 | 0.00111394 | 0.0114733 |
| IEMOCAPFour | 0.7 | visual | 8+ | 112 | 0.615221 | 0.614529 | 0.634186 | -0.000692011 | 0.0196568 | 0.772948 | 0.772948 | 0.762344 | 0.855869 | 0.838785 | 0.840065 | 0.000751138 | 0.00536597 |
