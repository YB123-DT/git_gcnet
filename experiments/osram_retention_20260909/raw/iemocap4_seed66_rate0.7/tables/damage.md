# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 1 | 2256 | 0.0016343 | 0.0208119 | 0.258333 | 0.0191776 | 0.237521 | 0.999999 | 0.999999 | 0.942897 | 0.999128 | 0.979233 | 0.971688 | 0.0193057 | 0.152309 |
| IEMOCAPFour | 0.7 | audio | 2-3 | 2064 | 0.299395 | 0.302265 | 0.405516 | 0.00286961 | 0.103251 | 0.926807 | 0.926807 | 0.887135 | 0.960605 | 0.94148 | 0.936933 | 0.00382061 | 0.0409511 |
| IEMOCAPFour | 0.7 | audio | 4-7 | 840 | 0.455246 | 0.456801 | 0.502227 | 0.00155415 | 0.0454259 | 0.867249 | 0.867249 | 0.843453 | 0.88976 | 0.872046 | 0.871147 | 0.00276603 | 0.0117407 |
| IEMOCAPFour | 0.7 | audio | 8+ | 136 | 0.593191 | 0.593991 | 0.641981 | 0.000799438 | 0.0479903 | 0.804813 | 0.804813 | 0.772822 | 0.842546 | 0.825784 | 0.822125 | 0.00247802 | 0.0107023 |
| IEMOCAPFour | 0.7 | text | 1 | 2408 | 0.00225939 | 0.0214631 | 0.267062 | 0.0192037 | 0.245599 | 0.999999 | 0.999999 | 0.94233 | 0.998483 | 0.978601 | 0.950197 | 0.0193027 | 0.172225 |
| IEMOCAPFour | 0.7 | text | 2-3 | 2080 | 0.302279 | 0.305542 | 0.391827 | 0.00326228 | 0.0862852 | 0.930314 | 0.930314 | 0.897051 | 0.948418 | 0.929532 | 0.924514 | 0.00440286 | 0.0275365 |
| IEMOCAPFour | 0.7 | text | 4-7 | 752 | 0.440452 | 0.442666 | 0.490601 | 0.00221348 | 0.0479354 | 0.878547 | 0.878547 | 0.852043 | 0.885079 | 0.867452 | 0.874119 | 0.00342692 | 0.0119764 |
| IEMOCAPFour | 0.7 | text | 8+ | 56 | 0.654988 | 0.653068 | 0.680155 | -0.00192015 | 0.0270869 | 0.754779 | 0.754779 | 0.741881 | 0.879593 | 0.862075 | 0.895831 | -0.000194609 | 0.00340116 |
| IEMOCAPFour | 0.7 | visual | 1 | 2264 | 0.001626 | 0.0208934 | 0.237597 | 0.0192674 | 0.216704 | 0.999999 | 0.999999 | 0.950585 | 0.999042 | 0.979149 | 0.970644 | 0.0193291 | 0.142417 |
| IEMOCAPFour | 0.7 | visual | 2-3 | 2056 | 0.282417 | 0.285656 | 0.382767 | 0.00323883 | 0.0971114 | 0.934391 | 0.934391 | 0.897995 | 0.964617 | 0.945413 | 0.948319 | 0.00394807 | 0.033291 |
| IEMOCAPFour | 0.7 | visual | 4-7 | 1160 | 0.48629 | 0.486354 | 0.527856 | 6.41254e-05 | 0.0415024 | 0.849684 | 0.849684 | 0.829776 | 0.933506 | 0.914921 | 0.919275 | 0.00085023 | 0.0102785 |
| IEMOCAPFour | 0.7 | visual | 8+ | 104 | 0.690992 | 0.687937 | 0.73989 | -0.0030552 | 0.0519539 | 0.76639 | 0.76639 | 0.739766 | 0.981473 | 0.961941 | 0.981264 | 0.00197427 | 0.0296899 |
