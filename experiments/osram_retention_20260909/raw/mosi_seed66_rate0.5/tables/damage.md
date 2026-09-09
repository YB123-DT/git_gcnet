# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | audio | 1 | 1384 | 0.000639108 | 0.0215786 | 0.127304 | 0.0209395 | 0.105726 | 1 | 1 | 0.981511 | 0.999686 | 0.978428 | 0.977657 | 0.021016 | 0.0483463 |
| CMUMOSI | 0.5 | audio | 2-3 | 936 | 0.14153 | 0.149771 | 0.174243 | 0.00824043 | 0.024472 | 0.978604 | 0.978604 | 0.971657 | 0.96839 | 0.947802 | 0.942694 | 0.00982308 | 0.00339445 |
| CMUMOSI | 0.5 | audio | 4-7 | 224 | 0.23192 | 0.240384 | 0.283297 | 0.00846399 | 0.0429124 | 0.946323 | 0.946323 | 0.9237 | 0.913647 | 0.894229 | 0.900716 | 0.0105389 | 0.0024917 |
| CMUMOSI | 0.5 | audio | 8+ | 8 | 0.425651 | 0.429953 | 0.422649 | 0.00430244 | -0.0073038 | 0.833084 | 0.833084 | 0.846656 | 0.848866 | 0.830823 | 0.831044 | 0.00511357 | -0.00722082 |
| CMUMOSI | 0.5 | text | 1 | 1352 | 0.0018672 | 0.0225542 | 0.106537 | 0.020687 | 0.0839831 | 0.999999 | 0.999999 | 0.991285 | 0.998726 | 0.977488 | 0.950571 | 0.0207506 | 0.0434894 |
| CMUMOSI | 0.5 | text | 2-3 | 896 | 0.117515 | 0.129906 | 0.141686 | 0.0123905 | 0.0117798 | 0.989803 | 0.989803 | 0.988769 | 0.941534 | 0.921516 | 0.91791 | 0.0136427 | 0.00243799 |
| CMUMOSI | 0.5 | text | 4-7 | 104 | 0.138961 | 0.15218 | 0.168032 | 0.0132192 | 0.0158519 | 0.990654 | 0.990654 | 0.987302 | 0.912645 | 0.893242 | 0.892456 | 0.013691 | 0.00540524 |
| CMUMOSI | 0.5 | visual | 1 | 1240 | 0.00107505 | 0.0217378 | 0.206468 | 0.0206628 | 0.18473 | 0.999999 | 0.999999 | 0.9586 | 0.999544 | 0.978288 | 1.01536 | 0.0207792 | 0.0706487 |
| CMUMOSI | 0.5 | visual | 2-3 | 744 | 0.212902 | 0.218517 | 0.271704 | 0.00561497 | 0.0531869 | 0.957771 | 0.957771 | 0.944231 | 1.00852 | 0.987067 | 1.00282 | 0.00662953 | 0.00340536 |
| CMUMOSI | 0.5 | visual | 4-7 | 216 | 0.336959 | 0.339662 | 0.3263 | 0.00270287 | -0.0133616 | 0.923002 | 0.923002 | 0.92543 | 0.988247 | 0.967222 | 0.96254 | 0.00352062 | -0.000742763 |
| CMUMOSI | 0.5 | visual | 8+ | 24 | 0.449154 | 0.450591 | 0.511554 | 0.00143654 | 0.0609636 | 0.876416 | 0.876416 | 0.847304 | 0.958526 | 0.938137 | 0.967306 | 0.0028469 | 0.0029657 |
