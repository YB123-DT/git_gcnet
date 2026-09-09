# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | audio | 1 | 1000 | 0.000582211 | 0.021605 | 0.163248 | 0.0210228 | 0.141643 | 1 | 1 | 0.977214 | 0.99981 | 0.978402 | 0.991902 | 0.0211058 | 0.0889336 |
| CMUMOSI | 0.3 | audio | 2-3 | 352 | 0.184893 | 0.188808 | 0.233024 | 0.00391475 | 0.044216 | 0.972219 | 0.972219 | 0.961511 | 0.98895 | 0.967773 | 0.983408 | 0.00400913 | 0.0126084 |
| CMUMOSI | 0.3 | audio | 4-7 | 56 | 0.260179 | 0.265043 | 0.263008 | 0.00486451 | -0.00203511 | 0.956471 | 0.956471 | 0.957084 | 0.970185 | 0.949408 | 0.947603 | 0.00785431 | -0.0015244 |
| CMUMOSI | 0.3 | text | 1 | 1048 | 0.00166379 | 0.0224401 | 0.111923 | 0.0207763 | 0.0894832 | 0.999999 | 0.999999 | 0.988209 | 0.998992 | 0.977601 | 0.955872 | 0.0208621 | 0.0438236 |
| CMUMOSI | 0.3 | text | 2-3 | 376 | 0.129937 | 0.139775 | 0.165601 | 0.00983862 | 0.0258253 | 0.986121 | 0.986121 | 0.980998 | 0.964674 | 0.944017 | 0.934283 | 0.0129151 | 0.00581097 |
| CMUMOSI | 0.3 | text | 4-7 | 40 | 0.195525 | 0.204257 | 0.195976 | 0.00873213 | -0.00828128 | 0.980745 | 0.980745 | 0.982954 | 0.929913 | 0.910001 | 0.91336 | 0.0106476 | -0.00173244 |
| CMUMOSI | 0.3 | visual | 1 | 1064 | 0.00107679 | 0.0218719 | 0.181364 | 0.0207951 | 0.159492 | 0.999999 | 0.999999 | 0.966352 | 0.999557 | 0.978154 | 1.00663 | 0.0208562 | 0.0511279 |
| CMUMOSI | 0.3 | visual | 2-3 | 424 | 0.199754 | 0.205846 | 0.211992 | 0.00609149 | 0.00614605 | 0.961543 | 0.961543 | 0.965009 | 1.013 | 0.991308 | 0.980354 | 0.00962516 | 0.00285665 |
| CMUMOSI | 0.3 | visual | 4-7 | 40 | 0.161426 | 0.17207 | 0.170473 | 0.0106442 | -0.00159645 | 0.986939 | 0.986939 | 0.987834 | 0.925286 | 0.905473 | 0.901571 | 0.0132247 | 0.00192002 |
