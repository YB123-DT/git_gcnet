# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | audio | 1 | 1208 | 0.000622814 | 0.0216035 | 0.123565 | 0.0209807 | 0.101961 | 1 | 1 | 0.983827 | 0.99966 | 0.978402 | 0.976845 | 0.0210454 | 0.0486558 |
| CMUMOSI | 0.7 | audio | 2-3 | 1208 | 0.134919 | 0.143475 | 0.175773 | 0.0085559 | 0.0322979 | 0.983128 | 0.983128 | 0.975232 | 0.96858 | 0.947984 | 0.945638 | 0.0101876 | 0.00384711 |
| CMUMOSI | 0.7 | audio | 4-7 | 432 | 0.224004 | 0.231443 | 0.242285 | 0.00743878 | 0.010842 | 0.962617 | 0.962617 | 0.95571 | 0.91981 | 0.900252 | 0.894157 | 0.00877668 | 0.00134043 |
| CMUMOSI | 0.7 | audio | 8+ | 48 | 0.299196 | 0.310141 | 0.31906 | 0.0109445 | 0.00891988 | 0.95238 | 0.95238 | 0.947155 | 0.791854 | 0.775027 | 0.764535 | 0.0120206 | -0.00159007 |
| CMUMOSI | 0.7 | text | 1 | 1264 | 0.00168555 | 0.0224153 | 0.109444 | 0.0207298 | 0.0870287 | 0.999999 | 0.999999 | 0.990397 | 0.998856 | 0.977615 | 0.948659 | 0.0207449 | 0.0429952 |
| CMUMOSI | 0.7 | text | 2-3 | 1024 | 0.134237 | 0.145971 | 0.167469 | 0.0117339 | 0.0214983 | 0.986632 | 0.986632 | 0.983164 | 0.93721 | 0.917284 | 0.910349 | 0.0132554 | 0.00287748 |
| CMUMOSI | 0.7 | text | 4-7 | 320 | 0.199708 | 0.210878 | 0.216961 | 0.0111698 | 0.00608328 | 0.978581 | 0.978581 | 0.976539 | 0.8932 | 0.87421 | 0.87416 | 0.0127287 | 0.000373036 |
| CMUMOSI | 0.7 | text | 8+ | 40 | 0.226101 | 0.237501 | 0.23942 | 0.0113999 | 0.00191889 | 0.981854 | 0.981854 | 0.981502 | 0.859469 | 0.841196 | 0.841276 | 0.0136363 | 0.00112513 |
| CMUMOSI | 0.7 | visual | 1 | 1232 | 0.00103115 | 0.0217015 | 0.176647 | 0.0206703 | 0.154946 | 0.999999 | 0.999999 | 0.965722 | 0.999579 | 0.978323 | 1.00269 | 0.020772 | 0.0502944 |
| CMUMOSI | 0.7 | visual | 2-3 | 1200 | 0.218738 | 0.224645 | 0.280736 | 0.00590664 | 0.0560912 | 0.954536 | 0.954536 | 0.939315 | 1.00933 | 0.987861 | 1.00307 | 0.00773648 | 0.00452064 |
| CMUMOSI | 0.7 | visual | 4-7 | 608 | 0.310407 | 0.314637 | 0.332878 | 0.00422994 | 0.0182415 | 0.933518 | 0.933518 | 0.926809 | 0.977465 | 0.956674 | 0.969513 | 0.00568105 | 0.000513125 |
| CMUMOSI | 0.7 | visual | 8+ | 32 | 0.328673 | 0.333382 | 0.395075 | 0.00470848 | 0.0616932 | 0.936576 | 0.936576 | 0.907663 | 0.937302 | 0.917351 | 0.939527 | 0.00492087 | -0.000827715 |
