# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | audio | 1 | 1184 | 0.000563516 | 0.0214392 | 0.136475 | 0.0208757 | 0.115036 | 1 | 1 | 0.981208 | 0.99969 | 0.978567 | 0.981555 | 0.0209047 | 0.0594052 |
| CMUMOSI | 0.7 | audio | 2-3 | 1112 | 0.143555 | 0.152269 | 0.1856 | 0.00871343 | 0.0333313 | 0.98102 | 0.98102 | 0.972857 | 0.972131 | 0.95159 | 0.955912 | 0.00992431 | 0.00651138 |
| CMUMOSI | 0.7 | audio | 4-7 | 424 | 0.235221 | 0.243068 | 0.240572 | 0.00784741 | -0.00249596 | 0.961595 | 0.961595 | 0.9624 | 0.932552 | 0.912846 | 0.913596 | 0.00898593 | -0.000163529 |
| CMUMOSI | 0.7 | audio | 8+ | 56 | 0.33411 | 0.341116 | 0.365069 | 0.00700584 | 0.0239529 | 0.935901 | 0.935901 | 0.925577 | 0.874445 | 0.855972 | 0.865651 | 0.00873886 | 0.00559393 |
| CMUMOSI | 0.7 | text | 1 | 1256 | 0.00186223 | 0.0224704 | 0.0912562 | 0.0206081 | 0.0687858 | 0.999999 | 0.999999 | 0.991971 | 0.998665 | 0.977564 | 0.950235 | 0.0206905 | 0.0237194 |
| CMUMOSI | 0.7 | text | 2-3 | 1024 | 0.110613 | 0.124107 | 0.140501 | 0.0134938 | 0.016394 | 0.987896 | 0.987896 | 0.984856 | 0.941642 | 0.921744 | 0.916036 | 0.0151418 | 0.00178087 |
| CMUMOSI | 0.7 | text | 4-7 | 584 | 0.184644 | 0.198218 | 0.202368 | 0.0135739 | 0.00414994 | 0.978385 | 0.978385 | 0.977307 | 0.878999 | 0.860425 | 0.86061 | 0.0157549 | 0.000664998 |
| CMUMOSI | 0.7 | text | 8+ | 136 | 0.251554 | 0.264427 | 0.265617 | 0.0128723 | 0.00119068 | 0.969992 | 0.969992 | 0.969864 | 0.819361 | 0.802048 | 0.801794 | 0.0150123 | 0.000398323 |
| CMUMOSI | 0.7 | visual | 1 | 1264 | 0.000950298 | 0.0215651 | 0.16926 | 0.0206148 | 0.147695 | 1 | 1 | 0.969758 | 0.999575 | 0.978454 | 1.00622 | 0.0207556 | 0.0556935 |
| CMUMOSI | 0.7 | visual | 2-3 | 1080 | 0.19228 | 0.198702 | 0.231893 | 0.00642253 | 0.033191 | 0.963997 | 0.963997 | 0.954906 | 1.00197 | 0.980794 | 0.988574 | 0.00843514 | 0.00250077 |
| CMUMOSI | 0.7 | visual | 4-7 | 376 | 0.282693 | 0.288382 | 0.333319 | 0.00568842 | 0.0449368 | 0.941454 | 0.941454 | 0.926371 | 0.972015 | 0.951471 | 0.97331 | 0.0083385 | 0.00436191 |
