# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | audio | 1 | 1912 | 0.00183132 | 0.0209357 | 0.289326 | 0.0191044 | 0.26839 | 0.999999 | 0.999999 | 0.939019 | 0.999067 | 0.979122 | 0.973711 | 0.0191968 | 0.221469 |
| IEMOCAPFour | 0.3 | audio | 2-3 | 712 | 0.320806 | 0.322144 | 0.411203 | 0.00133831 | 0.0890593 | 0.925681 | 0.925681 | 0.892736 | 0.963647 | 0.94441 | 0.949225 | 0.00192887 | 0.045477 |
| IEMOCAPFour | 0.3 | audio | 4-7 | 48 | 0.449108 | 0.446965 | 0.496672 | -0.00214305 | 0.0497066 | 0.878343 | 0.878343 | 0.857911 | 0.964476 | 0.945232 | 0.943866 | -0.00214514 | 0.0181231 |
| IEMOCAPFour | 0.3 | text | 1 | 1984 | 0.00247945 | 0.0215855 | 0.253924 | 0.019106 | 0.232339 | 0.999998 | 0.999998 | 0.948454 | 0.998431 | 0.978499 | 0.950801 | 0.0192269 | 0.172713 |
| IEMOCAPFour | 0.3 | text | 2-3 | 528 | 0.285178 | 0.288365 | 0.377685 | 0.00318714 | 0.08932 | 0.936961 | 0.936961 | 0.904814 | 0.946023 | 0.927137 | 0.924215 | 0.00393825 | 0.0424792 |
| IEMOCAPFour | 0.3 | text | 4-7 | 8 | 0.453817 | 0.457867 | 0.459474 | 0.00404984 | 0.00160674 | 0.880064 | 0.880064 | 0.882354 | 0.800876 | 0.784881 | 0.777338 | 0.00521874 | -0.0010211 |
| IEMOCAPFour | 0.3 | visual | 1 | 2000 | 0.00179932 | 0.0209648 | 0.303338 | 0.0191655 | 0.282374 | 0.999999 | 0.999999 | 0.93267 | 0.999037 | 0.979092 | 0.972988 | 0.0192874 | 0.221483 |
| IEMOCAPFour | 0.3 | visual | 2-3 | 656 | 0.344029 | 0.344588 | 0.455079 | 0.000558415 | 0.110492 | 0.91601 | 0.91601 | 0.872222 | 0.960153 | 0.940984 | 0.948047 | 0.00109237 | 0.0577279 |
| IEMOCAPFour | 0.3 | visual | 4-7 | 88 | 0.583648 | 0.582091 | 0.644656 | -0.00155619 | 0.0625643 | 0.798065 | 0.798065 | 0.758947 | 0.903739 | 0.885691 | 0.876783 | 0.000620484 | 0.0297397 |
