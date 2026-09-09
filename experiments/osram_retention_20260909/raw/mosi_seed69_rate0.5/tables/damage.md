# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | audio | 1 | 1272 | 0.000687043 | 0.0217276 | 0.176004 | 0.0210406 | 0.154276 | 1 | 1 | 0.972267 | 0.999686 | 0.978281 | 0.993343 | 0.0211359 | 0.0917806 |
| CMUMOSI | 0.5 | audio | 2-3 | 952 | 0.177985 | 0.183173 | 0.222016 | 0.00518772 | 0.0388429 | 0.974618 | 0.974618 | 0.965667 | 0.98845 | 0.967283 | 0.970388 | 0.00562882 | 0.0112742 |
| CMUMOSI | 0.5 | audio | 4-7 | 288 | 0.285094 | 0.287804 | 0.318358 | 0.00271016 | 0.0305543 | 0.951797 | 0.951797 | 0.94139 | 0.957796 | 0.937284 | 0.944153 | 0.00262921 | 0.00503378 |
| CMUMOSI | 0.5 | text | 1 | 1336 | 0.00155887 | 0.0224121 | 0.132338 | 0.0208533 | 0.109926 | 0.999999 | 0.999999 | 0.983313 | 0.999013 | 0.977622 | 0.948591 | 0.0209098 | 0.0457253 |
| CMUMOSI | 0.5 | text | 2-3 | 896 | 0.148119 | 0.15857 | 0.190529 | 0.010451 | 0.031959 | 0.978116 | 0.978116 | 0.970578 | 0.948288 | 0.927983 | 0.918477 | 0.0130691 | 0.00427864 |
| CMUMOSI | 0.5 | text | 4-7 | 168 | 0.208199 | 0.218816 | 0.222823 | 0.0106165 | 0.00400742 | 0.970959 | 0.970959 | 0.968595 | 0.901856 | 0.882543 | 0.881443 | 0.0123742 | 0.00334685 |
| CMUMOSI | 0.5 | visual | 1 | 1384 | 0.00102294 | 0.021867 | 0.160766 | 0.0208441 | 0.138899 | 0.999999 | 0.999999 | 0.971235 | 0.999558 | 0.978156 | 0.990323 | 0.0209122 | 0.0506801 |
| CMUMOSI | 0.5 | visual | 2-3 | 904 | 0.184713 | 0.191699 | 0.239964 | 0.00698572 | 0.0482654 | 0.965699 | 0.965699 | 0.952729 | 0.987064 | 0.965928 | 0.969788 | 0.00835414 | 0.00765062 |
| CMUMOSI | 0.5 | visual | 4-7 | 200 | 0.301645 | 0.307674 | 0.320759 | 0.0060292 | 0.0130857 | 0.928851 | 0.928851 | 0.92329 | 0.937596 | 0.917521 | 0.929694 | 0.00800047 | -0.000183098 |
