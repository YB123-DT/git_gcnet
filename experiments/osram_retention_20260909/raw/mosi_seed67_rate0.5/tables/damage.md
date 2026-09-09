# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | audio | 1 | 1344 | 0.00058628 | 0.0214245 | 0.133194 | 0.0208383 | 0.11177 | 1 | 1 | 0.982101 | 0.999706 | 0.978583 | 0.98599 | 0.0208884 | 0.0565038 |
| CMUMOSI | 0.5 | audio | 2-3 | 864 | 0.14658 | 0.154831 | 0.187842 | 0.00825067 | 0.0330116 | 0.979493 | 0.979493 | 0.971774 | 0.981363 | 0.960628 | 0.966685 | 0.00943965 | 0.00693632 |
| CMUMOSI | 0.5 | audio | 4-7 | 136 | 0.189391 | 0.199163 | 0.194562 | 0.00977195 | -0.00460009 | 0.973564 | 0.973564 | 0.977764 | 0.946563 | 0.926564 | 0.923403 | 0.0126977 | 0.000938635 |
| CMUMOSI | 0.5 | text | 1 | 1352 | 0.00187506 | 0.0224349 | 0.0768528 | 0.0205599 | 0.0544178 | 0.999999 | 0.999999 | 0.994831 | 0.998704 | 0.977602 | 0.959587 | 0.0206085 | 0.018699 |
| CMUMOSI | 0.5 | text | 2-3 | 880 | 0.0914434 | 0.106061 | 0.116792 | 0.0146173 | 0.0107316 | 0.993568 | 0.993568 | 0.992031 | 0.946102 | 0.92611 | 0.921552 | 0.0165302 | 0.00130081 |
| CMUMOSI | 0.5 | text | 4-7 | 120 | 0.164991 | 0.179016 | 0.183301 | 0.0140245 | 0.00428544 | 0.98638 | 0.98638 | 0.984997 | 0.889893 | 0.871092 | 0.872209 | 0.0156718 | 0.000912789 |
| CMUMOSI | 0.5 | visual | 1 | 1328 | 0.000976105 | 0.0215184 | 0.174217 | 0.0205423 | 0.152698 | 1 | 1 | 0.968205 | 0.999625 | 0.978504 | 1.00939 | 0.020698 | 0.0582888 |
| CMUMOSI | 0.5 | visual | 2-3 | 824 | 0.187552 | 0.193849 | 0.234318 | 0.00629657 | 0.0404694 | 0.965186 | 0.965186 | 0.9545 | 0.996436 | 0.975378 | 0.986462 | 0.00809967 | 0.00405632 |
| CMUMOSI | 0.5 | visual | 4-7 | 104 | 0.227866 | 0.236048 | 0.212238 | 0.00818196 | -0.0238103 | 0.961703 | 0.961703 | 0.972004 | 0.962041 | 0.941705 | 0.922131 | 0.0102302 | -0.000712331 |
