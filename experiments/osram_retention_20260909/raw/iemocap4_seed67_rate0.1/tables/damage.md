# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 1 | 792 | 0.00169093 | 0.0210166 | 0.31059 | 0.0193257 | 0.289573 | 0.999999 | 0.999999 | 0.929781 | 0.999208 | 0.979034 | 0.980869 | 0.0193221 | 0.236404 |
| IEMOCAPFour | 0.1 | audio | 2-3 | 56 | 0.309509 | 0.309282 | 0.387053 | -0.000227361 | 0.0777713 | 0.939317 | 0.939317 | 0.911065 | 1.01114 | 0.990727 | 0.990458 | -0.000495538 | 0.0868213 |
| IEMOCAPFour | 0.1 | text | 1 | 832 | 0.00252084 | 0.02177 | 0.295481 | 0.0192492 | 0.273711 | 0.999998 | 0.999998 | 0.927321 | 0.99848 | 0.978321 | 0.952142 | 0.0192927 | 0.191901 |
| IEMOCAPFour | 0.1 | text | 2-3 | 104 | 0.346324 | 0.347597 | 0.455688 | 0.00127261 | 0.108091 | 0.894377 | 0.894377 | 0.851318 | 0.949861 | 0.930678 | 0.947385 | 0.00115567 | 0.0486964 |
| IEMOCAPFour | 0.1 | visual | 1 | 784 | 0.00167228 | 0.0209441 | 0.323847 | 0.0192718 | 0.302903 | 0.999999 | 0.999999 | 0.922415 | 0.999289 | 0.979113 | 0.984603 | 0.0193156 | 0.230911 |
| IEMOCAPFour | 0.1 | visual | 2-3 | 72 | 0.35441 | 0.352808 | 0.408469 | -0.00160202 | 0.0556608 | 0.911153 | 0.911153 | 0.899882 | 1.01354 | 0.993054 | 0.99209 | -0.00145365 | 0.050524 |
