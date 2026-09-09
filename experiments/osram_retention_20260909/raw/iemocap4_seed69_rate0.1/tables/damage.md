# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 1 | 784 | 0.00151 | 0.0205189 | 0.313909 | 0.0190089 | 0.29339 | 0.999999 | 0.999999 | 0.92647 | 0.999391 | 0.979529 | 0.988264 | 0.0191262 | 0.227943 |
| IEMOCAPFour | 0.1 | audio | 2-3 | 80 | 0.326578 | 0.326572 | 0.387627 | -5.70707e-06 | 0.0610546 | 0.917401 | 0.917401 | 0.896351 | 0.979143 | 0.959683 | 0.951878 | 0.00124159 | 0.0307427 |
| IEMOCAPFour | 0.1 | text | 1 | 864 | 0.00255506 | 0.0215246 | 0.281459 | 0.0189696 | 0.259934 | 0.999998 | 0.999998 | 0.938219 | 0.998411 | 0.978568 | 0.959746 | 0.0191434 | 0.182181 |
| IEMOCAPFour | 0.1 | text | 2-3 | 80 | 0.359731 | 0.359927 | 0.422242 | 0.00019635 | 0.0623148 | 0.915211 | 0.915211 | 0.891038 | 0.981854 | 0.962346 | 0.967138 | 0.00283927 | 0.0323688 |
| IEMOCAPFour | 0.1 | visual | 1 | 960 | 0.0015845 | 0.0206828 | 0.315882 | 0.0190983 | 0.295199 | 0.999999 | 0.999999 | 0.919708 | 0.999221 | 0.979362 | 0.970119 | 0.0192304 | 0.222732 |
| IEMOCAPFour | 0.1 | visual | 2-3 | 136 | 0.330209 | 0.331086 | 0.417633 | 0.000877028 | 0.0865466 | 0.911125 | 0.911125 | 0.884102 | 0.967287 | 0.948058 | 0.940096 | 0.00214316 | 0.0415784 |
| IEMOCAPFour | 0.1 | visual | 4-7 | 8 | 0.495494 | 0.495463 | 0.537505 | -3.13502e-05 | 0.0420421 | 0.845871 | 0.845871 | 0.842985 | 0.928553 | 0.910075 | 0.917118 | 0.00134793 | 0.0519529 |
