# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 1 | 936 | 0.00169353 | 0.0206918 | 0.335775 | 0.0189983 | 0.315083 | 0.999999 | 0.999999 | 0.914221 | 0.999263 | 0.979365 | 0.973526 | 0.0191488 | 0.228182 |
| IEMOCAPFour | 0.1 | audio | 2-3 | 88 | 0.292874 | 0.296039 | 0.450452 | 0.00316531 | 0.154413 | 0.927565 | 0.927565 | 0.876759 | 0.95194 | 0.93299 | 0.972304 | 0.00201216 | 0.0672232 |
| IEMOCAPFour | 0.1 | text | 1 | 944 | 0.00252305 | 0.0214616 | 0.305742 | 0.0189385 | 0.28428 | 0.999998 | 0.999998 | 0.935078 | 0.998519 | 0.978636 | 0.961321 | 0.0191161 | 0.221186 |
| IEMOCAPFour | 0.1 | text | 2-3 | 152 | 0.288513 | 0.289919 | 0.389074 | 0.0014059 | 0.099155 | 0.943997 | 0.943997 | 0.915846 | 0.973048 | 0.953671 | 0.935261 | 0.00211808 | 0.0521573 |
| IEMOCAPFour | 0.1 | visual | 1 | 776 | 0.00182637 | 0.0208513 | 0.355479 | 0.019025 | 0.334628 | 0.999999 | 0.999999 | 0.915034 | 0.999107 | 0.979212 | 0.995829 | 0.0191502 | 0.254454 |
| IEMOCAPFour | 0.1 | visual | 2-3 | 112 | 0.363936 | 0.362752 | 0.453665 | -0.0011847 | 0.0909133 | 0.915811 | 0.915811 | 0.882055 | 1.0223 | 1.00196 | 1.01551 | -5.21466e-05 | 0.0371802 |
