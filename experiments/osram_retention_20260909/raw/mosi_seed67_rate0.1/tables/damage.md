# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | audio | 1 | 496 | 0.00054346 | 0.0212791 | 0.160629 | 0.0207356 | 0.13935 | 1 | 1 | 0.975552 | 0.999857 | 0.978731 | 1.00322 | 0.0208351 | 0.07282 |
| CMUMOSI | 0.1 | audio | 2-3 | 64 | 0.176691 | 0.181949 | 0.217669 | 0.00525746 | 0.0357201 | 0.969803 | 0.969803 | 0.961262 | 1.00232 | 0.98115 | 0.992456 | 0.0047121 | 0.00456378 |
| CMUMOSI | 0.1 | text | 1 | 480 | 0.00154343 | 0.0221131 | 0.061993 | 0.0205697 | 0.0398799 | 0.999999 | 0.999999 | 0.997674 | 0.999027 | 0.977918 | 0.96809 | 0.0206194 | 0.0206503 |
| CMUMOSI | 0.1 | text | 2-3 | 72 | 0.0525219 | 0.0667946 | 0.0684353 | 0.0142727 | 0.00164067 | 0.998833 | 0.998833 | 0.998792 | 0.969372 | 0.948888 | 0.94807 | 0.0158118 | 8.32099e-05 |
| CMUMOSI | 0.1 | visual | 1 | 472 | 0.000971136 | 0.0214385 | 0.178411 | 0.0204674 | 0.156973 | 0.999999 | 0.999999 | 0.968345 | 0.999712 | 0.978588 | 1.02123 | 0.0206165 | 0.0676225 |
| CMUMOSI | 0.1 | visual | 2-3 | 40 | 0.186613 | 0.190616 | 0.221259 | 0.00400298 | 0.0306426 | 0.96234 | 0.96234 | 0.958869 | 1.01697 | 0.995478 | 1.00188 | 0.0045131 | 0.000337867 |
