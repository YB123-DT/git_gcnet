# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | audio | 1 | 944 | 0.0005399 | 0.0214898 | 0.141701 | 0.0209499 | 0.120211 | 1 | 1 | 0.978438 | 0.999775 | 0.978515 | 0.979036 | 0.0210299 | 0.0623919 |
| CMUMOSI | 0.3 | audio | 2-3 | 272 | 0.167825 | 0.174294 | 0.195214 | 0.0064688 | 0.0209201 | 0.96879 | 0.96879 | 0.963696 | 0.973102 | 0.952416 | 0.947031 | 0.00711048 | 0.0051391 |
| CMUMOSI | 0.3 | audio | 4-7 | 8 | 0.300543 | 0.306435 | 0.354213 | 0.00589199 | 0.0477778 | 0.898 | 0.898 | 0.890531 | 0.892768 | 0.873797 | 0.882278 | 0.00674204 | 0.0102942 |
| CMUMOSI | 0.3 | text | 1 | 1088 | 0.00192989 | 0.0225377 | 0.0908636 | 0.0206078 | 0.0683259 | 0.999999 | 0.999999 | 0.993108 | 0.998746 | 0.977507 | 0.962866 | 0.0206608 | 0.0321972 |
| CMUMOSI | 0.3 | text | 2-3 | 440 | 0.105214 | 0.117936 | 0.130285 | 0.0127224 | 0.0123483 | 0.989937 | 0.989937 | 0.988735 | 0.952873 | 0.932613 | 0.930828 | 0.0148442 | 0.0028716 |
| CMUMOSI | 0.3 | text | 4-7 | 32 | 0.100405 | 0.114447 | 0.13376 | 0.0140417 | 0.0193135 | 0.996489 | 0.996489 | 0.992862 | 0.93561 | 0.915713 | 0.923335 | 0.014236 | 0.0104292 |
| CMUMOSI | 0.3 | visual | 1 | 1152 | 0.000981752 | 0.0216396 | 0.214547 | 0.0206579 | 0.192907 | 1 | 1 | 0.956177 | 0.99964 | 0.978383 | 1.02553 | 0.0207369 | 0.0738784 |
| CMUMOSI | 0.3 | visual | 2-3 | 320 | 0.193477 | 0.198514 | 0.245745 | 0.00503733 | 0.0472305 | 0.965358 | 0.965358 | 0.953069 | 1.01217 | 0.990636 | 1.00485 | 0.00647044 | 0.000990473 |
| CMUMOSI | 0.3 | visual | 4-7 | 8 | 0.146278 | 0.15684 | 0.197479 | 0.0105627 | 0.0406384 | 0.98815 | 0.98815 | 0.977993 | 0.935721 | 0.915825 | 0.955599 | 0.011012 | 0.0352874 |
