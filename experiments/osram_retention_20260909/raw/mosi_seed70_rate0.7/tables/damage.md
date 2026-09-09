# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | audio | 1 | 1216 | 0.000770777 | 0.0216506 | 0.203166 | 0.0208798 | 0.181516 | 1 | 1 | 0.960185 | 0.999568 | 0.978363 | 0.98945 | 0.0208974 | 0.089516 |
| CMUMOSI | 0.7 | audio | 2-3 | 1176 | 0.213335 | 0.21954 | 0.288437 | 0.00620568 | 0.0688971 | 0.957257 | 0.957257 | 0.935521 | 0.979081 | 0.958307 | 0.967064 | 0.00701588 | 0.00970066 |
| CMUMOSI | 0.7 | audio | 4-7 | 416 | 0.399715 | 0.40136 | 0.44316 | 0.00164517 | 0.0418006 | 0.893895 | 0.893895 | 0.876603 | 0.964432 | 0.943964 | 0.952399 | 0.0024446 | 0.00310364 |
| CMUMOSI | 0.7 | audio | 8+ | 56 | 0.655904 | 0.656125 | 0.640781 | 0.000220644 | -0.0153433 | 0.768113 | 0.768113 | 0.772924 | 0.889595 | 0.870722 | 0.852024 | 0.00330162 | -0.00202432 |
| CMUMOSI | 0.7 | text | 1 | 1336 | 0.00164219 | 0.0223362 | 0.188736 | 0.020694 | 0.1664 | 0.999999 | 0.999999 | 0.973822 | 0.998886 | 0.977695 | 0.923619 | 0.0207159 | 0.123211 |
| CMUMOSI | 0.7 | text | 2-3 | 1288 | 0.213361 | 0.22201 | 0.278357 | 0.00864957 | 0.0563468 | 0.969277 | 0.969277 | 0.951237 | 0.911278 | 0.891953 | 0.879974 | 0.0100773 | 0.0128104 |
| CMUMOSI | 0.7 | text | 4-7 | 416 | 0.288418 | 0.296554 | 0.316706 | 0.00813587 | 0.0201519 | 0.952183 | 0.952183 | 0.944721 | 0.868591 | 0.850172 | 0.850009 | 0.00968006 | 0.00378184 |
| CMUMOSI | 0.7 | text | 8+ | 8 | 0.285498 | 0.294216 | 0.298167 | 0.00871785 | 0.00395161 | 0.968239 | 0.968239 | 0.965806 | 0.821899 | 0.804459 | 0.805416 | 0.0107644 | 0.00564118 |
| CMUMOSI | 0.7 | visual | 1 | 1328 | 0.000936378 | 0.0216983 | 0.224275 | 0.0207619 | 0.202577 | 1 | 1 | 0.956125 | 0.999523 | 0.978318 | 0.978723 | 0.0208315 | 0.124819 |
| CMUMOSI | 0.7 | visual | 2-3 | 1152 | 0.238633 | 0.245539 | 0.309121 | 0.00690653 | 0.0635818 | 0.950137 | 0.950137 | 0.93144 | 0.961916 | 0.941503 | 0.944676 | 0.00820723 | 0.00521173 |
| CMUMOSI | 0.7 | visual | 4-7 | 344 | 0.359531 | 0.362723 | 0.365496 | 0.00319232 | 0.00277219 | 0.914626 | 0.914626 | 0.914177 | 0.924152 | 0.904536 | 0.900845 | 0.00400366 | 0.000939384 |
| CMUMOSI | 0.7 | visual | 8+ | 8 | 0.554349 | 0.554482 | 0.590915 | 0.000133622 | 0.0364324 | 0.842148 | 0.842148 | 0.826031 | 0.828536 | 0.81094 | 0.84669 | 0.00262034 | 0.0154487 |
