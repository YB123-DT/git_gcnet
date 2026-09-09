# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | audio | 1 | 1896 | 0.00176698 | 0.0206643 | 0.305028 | 0.0188973 | 0.284364 | 0.999999 | 0.999999 | 0.927621 | 0.99912 | 0.979393 | 0.98 | 0.0189862 | 0.212991 |
| IEMOCAPFour | 0.3 | audio | 2-3 | 696 | 0.325063 | 0.325109 | 0.435257 | 4.614e-05 | 0.110148 | 0.920027 | 0.920027 | 0.871587 | 0.987787 | 0.968284 | 0.964261 | 0.000458984 | 0.045962 |
| IEMOCAPFour | 0.3 | audio | 4-7 | 64 | 0.636728 | 0.632884 | 0.756577 | -0.00384402 | 0.123693 | 0.729838 | 0.729838 | 0.64017 | 0.944534 | 0.92589 | 0.92549 | -0.00309166 | 0.0698251 |
| IEMOCAPFour | 0.3 | text | 1 | 1984 | 0.00245123 | 0.0213725 | 0.273397 | 0.0189213 | 0.252024 | 0.999998 | 0.999998 | 0.938228 | 0.998421 | 0.978708 | 0.952328 | 0.0190266 | 0.179291 |
| IEMOCAPFour | 0.3 | text | 2-3 | 768 | 0.285789 | 0.289179 | 0.378476 | 0.00338971 | 0.0892969 | 0.934331 | 0.934331 | 0.89901 | 0.942167 | 0.923569 | 0.917721 | 0.00407057 | 0.0272525 |
| IEMOCAPFour | 0.3 | text | 4-7 | 64 | 0.40248 | 0.4019 | 0.43078 | -0.000579683 | 0.0288804 | 0.914477 | 0.914477 | 0.899852 | 0.975699 | 0.956442 | 0.945399 | 0.00112154 | 0.00473789 |
| IEMOCAPFour | 0.3 | visual | 1 | 1824 | 0.00156452 | 0.0206079 | 0.273754 | 0.0190434 | 0.253146 | 0.999999 | 0.999999 | 0.941473 | 0.999163 | 0.979435 | 0.979424 | 0.0190838 | 0.195183 |
| IEMOCAPFour | 0.3 | visual | 2-3 | 832 | 0.296187 | 0.297486 | 0.371331 | 0.00129975 | 0.0738443 | 0.930565 | 0.930565 | 0.906183 | 0.972567 | 0.953358 | 0.95249 | 0.00160972 | 0.0299332 |
| IEMOCAPFour | 0.3 | visual | 4-7 | 88 | 0.448769 | 0.448073 | 0.511361 | -0.000695879 | 0.0632881 | 0.881097 | 0.881097 | 0.847895 | 0.963058 | 0.944037 | 0.95072 | 0.000407055 | 0.0161771 |
