# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | audio | 1 | 2048 | 0.0015906 | 0.0206319 | 0.287926 | 0.0190413 | 0.267294 | 0.999999 | 0.999999 | 0.936765 | 0.999278 | 0.979418 | 0.988984 | 0.0191452 | 0.192441 |
| IEMOCAPFour | 0.3 | audio | 2-3 | 736 | 0.306526 | 0.306994 | 0.401561 | 0.000468421 | 0.0945666 | 0.931791 | 0.931791 | 0.895942 | 0.989554 | 0.969888 | 0.964578 | 0.00119692 | 0.0363481 |
| IEMOCAPFour | 0.3 | audio | 4-7 | 48 | 0.469459 | 0.469904 | 0.508742 | 0.000444445 | 0.0388385 | 0.871704 | 0.871704 | 0.849588 | 0.906964 | 0.888955 | 0.90184 | 0.00174549 | 0.0109884 |
| IEMOCAPFour | 0.3 | text | 1 | 1960 | 0.00246749 | 0.021451 | 0.295172 | 0.0189835 | 0.273721 | 0.999998 | 0.999998 | 0.928498 | 0.998479 | 0.978635 | 0.96133 | 0.0191421 | 0.186857 |
| IEMOCAPFour | 0.3 | text | 2-3 | 752 | 0.330112 | 0.332056 | 0.415324 | 0.00194353 | 0.0832686 | 0.916557 | 0.916557 | 0.884623 | 0.961509 | 0.942402 | 0.937018 | 0.00281174 | 0.0280897 |
| IEMOCAPFour | 0.3 | text | 4-7 | 24 | 0.544465 | 0.543228 | 0.626808 | -0.00123705 | 0.0835795 | 0.781773 | 0.781773 | 0.748225 | 0.898595 | 0.880723 | 0.90367 | -2.59429e-05 | 0.0769555 |
| IEMOCAPFour | 0.3 | visual | 1 | 2008 | 0.00159154 | 0.020715 | 0.286531 | 0.0191234 | 0.265816 | 0.999999 | 0.999999 | 0.932884 | 0.999189 | 0.979331 | 0.968089 | 0.0192477 | 0.19577 |
| IEMOCAPFour | 0.3 | visual | 2-3 | 736 | 0.302181 | 0.3037 | 0.39611 | 0.00151956 | 0.0924092 | 0.929434 | 0.929434 | 0.896946 | 0.970548 | 0.951259 | 0.959567 | 0.00158654 | 0.0374815 |
| IEMOCAPFour | 0.3 | visual | 4-7 | 64 | 0.443607 | 0.443154 | 0.474122 | -0.000452708 | 0.0309677 | 0.875716 | 0.875716 | 0.860547 | 0.917798 | 0.899552 | 0.900512 | -0.000673413 | -0.00242382 |
