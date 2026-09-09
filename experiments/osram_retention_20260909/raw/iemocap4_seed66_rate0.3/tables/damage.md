# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | audio | 1 | 2096 | 0.00170213 | 0.0207824 | 0.312314 | 0.0190803 | 0.291531 | 0.999999 | 0.999999 | 0.922461 | 0.999166 | 0.979271 | 0.975918 | 0.0192142 | 0.207402 |
| IEMOCAPFour | 0.3 | audio | 2-3 | 776 | 0.3352 | 0.336756 | 0.432592 | 0.00155609 | 0.0958359 | 0.914152 | 0.914152 | 0.872507 | 0.972968 | 0.953593 | 0.951136 | 0.00230934 | 0.0312902 |
| IEMOCAPFour | 0.3 | audio | 4-7 | 32 | 0.586385 | 0.584231 | 0.625464 | -0.00215365 | 0.0412329 | 0.80397 | 0.80397 | 0.780599 | 0.92648 | 0.908042 | 0.919562 | -0.00199257 | 0.015178 |
| IEMOCAPFour | 0.3 | text | 1 | 2120 | 0.00238126 | 0.0214222 | 0.285013 | 0.0190409 | 0.263591 | 0.999998 | 0.999998 | 0.937022 | 0.99854 | 0.978657 | 0.955474 | 0.0192005 | 0.191519 |
| IEMOCAPFour | 0.3 | text | 2-3 | 712 | 0.317176 | 0.319482 | 0.399045 | 0.0023053 | 0.0795633 | 0.922875 | 0.922875 | 0.89414 | 0.948421 | 0.929533 | 0.936861 | 0.00286533 | 0.0257058 |
| IEMOCAPFour | 0.3 | text | 4-7 | 40 | 0.308567 | 0.30971 | 0.384452 | 0.00114286 | 0.0747417 | 0.942387 | 0.942387 | 0.908122 | 0.944519 | 0.925719 | 0.943817 | 0.00151235 | 0.048313 |
| IEMOCAPFour | 0.3 | visual | 1 | 2056 | 0.00175679 | 0.0208678 | 0.296137 | 0.019111 | 0.275269 | 0.999999 | 0.999999 | 0.931725 | 0.999082 | 0.979188 | 0.982836 | 0.0192116 | 0.197088 |
| IEMOCAPFour | 0.3 | visual | 2-3 | 640 | 0.321391 | 0.322335 | 0.409178 | 0.000943821 | 0.0868435 | 0.917105 | 0.917105 | 0.886279 | 0.977812 | 0.958342 | 0.961853 | 0.00142227 | 0.0335015 |
| IEMOCAPFour | 0.3 | visual | 4-7 | 32 | 0.575027 | 0.573019 | 0.646202 | -0.00200727 | 0.0731829 | 0.806938 | 0.806938 | 0.774314 | 0.922283 | 0.903918 | 0.942397 | -0.00137147 | 0.0297495 |
