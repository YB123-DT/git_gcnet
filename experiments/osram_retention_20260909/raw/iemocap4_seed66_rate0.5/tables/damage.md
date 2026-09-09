# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | audio | 1 | 2504 | 0.00164462 | 0.0207553 | 0.274209 | 0.0191107 | 0.253453 | 0.999999 | 0.999999 | 0.937776 | 0.99919 | 0.979294 | 0.976996 | 0.0192515 | 0.168871 |
| IEMOCAPFour | 0.5 | audio | 2-3 | 1488 | 0.291712 | 0.294494 | 0.388674 | 0.00278201 | 0.0941791 | 0.931235 | 0.931235 | 0.898797 | 0.969098 | 0.949806 | 0.950724 | 0.00371787 | 0.0341859 |
| IEMOCAPFour | 0.5 | audio | 4-7 | 264 | 0.47771 | 0.477852 | 0.529328 | 0.000142217 | 0.0514753 | 0.859551 | 0.859551 | 0.833136 | 0.930076 | 0.911563 | 0.923636 | 0.00262821 | 0.00797561 |
| IEMOCAPFour | 0.5 | audio | 8+ | 8 | 0.457418 | 0.462986 | 0.50127 | 0.00556856 | 0.0382837 | 0.904585 | 0.904585 | 0.876006 | 0.732317 | 0.717753 | 0.722056 | 0.00554954 | 0.0392103 |
| IEMOCAPFour | 0.5 | text | 1 | 2456 | 0.00227579 | 0.0214181 | 0.274654 | 0.0191423 | 0.253236 | 0.999999 | 0.999999 | 0.938552 | 0.998532 | 0.978649 | 0.952424 | 0.0192652 | 0.175135 |
| IEMOCAPFour | 0.5 | text | 2-3 | 1520 | 0.306207 | 0.309399 | 0.41564 | 0.00319225 | 0.106241 | 0.925307 | 0.925307 | 0.881688 | 0.944988 | 0.926167 | 0.929041 | 0.00377287 | 0.0396245 |
| IEMOCAPFour | 0.5 | text | 4-7 | 368 | 0.466875 | 0.468505 | 0.497918 | 0.00162957 | 0.0294133 | 0.857727 | 0.857727 | 0.842672 | 0.873194 | 0.855807 | 0.860539 | 0.00240805 | 0.00355986 |
| IEMOCAPFour | 0.5 | text | 8+ | 24 | 0.525096 | 0.525727 | 0.508247 | 0.000630882 | -0.0174795 | 0.850937 | 0.850937 | 0.846513 | 0.84105 | 0.824313 | 0.793587 | 0.00206706 | -0.00164587 |
| IEMOCAPFour | 0.5 | visual | 1 | 2440 | 0.00170212 | 0.0208852 | 0.259502 | 0.0191831 | 0.238617 | 0.999999 | 0.999999 | 0.943788 | 0.999057 | 0.979164 | 0.982024 | 0.0192723 | 0.162219 |
| IEMOCAPFour | 0.5 | visual | 2-3 | 1544 | 0.294739 | 0.296938 | 0.384435 | 0.00219891 | 0.0874967 | 0.931067 | 0.931067 | 0.899975 | 0.976862 | 0.957414 | 0.95687 | 0.002914 | 0.0393561 |
| IEMOCAPFour | 0.5 | visual | 4-7 | 384 | 0.455524 | 0.455395 | 0.501355 | -0.000128593 | 0.0459602 | 0.866111 | 0.866111 | 0.844146 | 0.944435 | 0.925635 | 0.933718 | 0.000453159 | 0.0131583 |
| IEMOCAPFour | 0.5 | visual | 8+ | 40 | 0.651048 | 0.647891 | 0.627576 | -0.0031579 | -0.0203147 | 0.783356 | 0.783356 | 0.787377 | 0.980808 | 0.961309 | 0.931753 | -0.00152676 | -0.000195935 |
