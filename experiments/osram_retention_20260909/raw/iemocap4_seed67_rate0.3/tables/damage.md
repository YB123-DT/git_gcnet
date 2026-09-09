# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | audio | 1 | 1992 | 0.0017651 | 0.0211037 | 0.284221 | 0.0193386 | 0.263118 | 0.999999 | 0.999999 | 0.94043 | 0.999122 | 0.97895 | 0.983627 | 0.019357 | 0.202496 |
| IEMOCAPFour | 0.3 | audio | 2-3 | 584 | 0.299896 | 0.301173 | 0.369961 | 0.00127626 | 0.0687882 | 0.934428 | 0.934428 | 0.909032 | 0.973427 | 0.95377 | 0.95179 | 0.00172645 | 0.0280105 |
| IEMOCAPFour | 0.3 | audio | 4-7 | 32 | 0.521304 | 0.518494 | 0.574427 | -0.00280918 | 0.0559331 | 0.838 | 0.838 | 0.814228 | 0.997321 | 0.9772 | 0.999463 | -0.0014625 | 0.00498382 |
| IEMOCAPFour | 0.3 | text | 1 | 1872 | 0.00248954 | 0.0218052 | 0.284912 | 0.0193157 | 0.263107 | 0.999998 | 0.999998 | 0.932927 | 0.998438 | 0.97828 | 0.943608 | 0.0193244 | 0.191324 |
| IEMOCAPFour | 0.3 | text | 2-3 | 752 | 0.322204 | 0.324987 | 0.399028 | 0.00278302 | 0.074041 | 0.917067 | 0.917067 | 0.888261 | 0.944526 | 0.92546 | 0.919528 | 0.00307392 | 0.0252468 |
| IEMOCAPFour | 0.3 | text | 4-7 | 72 | 0.480974 | 0.482708 | 0.554484 | 0.00173391 | 0.0717762 | 0.849926 | 0.849926 | 0.801728 | 0.871345 | 0.853754 | 0.865678 | 0.00276889 | 0.0174205 |
| IEMOCAPFour | 0.3 | visual | 1 | 2048 | 0.00169299 | 0.0210695 | 0.287047 | 0.0193765 | 0.265977 | 0.999999 | 0.999999 | 0.93615 | 0.999156 | 0.978983 | 0.982441 | 0.0193819 | 0.201763 |
| IEMOCAPFour | 0.3 | visual | 2-3 | 792 | 0.302148 | 0.303072 | 0.413559 | 0.000924741 | 0.110487 | 0.930934 | 0.930934 | 0.890129 | 0.976096 | 0.956391 | 0.968259 | 0.00092724 | 0.0540748 |
| IEMOCAPFour | 0.3 | visual | 4-7 | 40 | 0.426727 | 0.425773 | 0.43584 | -0.000953931 | 0.0100678 | 0.887191 | 0.887191 | 0.88045 | 0.982702 | 0.962854 | 0.945368 | -0.0011797 | -0.00270011 |
