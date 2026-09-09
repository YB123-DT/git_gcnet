# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | audio | 1 | 2496 | 0.00172186 | 0.0206705 | 0.272827 | 0.0189487 | 0.252156 | 0.999999 | 0.999999 | 0.937226 | 0.999109 | 0.979382 | 0.973875 | 0.0190244 | 0.177925 |
| IEMOCAPFour | 0.5 | audio | 2-3 | 1584 | 0.300398 | 0.30264 | 0.40671 | 0.00224232 | 0.10407 | 0.927961 | 0.927961 | 0.884299 | 0.969748 | 0.9506 | 0.953945 | 0.00279773 | 0.0381975 |
| IEMOCAPFour | 0.5 | audio | 4-7 | 464 | 0.534094 | 0.532784 | 0.576965 | -0.00130972 | 0.0441803 | 0.821404 | 0.821404 | 0.797134 | 0.927885 | 0.909559 | 0.907435 | -0.000933036 | 0.00969383 |
| IEMOCAPFour | 0.5 | audio | 8+ | 32 | 0.684572 | 0.683066 | 0.748821 | -0.00150624 | 0.0657559 | 0.714807 | 0.714807 | 0.668924 | 0.820273 | 0.804073 | 0.815709 | -0.000504345 | 0.0184238 |
| IEMOCAPFour | 0.5 | text | 1 | 2280 | 0.00235098 | 0.0213557 | 0.244822 | 0.0190047 | 0.223467 | 0.999998 | 0.999998 | 0.946514 | 0.998427 | 0.978714 | 0.943976 | 0.0190818 | 0.150578 |
| IEMOCAPFour | 0.5 | text | 2-3 | 1656 | 0.271584 | 0.275801 | 0.364671 | 0.00421734 | 0.0888704 | 0.938025 | 0.938025 | 0.905614 | 0.94355 | 0.92492 | 0.913203 | 0.00486636 | 0.0321095 |
| IEMOCAPFour | 0.5 | text | 4-7 | 304 | 0.435516 | 0.43742 | 0.471275 | 0.00190432 | 0.0338547 | 0.865573 | 0.865573 | 0.85162 | 0.895525 | 0.877842 | 0.877572 | 0.00264174 | 0.00635859 |
| IEMOCAPFour | 0.5 | visual | 1 | 2496 | 0.00154487 | 0.0206614 | 0.245487 | 0.0191166 | 0.224826 | 0.999999 | 0.999999 | 0.948145 | 0.999103 | 0.979376 | 0.969009 | 0.0191591 | 0.155064 |
| IEMOCAPFour | 0.5 | visual | 2-3 | 1576 | 0.272141 | 0.275096 | 0.357256 | 0.00295487 | 0.0821601 | 0.938844 | 0.938844 | 0.909634 | 0.960281 | 0.941317 | 0.942583 | 0.00329132 | 0.0323275 |
| IEMOCAPFour | 0.5 | visual | 4-7 | 392 | 0.41723 | 0.418774 | 0.452557 | 0.00154344 | 0.0337835 | 0.884903 | 0.884903 | 0.871911 | 0.912256 | 0.894239 | 0.887403 | 0.00256639 | 0.00902098 |
| IEMOCAPFour | 0.5 | visual | 8+ | 32 | 0.490733 | 0.493863 | 0.542247 | 0.0031294 | 0.0483842 | 0.848285 | 0.848285 | 0.816502 | 0.794699 | 0.779014 | 0.796329 | 0.00248984 | 0.00762168 |
