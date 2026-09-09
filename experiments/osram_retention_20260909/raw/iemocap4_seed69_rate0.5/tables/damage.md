# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | audio | 1 | 2352 | 0.00160846 | 0.0206711 | 0.271402 | 0.0190626 | 0.250731 | 0.999999 | 0.999999 | 0.936759 | 0.999237 | 0.979378 | 0.980432 | 0.0191893 | 0.169676 |
| IEMOCAPFour | 0.5 | audio | 2-3 | 1616 | 0.302013 | 0.304275 | 0.3904 | 0.00226233 | 0.0861248 | 0.925593 | 0.925593 | 0.895073 | 0.978338 | 0.958894 | 0.960298 | 0.00262296 | 0.0263837 |
| IEMOCAPFour | 0.5 | audio | 4-7 | 328 | 0.479599 | 0.480244 | 0.515957 | 0.00064412 | 0.0357139 | 0.857227 | 0.857227 | 0.837902 | 0.925551 | 0.907151 | 0.905 | 0.00169709 | 0.00883669 |
| IEMOCAPFour | 0.5 | audio | 8+ | 16 | 0.565333 | 0.566454 | 0.553258 | 0.0011213 | -0.0131955 | 0.819375 | 0.819375 | 0.834875 | 0.821493 | 0.805165 | 0.820591 | 0.00465458 | -0.0058606 |
| IEMOCAPFour | 0.5 | text | 1 | 2520 | 0.00242897 | 0.0214775 | 0.280755 | 0.0190485 | 0.259277 | 0.999998 | 0.999998 | 0.932075 | 0.998446 | 0.978603 | 0.953595 | 0.0191843 | 0.172961 |
| IEMOCAPFour | 0.5 | text | 2-3 | 1616 | 0.310481 | 0.314172 | 0.411715 | 0.00369021 | 0.0975434 | 0.918095 | 0.918095 | 0.879143 | 0.945027 | 0.926247 | 0.922061 | 0.00469739 | 0.0327724 |
| IEMOCAPFour | 0.5 | text | 4-7 | 392 | 0.460906 | 0.462542 | 0.494803 | 0.00163559 | 0.0322608 | 0.858156 | 0.858156 | 0.840436 | 0.89226 | 0.87453 | 0.877767 | 0.00286757 | 0.0098746 |
| IEMOCAPFour | 0.5 | text | 8+ | 8 | 0.640255 | 0.635846 | 0.557453 | -0.00440953 | -0.0783929 | 0.746668 | 0.746668 | 0.771772 | 0.964161 | 0.945 | 0.847004 | -0.00537473 | -0.104742 |
| IEMOCAPFour | 0.5 | visual | 1 | 2288 | 0.00153997 | 0.0207974 | 0.250015 | 0.0192574 | 0.229218 | 0.999999 | 0.999999 | 0.944665 | 0.999094 | 0.979238 | 0.965685 | 0.0193515 | 0.155297 |
| IEMOCAPFour | 0.5 | visual | 2-3 | 1544 | 0.275719 | 0.278719 | 0.3654 | 0.00300001 | 0.0866808 | 0.937927 | 0.937927 | 0.906137 | 0.951981 | 0.93306 | 0.929906 | 0.00343797 | 0.0306586 |
| IEMOCAPFour | 0.5 | visual | 4-7 | 352 | 0.407063 | 0.408806 | 0.43671 | 0.00174264 | 0.0279036 | 0.889899 | 0.889899 | 0.879835 | 0.915832 | 0.897631 | 0.898724 | 0.00316456 | 0.00347811 |
| IEMOCAPFour | 0.5 | visual | 8+ | 16 | 0.560584 | 0.56177 | 0.546047 | 0.00118627 | -0.0157225 | 0.808612 | 0.808612 | 0.808805 | 0.87513 | 0.85775 | 0.856514 | 0.00223708 | -0.00712229 |
