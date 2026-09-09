# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 1 | 2336 | 0.00167578 | 0.020687 | 0.266393 | 0.0190112 | 0.245706 | 0.999999 | 0.999999 | 0.93874 | 0.999086 | 0.97936 | 0.97402 | 0.0190676 | 0.172967 |
| IEMOCAPFour | 0.7 | audio | 2-3 | 2032 | 0.300231 | 0.302572 | 0.386893 | 0.00234149 | 0.0843208 | 0.926356 | 0.926356 | 0.89515 | 0.961242 | 0.942264 | 0.941415 | 0.00269088 | 0.0303713 |
| IEMOCAPFour | 0.7 | audio | 4-7 | 824 | 0.488741 | 0.488435 | 0.544178 | -0.000305422 | 0.0557427 | 0.841241 | 0.841241 | 0.806823 | 0.923203 | 0.904978 | 0.898901 | 0.000618607 | 0.0165079 |
| IEMOCAPFour | 0.7 | audio | 8+ | 56 | 0.604207 | 0.602725 | 0.645006 | -0.00148191 | 0.0422806 | 0.788385 | 0.788385 | 0.756522 | 0.894953 | 0.87727 | 0.909698 | 0.000294104 | 0.0183902 |
| IEMOCAPFour | 0.7 | text | 1 | 2448 | 0.00227724 | 0.021311 | 0.23094 | 0.0190337 | 0.209629 | 0.999999 | 0.999999 | 0.951834 | 0.998466 | 0.978752 | 0.943661 | 0.0191112 | 0.141794 |
| IEMOCAPFour | 0.7 | text | 2-3 | 2128 | 0.270092 | 0.274625 | 0.350618 | 0.00453287 | 0.0759935 | 0.935831 | 0.935831 | 0.908725 | 0.934951 | 0.916489 | 0.906742 | 0.0050586 | 0.0280051 |
| IEMOCAPFour | 0.7 | text | 4-7 | 856 | 0.440163 | 0.442437 | 0.483016 | 0.00227403 | 0.0405798 | 0.867594 | 0.867594 | 0.846883 | 0.872663 | 0.855433 | 0.854868 | 0.0033007 | 0.00983 |
| IEMOCAPFour | 0.7 | text | 8+ | 104 | 0.556561 | 0.557907 | 0.554549 | 0.00134619 | -0.00335807 | 0.795641 | 0.795641 | 0.794681 | 0.807244 | 0.791297 | 0.786314 | 0.00137725 | -0.00238681 |
| IEMOCAPFour | 0.7 | visual | 1 | 2328 | 0.00154702 | 0.0206865 | 0.230269 | 0.0191395 | 0.209583 | 0.999999 | 0.999999 | 0.95296 | 0.999077 | 0.97935 | 0.966629 | 0.019171 | 0.142138 |
| IEMOCAPFour | 0.7 | visual | 2-3 | 2032 | 0.25998 | 0.263654 | 0.342785 | 0.00367396 | 0.0791308 | 0.945111 | 0.945111 | 0.91716 | 0.952304 | 0.933497 | 0.93217 | 0.00390954 | 0.0303106 |
| IEMOCAPFour | 0.7 | visual | 4-7 | 984 | 0.403062 | 0.405093 | 0.440687 | 0.00203068 | 0.035594 | 0.890941 | 0.890941 | 0.876963 | 0.903859 | 0.88601 | 0.889896 | 0.00272965 | 0.00988573 |
| IEMOCAPFour | 0.7 | visual | 8+ | 120 | 0.565237 | 0.566085 | 0.59227 | 0.000847773 | 0.0261853 | 0.815663 | 0.815663 | 0.791881 | 0.818036 | 0.801892 | 0.806584 | 0.00182247 | 0.00127891 |
