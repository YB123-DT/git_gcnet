# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | audio | 1 | 800 | 0.00179765 | 0.0206975 | 0.341513 | 0.0188998 | 0.320816 | 0.999999 | 0.999999 | 0.921252 | 0.999087 | 0.97936 | 0.992281 | 0.01899 | 0.258375 |
| IEMOCAPFour | 0.1 | audio | 2-3 | 72 | 0.389332 | 0.386358 | 0.49442 | -0.00297417 | 0.108062 | 0.906374 | 0.906374 | 0.854273 | 1.02171 | 1.00153 | 0.985845 | -0.0021995 | 0.0570497 |
| IEMOCAPFour | 0.1 | text | 1 | 944 | 0.00241767 | 0.0213481 | 0.309681 | 0.0189304 | 0.288333 | 0.999998 | 0.999998 | 0.927931 | 0.998444 | 0.97873 | 0.944358 | 0.0190434 | 0.233861 |
| IEMOCAPFour | 0.1 | text | 2-3 | 56 | 0.38632 | 0.386861 | 0.468861 | 0.000540864 | 0.0820006 | 0.894693 | 0.894693 | 0.865422 | 0.921683 | 0.903495 | 0.932168 | 0.00157404 | 0.0118176 |
| IEMOCAPFour | 0.1 | visual | 1 | 992 | 0.00160285 | 0.0205693 | 0.293877 | 0.0189664 | 0.273308 | 0.999999 | 0.999999 | 0.938624 | 0.999207 | 0.979478 | 0.986582 | 0.0190312 | 0.22058 |
| IEMOCAPFour | 0.1 | visual | 2-3 | 152 | 0.276493 | 0.277662 | 0.354832 | 0.00116942 | 0.0771705 | 0.946979 | 0.946979 | 0.921832 | 0.984668 | 0.965225 | 0.967664 | 0.00112984 | 0.0415762 |
