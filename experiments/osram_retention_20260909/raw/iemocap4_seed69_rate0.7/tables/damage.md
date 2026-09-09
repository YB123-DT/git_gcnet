# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | audio | 1 | 2224 | 0.00158007 | 0.0207194 | 0.265383 | 0.0191394 | 0.244664 | 0.999999 | 0.999999 | 0.938783 | 0.999182 | 0.979324 | 0.974014 | 0.0192457 | 0.159501 |
| IEMOCAPFour | 0.7 | audio | 2-3 | 2096 | 0.303119 | 0.30599 | 0.39688 | 0.00287095 | 0.0908898 | 0.924279 | 0.924279 | 0.889915 | 0.96599 | 0.946791 | 0.948295 | 0.00335518 | 0.0277411 |
| IEMOCAPFour | 0.7 | audio | 4-7 | 1096 | 0.47174 | 0.472624 | 0.504105 | 0.000884674 | 0.031481 | 0.854108 | 0.854108 | 0.837114 | 0.910998 | 0.89289 | 0.89061 | 0.00233187 | 0.00726599 |
| IEMOCAPFour | 0.7 | audio | 8+ | 128 | 0.567429 | 0.568689 | 0.574903 | 0.00126004 | 0.00621436 | 0.802865 | 0.802865 | 0.801458 | 0.848816 | 0.831941 | 0.827102 | 0.00296693 | 0.00339299 |
| IEMOCAPFour | 0.7 | text | 1 | 2344 | 0.00227451 | 0.0214507 | 0.259369 | 0.0191761 | 0.237918 | 0.999999 | 0.999999 | 0.935285 | 0.998456 | 0.978612 | 0.938662 | 0.0192777 | 0.144245 |
| IEMOCAPFour | 0.7 | text | 2-3 | 2224 | 0.297963 | 0.302334 | 0.39236 | 0.00437048 | 0.0900261 | 0.921678 | 0.921678 | 0.885867 | 0.928049 | 0.909609 | 0.89602 | 0.00511087 | 0.0280183 |
| IEMOCAPFour | 0.7 | text | 4-7 | 856 | 0.483867 | 0.486719 | 0.536527 | 0.00285166 | 0.0498078 | 0.842587 | 0.842587 | 0.812556 | 0.839728 | 0.823046 | 0.825978 | 0.0040935 | 0.00767964 |
| IEMOCAPFour | 0.7 | text | 8+ | 80 | 0.53842 | 0.542318 | 0.548691 | 0.00389737 | 0.00637343 | 0.809688 | 0.809688 | 0.800859 | 0.7596 | 0.744513 | 0.736261 | 0.0047594 | 0.00108737 |
| IEMOCAPFour | 0.7 | visual | 1 | 2400 | 0.00150807 | 0.0207981 | 0.236149 | 0.01929 | 0.215351 | 0.999999 | 0.999999 | 0.949651 | 0.999092 | 0.979236 | 0.964353 | 0.0193617 | 0.144407 |
| IEMOCAPFour | 0.7 | visual | 2-3 | 2008 | 0.274065 | 0.277112 | 0.357922 | 0.00304761 | 0.0808101 | 0.936226 | 0.936226 | 0.905735 | 0.953147 | 0.934203 | 0.927311 | 0.00339271 | 0.0256326 |
| IEMOCAPFour | 0.7 | visual | 4-7 | 888 | 0.434043 | 0.435328 | 0.469701 | 0.00128511 | 0.0343734 | 0.868776 | 0.868776 | 0.851261 | 0.903393 | 0.885436 | 0.885545 | 0.00209284 | 0.00639753 |
| IEMOCAPFour | 0.7 | visual | 8+ | 64 | 0.763681 | 0.761089 | 0.750822 | -0.00259262 | -0.0102662 | 0.625125 | 0.625125 | 0.640985 | 0.806586 | 0.790548 | 0.782695 | -0.00215867 | -0.00388721 |
