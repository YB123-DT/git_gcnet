# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.3 | [0,.1) | 1413 | 0.0529647 | 0.04939 | 0.0242382 | 0.0148485 | 0.0138348 |
| IEMOCAPFour | 0.3 | [.1,.2) | 1751 | 0.150195 | 0.133167 | 0.0959517 | 0.0840797 | 0.0139032 |
| IEMOCAPFour | 0.3 | [.2,.4) | 3087 | 0.294912 | 0.246329 | 0.215155 | 0.196075 | 0.0133925 |
| IEMOCAPFour | 0.3 | [.4,.6) | 1524 | 0.484338 | 0.386883 | 0.38798 | 0.369749 | 0.0131401 |
| IEMOCAPFour | 0.3 | [.6,1] | 441 | 0.663188 | 0.497254 | 0.601361 | 0.597913 | 0.0140939 |
