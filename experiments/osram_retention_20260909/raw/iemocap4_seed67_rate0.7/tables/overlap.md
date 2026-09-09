# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | [0,.1) | 3510 | 0.0497099 | 0.0490118 | 0.0166682 | 0.00692859 | 0.0105348 |
| IEMOCAPFour | 0.7 | [.1,.2) | 3806 | 0.149528 | 0.145811 | 0.0637196 | 0.0546255 | 0.0100065 |
| IEMOCAPFour | 0.7 | [.2,.4) | 5822 | 0.289776 | 0.278871 | 0.157716 | 0.142426 | 0.00987105 |
| IEMOCAPFour | 0.7 | [.4,.6) | 2414 | 0.481133 | 0.455689 | 0.308668 | 0.303303 | 0.00934364 |
| IEMOCAPFour | 0.7 | [.6,1] | 816 | 0.681437 | 0.638521 | 0.498102 | 0.528844 | 0.0103367 |
