# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | [0,.1) | 285 | 0.059527 | 0.0495698 | 0.0373791 | 0.0276919 | 0.0172194 |
| IEMOCAPFour | 0.1 | [.1,.2) | 612 | 0.151283 | 0.120988 | 0.121169 | 0.105786 | 0.0175751 |
| IEMOCAPFour | 0.1 | [.2,.4) | 1269 | 0.296095 | 0.226408 | 0.258346 | 0.230398 | 0.0169849 |
| IEMOCAPFour | 0.1 | [.4,.6) | 643 | 0.486201 | 0.354643 | 0.442643 | 0.396142 | 0.0170697 |
| IEMOCAPFour | 0.1 | [.6,1] | 207 | 0.670232 | 0.454458 | 0.61253 | 0.602153 | 0.0174184 |
