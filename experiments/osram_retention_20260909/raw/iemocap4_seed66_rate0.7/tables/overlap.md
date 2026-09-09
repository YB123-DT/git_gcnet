# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.7 | [0,.1) | 3736 | 0.0506185 | 0.0499944 | 0.0171362 | 0.00790733 | 0.0103521 |
| IEMOCAPFour | 0.7 | [.1,.2) | 3622 | 0.149452 | 0.14547 | 0.063583 | 0.0525734 | 0.00959919 |
| IEMOCAPFour | 0.7 | [.2,.4) | 5278 | 0.291308 | 0.279184 | 0.153284 | 0.133838 | 0.00945033 |
| IEMOCAPFour | 0.7 | [.4,.6) | 2920 | 0.490259 | 0.463622 | 0.326376 | 0.301248 | 0.00900149 |
| IEMOCAPFour | 0.7 | [.6,1] | 620 | 0.660581 | 0.624046 | 0.466994 | 0.452138 | 0.00905933 |
