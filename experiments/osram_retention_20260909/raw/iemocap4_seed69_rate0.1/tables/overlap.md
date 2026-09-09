# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.1 | [0,.1) | 343 | 0.0532777 | 0.0467156 | 0.0313102 | 0.0173193 | 0.01711 |
| IEMOCAPFour | 0.1 | [.1,.2) | 628 | 0.151295 | 0.122355 | 0.120829 | 0.0986243 | 0.0175916 |
| IEMOCAPFour | 0.1 | [.2,.4) | 1046 | 0.288809 | 0.22393 | 0.235147 | 0.20704 | 0.0169331 |
| IEMOCAPFour | 0.1 | [.4,.6) | 661 | 0.49051 | 0.360333 | 0.417276 | 0.377909 | 0.0168138 |
| IEMOCAPFour | 0.1 | [.6,1] | 234 | 0.676506 | 0.465981 | 0.647717 | 0.604756 | 0.0171757 |
