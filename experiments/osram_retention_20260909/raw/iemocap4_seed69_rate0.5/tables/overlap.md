# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAPFour | 0.5 | [0,.1) | 2788 | 0.0508997 | 0.049328 | 0.0190873 | 0.0086305 | 0.0122493 |
| IEMOCAPFour | 0.5 | [.1,.2) | 2881 | 0.149856 | 0.140739 | 0.0770743 | 0.0621833 | 0.0118429 |
| IEMOCAPFour | 0.5 | [.2,.4) | 4102 | 0.289147 | 0.260333 | 0.17361 | 0.153025 | 0.01156 |
| IEMOCAPFour | 0.5 | [.4,.6) | 2470 | 0.490904 | 0.431157 | 0.321266 | 0.297658 | 0.0110866 |
| IEMOCAPFour | 0.5 | [.6,1] | 807 | 0.68445 | 0.56499 | 0.558043 | 0.53873 | 0.0118264 |
