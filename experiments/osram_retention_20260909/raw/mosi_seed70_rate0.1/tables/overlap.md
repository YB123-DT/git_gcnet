# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 44 | 0.0613936 | 0.0597798 | 0.0145826 | 0.00240885 | 0.0199095 |
| CMUMOSI | 0.1 | [.1,.2) | 120 | 0.162787 | 0.141417 | 0.0731588 | 0.0505958 | 0.0197586 |
| CMUMOSI | 0.1 | [.2,.4) | 804 | 0.317454 | 0.242476 | 0.16173 | 0.0876416 | 0.0191927 |
| CMUMOSI | 0.1 | [.4,.6) | 624 | 0.464116 | 0.344835 | 0.269755 | 0.17078 | 0.0188992 |
| CMUMOSI | 0.1 | [.6,1] | 24 | 0.630027 | 0.422321 | 0.262316 | 0.14583 | 0.0200771 |
