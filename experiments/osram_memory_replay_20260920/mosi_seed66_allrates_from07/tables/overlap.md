# Overlap

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | overlap_bucket | head_count | max_key_overlap_mean | mean_key_overlap_mean | write_damage_mean | write_damage_median | decay_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.1 | [0,.1) | 316 | 0.0519748 | 0.0448884 | 0.00149011 | 0.000167355 | 0.010424 |
| CMUMOSI | 0.1 | [.1,.2) | 341 | 0.153575 | 0.126914 | 0.0175076 | 0.00519155 | 0.0102797 |
| CMUMOSI | 0.1 | [.2,.4) | 583 | 0.290347 | 0.22499 | 0.0496149 | 0.0147221 | 0.00990688 |
| CMUMOSI | 0.1 | [.4,.6) | 297 | 0.453851 | 0.310319 | 0.102969 | 0.0559108 | 0.00911948 |
| CMUMOSI | 0.1 | [.6,1] | 119 | 0.668254 | 0.482007 | 0.120704 | 0.0727077 | 0.00894666 |
| CMUMOSI | 0.3 | [0,.1) | 1001 | 0.0516932 | 0.0471745 | 0.0018694 | 0.000225499 | 0.0107188 |
| CMUMOSI | 0.3 | [.1,.2) | 1050 | 0.150103 | 0.132466 | 0.0162867 | 0.00537587 | 0.0103008 |
| CMUMOSI | 0.3 | [.2,.4) | 1381 | 0.298267 | 0.247626 | 0.0493693 | 0.0142465 | 0.0100512 |
| CMUMOSI | 0.3 | [.4,.6) | 585 | 0.457506 | 0.346994 | 0.0928275 | 0.0482688 | 0.00905526 |
| CMUMOSI | 0.3 | [.6,1] | 247 | 0.666791 | 0.505179 | 0.112972 | 0.072713 | 0.00923734 |
| CMUMOSI | 0.5 | [0,.1) | 2047 | 0.0507009 | 0.0491005 | 0.00189632 | 0.000197478 | 0.011243 |
| CMUMOSI | 0.5 | [.1,.2) | 1792 | 0.145246 | 0.138157 | 0.0146806 | 0.0067942 | 0.0109095 |
| CMUMOSI | 0.5 | [.2,.4) | 2101 | 0.295436 | 0.267416 | 0.0503541 | 0.0221454 | 0.0102569 |
| CMUMOSI | 0.5 | [.4,.6) | 846 | 0.454267 | 0.380928 | 0.0746822 | 0.045923 | 0.00871852 |
| CMUMOSI | 0.5 | [.6,1] | 342 | 0.673682 | 0.578788 | 0.143362 | 0.120126 | 0.00920431 |
| CMUMOSI | 0.7 | [0,.1) | 2701 | 0.0519654 | 0.0515258 | 0.00205348 | 0.000186086 | 0.0113237 |
| CMUMOSI | 0.7 | [.1,.2) | 2131 | 0.140211 | 0.137502 | 0.0129708 | 0.00650525 | 0.0109689 |
| CMUMOSI | 0.7 | [.2,.4) | 2491 | 0.29204 | 0.281095 | 0.0431613 | 0.022623 | 0.00990486 |
| CMUMOSI | 0.7 | [.4,.6) | 914 | 0.451375 | 0.417882 | 0.07973 | 0.0522542 | 0.00915541 |
| CMUMOSI | 0.7 | [.6,1] | 379 | 0.678326 | 0.63881 | 0.169215 | 0.149962 | 0.00797615 |
