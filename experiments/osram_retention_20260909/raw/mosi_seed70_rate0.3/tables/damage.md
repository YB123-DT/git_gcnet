# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.3 | audio | 1 | 1128 | 0.000891981 | 0.0215219 | 0.257742 | 0.0206299 | 0.23622 | 0.999999 | 0.999999 | 0.946491 | 0.999717 | 0.978508 | 1.01723 | 0.0208066 | 0.136288 |
| CMUMOSI | 0.3 | audio | 2-3 | 512 | 0.291494 | 0.293268 | 0.376904 | 0.00177442 | 0.0836353 | 0.936094 | 0.936094 | 0.906954 | 1.0198 | 0.998165 | 1.01989 | 0.00194349 | 0.0104434 |
| CMUMOSI | 0.3 | audio | 4-7 | 72 | 0.415105 | 0.415191 | 0.44196 | 8.62981e-05 | 0.0267685 | 0.89032 | 0.89032 | 0.877223 | 1.01054 | 0.989108 | 0.995363 | 0.000873119 | 0.000181139 |
| CMUMOSI | 0.3 | text | 1 | 1104 | 0.00192352 | 0.0224542 | 0.144249 | 0.0205306 | 0.121795 | 0.999999 | 0.999999 | 0.982528 | 0.998781 | 0.977593 | 0.944812 | 0.0205745 | 0.0727553 |
| CMUMOSI | 0.3 | text | 2-3 | 408 | 0.182871 | 0.191895 | 0.229955 | 0.00902398 | 0.0380603 | 0.970587 | 0.970587 | 0.958722 | 0.932614 | 0.912834 | 0.89965 | 0.0105828 | 0.00996559 |
| CMUMOSI | 0.3 | text | 4-7 | 32 | 0.20397 | 0.211053 | 0.207099 | 0.00708317 | -0.00395404 | 0.976526 | 0.976526 | 0.978223 | 0.921924 | 0.902376 | 0.90347 | 0.00797439 | 0.000192821 |
| CMUMOSI | 0.3 | visual | 1 | 1000 | 0.000959057 | 0.0215553 | 0.240595 | 0.0205963 | 0.219039 | 0.999999 | 0.999999 | 0.952018 | 0.999675 | 0.978468 | 1.00748 | 0.0207195 | 0.132692 |
| CMUMOSI | 0.3 | visual | 2-3 | 328 | 0.279371 | 0.280617 | 0.373326 | 0.00124582 | 0.092709 | 0.943636 | 0.943636 | 0.918085 | 1.02303 | 1.00132 | 1.02895 | 0.00135472 | 0.0158467 |
| CMUMOSI | 0.3 | visual | 4-7 | 24 | 0.580675 | 0.571942 | 0.584875 | -0.00873316 | 0.0129328 | 0.861276 | 0.861276 | 0.856742 | 1.13449 | 1.11041 | 1.11682 | -0.00774285 | 0.0108454 |
