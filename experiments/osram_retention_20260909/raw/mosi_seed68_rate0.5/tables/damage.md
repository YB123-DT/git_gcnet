# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | audio | 1 | 1376 | 0.00051202 | 0.0209132 | 0.116535 | 0.0204012 | 0.0956219 | 1 | 1 | 0.984872 | 0.999729 | 0.979092 | 0.979192 | 0.0205016 | 0.0428002 |
| CMUMOSI | 0.5 | audio | 2-3 | 720 | 0.12462 | 0.133794 | 0.15484 | 0.009174 | 0.0210461 | 0.983318 | 0.983318 | 0.979351 | 0.969478 | 0.949465 | 0.948412 | 0.0112226 | 0.00344161 |
| CMUMOSI | 0.5 | audio | 4-7 | 144 | 0.19658 | 0.20686 | 0.24389 | 0.0102795 | 0.0370301 | 0.972123 | 0.972123 | 0.960082 | 0.914688 | 0.895806 | 0.897518 | 0.0120609 | 0.00510009 |
| CMUMOSI | 0.5 | text | 1 | 1392 | 0.00164682 | 0.0218169 | 0.112223 | 0.0201701 | 0.0904057 | 0.999999 | 0.999999 | 0.989006 | 0.998832 | 0.978213 | 0.944239 | 0.0203166 | 0.0373248 |
| CMUMOSI | 0.5 | text | 2-3 | 776 | 0.124551 | 0.136823 | 0.153289 | 0.0122717 | 0.0164661 | 0.987945 | 0.987945 | 0.985685 | 0.934904 | 0.915607 | 0.908781 | 0.0143909 | 0.00256283 |
| CMUMOSI | 0.5 | text | 4-7 | 88 | 0.147951 | 0.162934 | 0.168288 | 0.0149835 | 0.00535414 | 0.993018 | 0.993018 | 0.992078 | 0.882434 | 0.864219 | 0.864526 | 0.0152354 | 0.00118116 |
| CMUMOSI | 0.5 | text | 8+ | 8 | 0.227164 | 0.241394 | 0.230249 | 0.0142296 | -0.0111446 | 0.991883 | 0.991882 | 0.994609 | 0.800866 | 0.784334 | 0.787587 | 0.0135321 | -0.00678691 |
| CMUMOSI | 0.5 | visual | 1 | 1320 | 0.000950338 | 0.0210029 | 0.221302 | 0.0200526 | 0.200299 | 0.999999 | 0.999999 | 0.951975 | 0.999661 | 0.979026 | 1.01331 | 0.0202402 | 0.0802444 |
| CMUMOSI | 0.5 | visual | 2-3 | 944 | 0.244886 | 0.250399 | 0.312536 | 0.00551233 | 0.0621376 | 0.944484 | 0.944484 | 0.92827 | 1.00821 | 0.987394 | 0.998117 | 0.00817971 | 0.00747944 |
| CMUMOSI | 0.5 | visual | 4-7 | 176 | 0.316634 | 0.322313 | 0.349442 | 0.00567974 | 0.0271283 | 0.927484 | 0.927484 | 0.919453 | 0.957209 | 0.937445 | 0.947233 | 0.00924958 | 0.00574642 |
