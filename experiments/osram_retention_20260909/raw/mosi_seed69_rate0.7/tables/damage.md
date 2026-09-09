# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.7 | audio | 1 | 1256 | 0.000709501 | 0.0218043 | 0.189155 | 0.0210948 | 0.167351 | 1 | 1 | 0.969819 | 0.999607 | 0.978204 | 0.995794 | 0.021195 | 0.103717 |
| CMUMOSI | 0.7 | audio | 2-3 | 1176 | 0.211016 | 0.215418 | 0.261441 | 0.00440216 | 0.0460229 | 0.964818 | 0.964818 | 0.952549 | 0.987961 | 0.966803 | 0.968124 | 0.00450135 | 0.0087346 |
| CMUMOSI | 0.7 | audio | 4-7 | 512 | 0.297755 | 0.302165 | 0.337108 | 0.0044107 | 0.0349428 | 0.941864 | 0.941864 | 0.929771 | 0.936537 | 0.916479 | 0.932759 | 0.0052249 | 0.00639385 |
| CMUMOSI | 0.7 | audio | 8+ | 24 | 0.452592 | 0.449857 | 0.451125 | -0.00273457 | 0.00126763 | 0.898113 | 0.898113 | 0.897085 | 1.02039 | 0.998547 | 0.990733 | -0.00418913 | 0.00990897 |
| CMUMOSI | 0.7 | text | 1 | 1304 | 0.00156022 | 0.022436 | 0.149067 | 0.0208758 | 0.126631 | 0.999999 | 0.999999 | 0.976324 | 0.998987 | 0.977597 | 0.938856 | 0.0209195 | 0.0493407 |
| CMUMOSI | 0.7 | text | 2-3 | 1064 | 0.178453 | 0.189198 | 0.234278 | 0.0107451 | 0.0450797 | 0.96832 | 0.96832 | 0.955559 | 0.926316 | 0.906478 | 0.891721 | 0.0129454 | 0.00626159 |
| CMUMOSI | 0.7 | text | 4-7 | 344 | 0.311731 | 0.320268 | 0.337537 | 0.00853778 | 0.0172681 | 0.930957 | 0.930957 | 0.922275 | 0.85202 | 0.833772 | 0.834621 | 0.0104017 | 0.00242775 |
| CMUMOSI | 0.7 | text | 8+ | 80 | 0.501668 | 0.509404 | 0.519535 | 0.00773619 | 0.0101313 | 0.851804 | 0.851804 | 0.842443 | 0.662729 | 0.648519 | 0.670825 | 0.00905007 | 0.00251274 |
| CMUMOSI | 0.7 | visual | 1 | 1280 | 0.00102392 | 0.021954 | 0.175711 | 0.0209301 | 0.153757 | 1 | 1 | 0.966234 | 0.999465 | 0.978064 | 0.987831 | 0.0209739 | 0.0546066 |
| CMUMOSI | 0.7 | visual | 2-3 | 1112 | 0.209918 | 0.216672 | 0.25764 | 0.00675443 | 0.0409676 | 0.956095 | 0.956095 | 0.944322 | 0.980481 | 0.959487 | 0.958951 | 0.00877174 | 0.00310184 |
| CMUMOSI | 0.7 | visual | 4-7 | 440 | 0.293125 | 0.29912 | 0.313669 | 0.0059952 | 0.0145488 | 0.941597 | 0.941597 | 0.937234 | 0.939061 | 0.918954 | 0.918042 | 0.00814403 | 0.000900492 |
| CMUMOSI | 0.7 | visual | 8+ | 24 | 0.480807 | 0.480726 | 0.442773 | -8.02058e-05 | -0.0379533 | 0.87903 | 0.87903 | 0.897014 | 0.967946 | 0.947211 | 0.927982 | -0.00272919 | -0.00540081 |
