# Damage

Coverage and distance count unique missing queries, deduplicated by dataset, missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. Damage and overlap counts are head-level retention records, not query counts. Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded individually, never replaced by zero; empty CSV cells and NA denote unavailable statistics. Damage is signed (negative means improvement). err_decay measures the current read; err_post measures retention for future reads only, not the current prediction. P90 uses linear interpolation between sorted observations.

| dataset | missing_rate | target_modality | distance_bucket | head_count | err_pre_mean | err_decay_mean | err_post_mean | decay_damage_mean | write_damage_mean | cos_pre_mean | cos_decay_mean | cos_post_mean | norm_ratio_pre_mean | norm_ratio_decay_mean | norm_ratio_post_mean | decay_damage_median | write_damage_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMUMOSI | 0.5 | audio | 1 | 1264 | 0.000765614 | 0.0215579 | 0.224123 | 0.0207923 | 0.202565 | 1 | 1 | 0.954993 | 0.999667 | 0.97846 | 1.00724 | 0.0208664 | 0.094511 |
| CMUMOSI | 0.5 | audio | 2-3 | 864 | 0.245136 | 0.248973 | 0.302758 | 0.0038367 | 0.0537855 | 0.951209 | 0.951209 | 0.93668 | 1.00435 | 0.983037 | 0.99047 | 0.00452825 | 0.00854072 |
| CMUMOSI | 0.5 | audio | 4-7 | 136 | 0.378212 | 0.379995 | 0.392502 | 0.00178273 | 0.0125076 | 0.903622 | 0.903622 | 0.900215 | 0.979722 | 0.958934 | 0.962141 | 0.00209282 | 0.0021395 |
| CMUMOSI | 0.5 | text | 1 | 1304 | 0.00175848 | 0.0223894 | 0.155642 | 0.020631 | 0.133252 | 0.999999 | 0.999999 | 0.978851 | 0.998839 | 0.977649 | 0.93668 | 0.0206377 | 0.0775723 |
| CMUMOSI | 0.5 | text | 2-3 | 904 | 0.204198 | 0.212783 | 0.255044 | 0.00858545 | 0.0422609 | 0.965436 | 0.965436 | 0.953793 | 0.921166 | 0.901629 | 0.897734 | 0.0103381 | 0.00971975 |
| CMUMOSI | 0.5 | text | 4-7 | 272 | 0.302155 | 0.310618 | 0.305761 | 0.00846313 | -0.00485759 | 0.939143 | 0.939143 | 0.941557 | 0.862864 | 0.844568 | 0.835329 | 0.0113561 | 0.00384799 |
| CMUMOSI | 0.5 | text | 8+ | 8 | 0.403396 | 0.413291 | 0.41032 | 0.00989496 | -0.00297179 | 0.917707 | 0.917707 | 0.913866 | 0.718461 | 0.703228 | 0.714892 | 0.0117383 | -0.00655569 |
| CMUMOSI | 0.5 | visual | 1 | 1328 | 0.000961902 | 0.0216639 | 0.207275 | 0.020702 | 0.185612 | 1 | 1 | 0.959792 | 0.999562 | 0.978357 | 0.993118 | 0.020804 | 0.0962271 |
| CMUMOSI | 0.5 | visual | 2-3 | 824 | 0.215044 | 0.220686 | 0.27938 | 0.00564236 | 0.058694 | 0.95916 | 0.95916 | 0.942694 | 0.986102 | 0.96518 | 0.975494 | 0.00557007 | 0.00339415 |
| CMUMOSI | 0.5 | visual | 4-7 | 256 | 0.347468 | 0.349512 | 0.364147 | 0.00204425 | 0.0146345 | 0.924228 | 0.924228 | 0.915022 | 0.968053 | 0.947512 | 0.957891 | 0.00327693 | 0.00260063 |
