# Same-checkpoint Base / Gap intervention

INTERNAL DIAGNOSTIC ONLY

Original cfg84 Full checkpoints; no training. P_L masks Base and Gap only at the classification input; P_B retains Base and masks Gap; P_F retains both. Memory/query/write/Local/head remain fixed.
W-F1 (%) uses label != 0 and prediction > 0. Counts are rate exposures, not independent utterances. Each seed/rate is evaluated first; macro means weight nonempty rates equally within each seed, then seeds equally.
Per-rate BEST checkpoints are Test-oracle internal checkpoints, not formal validation-selected paper results.

|Rate|Local|Base|Full|Full−Base|Gap rescue|Gap harm|N|
|---|---:|---:|---:|---:|---:|---:|---:|
|66 / 0.0|87.068|88.205|88.205|0.000|0|0|656|
|66 / 0.1|84.074|86.087|86.507|0.420|7|4|656|
|66 / 0.2|77.419|81.783|83.187|1.403|25|15|656|
|66 / 0.3|75.999|79.972|80.763|0.791|29|23|656|
|66 / 0.4|71.487|77.873|80.827|2.953|46|25|656|
|66 / 0.5|70.977|75.697|77.494|1.797|39|25|656|
|66 / 0.6|66.102|71.054|75.790|4.736|61|28|656|
|66 / 0.7|65.430|74.845|75.773|0.929|43|35|656|
|8-rate mean|74.820|79.439|81.068|1.629|250|155|5248|
|High missing|67.503|73.865|76.352|2.487|143|88|1968|

## Observable comparison (equal-rate macro)

|Observable|Rescue mean|Harm mean|Paired-rate mean difference|Raw AUROC|AUC cells|Valid rescue / harm|
|---|---:|---:|---:|---:|---:|---:|
|obs_base_gap_A_cos|0.823|0.783|0.040|0.587|6|98 / 62|
|obs_base_gap_T_cos|0.831|0.803|0.028|0.549|7|241 / 148|
|obs_base_gap_V_cos|0.917|0.915|-0.001|0.449|6|99 / 56|
|obs_base_gap_cos_mean|0.829|0.795|0.033|0.557|7|250 / 155|
|obs_base_norm|16.520|15.650|0.869|0.552|7|250 / 155|
|obs_gap_A_norm|4.463|4.182|0.281|0.511|7|250 / 155|
|obs_gap_T_norm|13.967|12.796|1.171|0.572|7|250 / 155|
|obs_gap_V_norm|5.290|4.317|0.972|0.529|7|250 / 155|
|obs_gap_base_ratio|1.429|1.349|0.080|0.537|7|250 / 155|
|obs_local_margin|0.603|0.566|0.037|0.530|7|250 / 155|
|obs_query_cos_A|0.814|0.802|0.012|0.414|6|98 / 62|
|obs_query_cos_T|0.813|0.804|0.009|0.513|7|241 / 148|
|obs_query_cos_V|0.926|0.927|-0.004|0.429|6|99 / 56|
|obs_query_cos_mean|0.815|0.802|0.013|0.517|7|250 / 155|
|obs_query_eta_A|0.287|0.302|-0.015|0.574|6|98 / 62|
|obs_query_eta_T|0.289|0.300|-0.012|0.486|7|241 / 148|
|obs_query_eta_V|0.133|0.132|0.005|0.572|6|99 / 56|
|obs_query_eta_mean|0.285|0.302|-0.017|0.480|7|250 / 155|
|obs_query_rho_A|0.813|0.801|0.012|0.415|6|98 / 62|
|obs_query_rho_T|0.812|0.804|0.009|0.513|7|241 / 148|
|obs_query_rho_V|0.926|0.927|-0.004|0.429|6|99 / 56|
|obs_query_rho_mean|0.815|0.802|0.013|0.517|7|250 / 155|

AUROC treats rescue as positive and larger raw observable as a higher score. Below 0.5 indicates inverse direction, not absence of signal. No sign flipping, threshold selection, learned gate, significance test, or routing is performed.
AUROC is undefined when either group is absent. Undefined zero-vector cosines and ratios are excluded with explicit missing counts. Rescue/harm mean differences are averaged only over rate cells containing both groups; separately averaged means may use different cells.
Checkpoint weights differ across rates: raw norms are not pooled as primary evidence. See observables_per_rate.csv and availability-stratified rows in observables_macro.csv to inspect consistency and confounding by current modality support.
Gold labels define retrospective rescue/harm groups only. These descriptive associations do not establish deployable discrimination, emotion shift, reliability, or causal mechanism. No head/query intervention was run in this first-stage report.
