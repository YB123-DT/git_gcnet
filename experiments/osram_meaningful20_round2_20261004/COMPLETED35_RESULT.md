# 已完成35个候选的阶段结果

INTERNAL DIAGNOSTIC ONLY

2026-10-04 UTC；MOSI seed66，100 epochs，per-rate TEST-ORACLE；不是正式 validation 选模结果。只汇总完成的运行，不把未完成运行纳入排名。第一批20个使用 SUMMARY.json；第二批15个读取 biggpu runs/*/seed_66/metrics.json，且 PROVENANCE 标记 complete / exit_code=0 / outputs_verified=true。

原 Flat：8-rate 81.068095%，high-missing 76.352251%。35个完成候选的两个汇总指标均未超过该基线。单种子不能推断统计显著性，批量 Test-oracle 排名不能作为泛化收益证据。

|批次|方法|8-rate W-F1 (%)|Δ百分点|High (.5/.6/.7) (%)|
|---|---|---:|---:|---:|
|1|perceiver_io|80.818|-0.250|76.126|
|1|sheaf_hypergnn_diag|80.776|-0.292|76.236|
|2|differential_attention_v1|80.664|-0.404|76.206|
|2|dcformer_dynamic_head_composition|80.617|-0.451|75.385|
|1|residual_gated_graph_evidence|80.606|-0.462|76.117|
|1|egt_evidence|80.583|-0.486|76.067|
|1|hamburger_nmf_full|80.506|-0.562|75.362|
|2|contractive_deep_equilibrium_evidence|80.462|-0.606|75.829|
|1|pna_evidence|80.457|-0.611|75.919|
|1|crate_mssa_ista_full|80.430|-0.638|75.661|
|2|neural_logic_machine_evidence|80.429|-0.639|75.648|
|2|flowformer_conservation|80.381|-0.687|75.703|
|1|hyper_sagnn|80.310|-0.758|75.776|
|2|conditional_rational_quadratic_spline_coupling|80.294|-0.774|75.392|
|2|dense_coattention|80.286|-0.782|75.566|
|1|capsule_dynamic_routing|80.274|-0.794|75.225|
|2|dual_attention_symbolic_relations|80.269|-0.799|75.918|
|2|tokenlearner_fuser_v11|80.253|-0.815|75.495|
|1|equilibrium_aggregation|80.210|-0.858|75.491|
|2|mac_control_read_write|80.171|-0.897|75.383|
|1|slot_attention|80.144|-0.924|75.248|
|1|allset_transformer|80.140|-0.928|75.377|
|1|capsule_variational_bayes|80.090|-0.978|75.481|
|1|otke|80.075|-0.993|75.183|
|1|graph_multiset_transformer|80.074|-0.994|75.523|
|2|compositional_search_retrieval|80.030|-1.039|75.133|
|2|spdnet_bimap_reeig_logeig|80.024|-1.044|75.227|
|1|node|79.984|-1.084|75.121|
|2|mcan_encoder_decoder|79.960|-1.108|75.272|
|1|dgcnn_dynamic_edgeconv|79.913|-1.155|74.752|
|2|fspool_fsunpool_evidence|79.822|-1.246|74.914|
|1|ed_hnn|79.610|-1.458|74.639|
|1|rrn_evidence|79.583|-1.485|75.306|
|1|tabnet|79.485|-1.583|74.991|
|2|rat_spn_evidence_circuit|78.662|-2.406|73.268|

## Per-rate W-F1 (%)

|方法|0.0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Flat|88.205|86.507|83.187|80.763|80.827|77.494|75.790|75.773|
|perceiver_io|87.799|85.615|83.091|80.982|80.679|76.753|76.488|75.137|
|sheaf_hypergnn_diag|87.481|85.754|82.838|80.563|80.861|77.130|75.415|76.163|
|differential_attention_v1|86.953|85.350|82.642|80.999|80.747|76.302|76.480|75.837|
|dcformer_dynamic_head_composition|87.750|85.572|83.424|81.724|80.310|76.399|75.127|74.630|
|residual_gated_graph_evidence|87.435|85.577|83.097|80.574|79.815|76.803|76.095|75.453|
|egt_evidence|88.227|86.016|82.365|80.435|79.416|77.041|75.023|76.136|
|hamburger_nmf_full|87.548|85.989|83.303|80.345|80.773|77.040|74.988|74.059|
|contractive_deep_equilibrium_evidence|86.636|85.656|83.223|80.340|80.349|76.788|75.385|75.315|
|pna_evidence|86.822|85.606|82.088|80.538|80.842|75.967|75.843|75.948|
|crate_mssa_ista_full|87.137|85.725|83.257|80.514|79.824|76.287|74.975|75.721|
|neural_logic_machine_evidence|87.338|86.166|82.087|80.712|80.183|76.049|75.106|75.790|
|flowformer_conservation|87.317|85.755|82.349|80.975|79.543|75.886|75.539|75.684|
|hyper_sagnn|87.351|84.906|82.616|80.594|79.684|76.220|76.080|75.028|
|conditional_rational_quadratic_spline_coupling|87.418|85.746|82.417|80.755|79.842|75.250|75.681|75.245|
|dense_coattention|86.988|85.280|82.101|80.764|80.458|76.870|74.797|75.032|
|capsule_dynamic_routing|87.356|85.893|82.777|80.793|79.697|75.890|74.818|74.967|
|dual_attention_symbolic_relations|87.400|85.423|82.241|79.950|79.387|76.155|76.191|75.407|
|tokenlearner_fuser_v11|87.269|85.202|82.482|80.916|79.669|76.418|75.253|74.815|
|equilibrium_aggregation|87.058|85.409|82.815|79.983|79.944|75.933|75.505|75.036|
|mac_control_read_write|87.294|85.563|82.193|80.679|79.491|76.156|74.386|75.606|
|slot_attention|87.161|85.553|82.204|80.635|79.858|76.378|74.708|74.658|
|allset_transformer|87.331|85.428|82.041|80.741|79.444|75.740|74.636|75.756|
|capsule_variational_bayes|86.673|85.280|83.145|79.807|79.372|76.733|74.893|74.818|
|otke|86.839|85.670|82.675|80.170|79.696|75.869|74.686|74.993|
|graph_multiset_transformer|86.579|84.932|82.325|80.386|79.800|75.967|75.280|75.322|
|compositional_search_retrieval|86.764|85.880|82.789|80.070|79.336|76.370|74.561|74.467|
|spdnet_bimap_reeig_logeig|87.487|85.862|82.003|79.920|79.238|75.335|75.121|75.224|
|node|87.418|85.169|82.037|80.591|79.294|75.556|74.760|75.048|
|mcan_encoder_decoder|86.401|85.867|81.036|80.288|80.275|75.915|75.413|74.488|
|dgcnn_dynamic_edgeconv|87.310|85.568|82.544|81.032|78.592|75.993|73.938|74.326|
|fspool_fsunpool_evidence|87.702|85.264|81.732|79.354|79.781|75.808|74.259|74.676|
|ed_hnn|86.624|85.507|82.003|80.080|78.753|75.580|73.943|74.393|
|rrn_evidence|85.794|85.202|81.810|79.247|78.697|75.847|75.384|74.685|
|tabnet|86.412|84.974|82.005|79.944|77.576|75.500|74.506|74.967|
|rat_spn_evidence_circuit|87.351|84.938|80.638|79.462|77.101|73.482|73.627|72.697|

未纳入：Robust PCA 尚在训练；MBT、NetVLAD、ToMe、PPGN 尚未启动。另新增40个代码候选未训练，不属于这35个结果。
