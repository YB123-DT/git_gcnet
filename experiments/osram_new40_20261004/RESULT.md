# 新增40项可运行读出适配

INTERNAL DIAGNOSTIC ONLY — CODE IMPLEMENTATION, NOT PERFORMANCE RESULTS

## 交付范围

新增40个独立方法ID及完整核心计算，接入原 `--osram-meaningful-block` 开关。代码基准为 `789c6dd`，实际分支为 `feature/osram-uniform-forced-text`。本次没有启动训练或评测，没有新增W-F1。

保持原OSRAM read/write/query、Local skip、Flat/head、任务loss和random-missing协议。只改变原Flat adapter接收的Local/forward Base/active Gap。所有方法零初始化输出桥；关闭时不创建模块。新增依赖为零。

“有意义”指保留明确、非重复的计算机制及可检验迁移假设，不指已经提分，也不指40个互不相关的大方向。筛选允许相近假设，但不计深宽变化、改名或简单算子包装。见[来源与差异清单](CATALOG.json)和其中逐项design_file。

## 已验证与未验证

主代理实际运行：

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_new40
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_input tests.test_meaningful_block_integration
```

目录检查2项通过；共享输入与完整模型集成3项通过。共享模板覆盖所有新40，且遍历已有方法做回归：初始化identity、inactive NaN隔离、padding/首句、3次优化更新和有限梯度；完整模型train/eval初始预测、原参数及RNG一致。完整模型使用合成特征，不是MOSI正式实验。

尚未运行biggpu真实batch32 CUDA可行性检查、100epoch训练或多种子实验。CPU检查不能证明显存/吞吐、100epoch稳定性或提分；HNN/LNN、ODE及高阶图可能较慢。仍需沿用已有CUDA检查，在健康卡上生成新版本readiness，不能复用旧源码版本的性能profile。

## 重要适配说明

- NPS使用确定性straight-through argmax，不用源码Gumbel噪声，保留单rule/primary/context更新链；不消耗原训练forward RNG。
- LNN使用位置相关SPD质量矩阵的Lagrangian，仍计算速度Hessian和混合导数，解非奇异线性系统；不把失败的LNN替换为普通ODE。
- CPFlow用解析链式法则计算完整ICNN梯度映射，可在inference_mode运行；训练仍穿过混合二阶导数。
- Grassmann保留跨分支projector池化与rank-q选择；极小特征间隙使用显式导数正则。原论文含AFEW视频表情实验，不是MSA/MERC，不能声称来源完全不涉及情感。
- SOS先平方再做Gaussian交叉积分，未把缺失叶子简单置一。DST按论文修正Omega合成，并明确不等同作者有错误的实现。
- MFN原PDF直链受限，依据原文缓存、官方会议slides与作者完整核心代码；没有假称取得完整PDF。
- Difflogic保留soft训练/hard评估区别。所有“时间/守恒/图结构”均指内部计算或假设，不指真实情绪物理机制。
- 这些是独立公式适配，不是40篇论文整套模型/训练流程的复现；未移入额外loss、跨样本支持集或缺失预测。

## 使用方法

已有cfg84启动命令只切换下列参数，其他原参数不动：

```text
--osram-meaningful-block n3_recursive_neighbor_volumes
```

全部40个合法值见下表。训练入口与默认Flat开关未另造一套。

若使用现有受控实验启动器，先预览选择（已实测，不修改台账）：

```bash
python -m experiments.osram_new40_20261004.prepare --candidate n3_recursive_neighbor_volumes --dry-run
```

后续授权训练时，在所属biggpu环境选择并预留：

```bash
python -m experiments.osram_new40_20261004.prepare --candidate n3_recursive_neighbor_volumes --output experiments/osram_new40_20261004/SELECTED.json
```

非dry-run会更新方法台账的预留项，但不会启动进程；输出文件不可覆盖。提交选择与台账后，继续使用 `experiments.osram_meaningful20_round2_20261004.manifest snapshot`、共享 `cuda_check` 和原 `run` 入口，令其 `--manifest` 指向SELECTED.json；所有原必需的源码/数据/基线哈希、readiness、GPU UUID参数仍保留。不要把源码worktree直接当运行快照。

总训练上限仍为60；已有40项占位，新增代码40项不等于自动获准再训练40项，选择器最多再预留20项。训练仍为seed66/100epochs/原8率，沿用原per-rate Test-oracle协议时只能内部筛选，不能称validation选模或正式论文成绩。biggpu物理GPU4禁用。本次未操作任何既有远程队列。

## 新增可训练参数

从实际factory实例测量，包含投影/decoder；不是论文参数量。完整记录及实现文件SHA256见[PARAMETERS.json](PARAMETERS.json)。

| # | 方法ID | 新增参数 |
|---:|---|---:|
| 1 | `optimization_dpp_subset_readout` | 173,489 |
| 2 | `next_hyperbolic_gyrovector_readout` | 449,056 |
| 3 | `next_bernstein_spectral_readout` | 21,287 |
| 4 | `next_diffusion_scattering_readout` | 187,168 |
| 5 | `next_sandwich_lipschitz_readout` | 318,208 |
| 6 | `repr_kan_function_composition` | 334,336 |
| 7 | `repr_deep_lattice_composition` | 195,680 |
| 8 | `repr_differentiable_logic_circuit` | 240,128 |
| 9 | `conditional_03_neural_ode_finite_flow` | 299,648 |
| 10 | `matrix_tree_nonprojective_evidence` | 129,665 |
| 11 | `sparsemap_role_partition` | 322,819 |
| 12 | `janossy_full_role_symmetrization` | 347,648 |
| 13 | `diffpool_hierarchical_evidence_graph` | 130,386 |
| 14 | `cwn_cellular_evidence` | 362,112 |
| 15 | `simplicial_hodge_evidence` | 166,400 |
| 16 | `nested_gnn_rooted_evidence` | 159,235 |
| 17 | `graph_unet_evidence_encoder_decoder` | 122,176 |
| 18 | `crfrnn_latent_evidence_states` | 112,090 |
| 19 | `graph_matching_base_gap_pairs` | 91,712 |
| 20 | `conditional_new_01_full_dynamic_hypernetwork` | 825,792 |
| 21 | `conditional_new_02_ltc_conductance` | 449,088 |
| 22 | `conditional_new_03_hamiltonian_flow` | 305,793 |
| 23 | `conditional_new_04_lagrangian_flow` | 238,193 |
| 24 | `conditional_new_05_contracting_ren` | 298,504 |
| 25 | `conditional_new_06_rim` | 274,752 |
| 26 | `conditional_new_07_neural_production` | 175,168 |
| 27 | `conditional_new_08_neural_interpreter` | 141,296 |
| 28 | `conditional_new_09_sympnet` | 293,888 |
| 29 | `conditional_new_10_cornn` | 559,552 |
| 30 | `repr_dst_corrected_evidence_combination` | 99,958 |
| 31 | `repr_bcos_alignment_network` | 286,528 |
| 32 | `repr_mfn_gabor_filter_chain` | 258,496 |
| 33 | `repr_linf_distance_network` | 255,392 |
| 34 | `repr_lista_cpss_sparse_pursuit` | 239,365 |
| 35 | `repr_sos_signed_probability_circuit` | 13,961 |
| 36 | `repr_grassmann_projection_pooling` | 186,992 |
| 37 | `repr_neural_kernel_composition` | 117,131 |
| 38 | `repr_convex_potential_gradient_flow` | 78,578 |
| 39 | `n3_recursive_neighbor_volumes` | 119,201 |
| 40 | `pointcnn_x_transformed_evidence` | 114,624 |

没有性能结果，因此本次不推荐哪个“最能提分”，也不基于名称自动扩大多种子。
