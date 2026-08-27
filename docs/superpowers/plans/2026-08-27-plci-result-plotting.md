# PLCI 实验结果绘图实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个可直接读取 PLCI 与 Original GCNet 正式结果并生成完整论文诊断图的单文件命令行工具。

**架构：** 采集层把 JSON/NPZ 统一为 `ScoreRecord`；汇总层按 dataset/method/rate/seed 对齐；绘图层只消费统一矩阵并输出 CSV、PNG 和 PDF。Original 指标从保存预测重算，避免依赖文件名精度。

**技术栈：** Python 3、NumPy、Matplotlib、unittest。

---

### 任务 1：锁定采集和指标行为

**文件：**
- 创建：`tests/test_plot_plci_results.py`
- 创建：`scripts/plot_plci_results.py`

- [ ] **步骤 1：编写失败测试**

覆盖 weighted F1、PLCI JSON 读取、Original NPZ 读取、严格缺失检查与
seed/rate 矩阵对齐。

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_plot_plci_results -v`

预期：因 `scripts.plot_plci_results` 不存在而失败。

- [ ] **步骤 3：实现最小采集器**

实现 `weighted_f1_score`、`load_plci_records`、`load_original_records`、
`build_matrix` 和 `validate_grid`。NPZ 使用 `allow_pickle=True`，递归查找包含
`test_labels`、`test_preds`、`test_fmask` 的保存字典。

- [ ] **步骤 4：确认采集测试通过**

运行：`python -m unittest tests.test_plot_plci_results -v`

预期：全部通过。

### 任务 2：实现完整绘图命令

**文件：**
- 修改：`scripts/plot_plci_results.py`
- 修改：`tests/test_plot_plci_results.py`

- [ ] **步骤 1：补充端到端输出测试**

使用临时结果目录调用 `main()`，断言 `scores.csv`、四组 PNG/PDF 图存在且
非空。

- [ ] **步骤 2：实现 CLI 与图形输出**

CLI 接受 dataset、PLCI根、Original根、seeds、rates、metric、error和输出目录。
实现 mean curve、seed curves、score heatmaps、delta heatmap 和 delta curve；
默认 sample SD，保存300 DPI PNG及矢量PDF。

- [ ] **步骤 3：运行测试**

运行：`MPLBACKEND=Agg python -m unittest tests.test_plot_plci_results -v`

预期：全部通过且无交互式后端依赖。

### 任务 3：真实结果验证与中文说明

**文件：**
- 创建：`docs/PLCI_RESULT_PLOTTING_CN.md`

- [ ] **步骤 1：在 IEMOCAP-6 真实结果上运行**

使用 seeds 66–70、rates 0.0–0.7，生成 `experiments/.../figures`；严格检查
40+40条记录，并核对均值与已有总账一致。

- [ ] **步骤 2：编写中文说明**

记录安装要求、单数据集与三数据集命令、输出文件解释、论文图注模板和
“seed-aligned但mask未严格配对”的限制。

- [ ] **步骤 3：最终验证与提交**

运行完整 unittest、`python -m py_compile scripts/plot_plci_results.py`，检查
CSV行数、PNG/PDF存在、Git diff；按 Lore 协议提交。

