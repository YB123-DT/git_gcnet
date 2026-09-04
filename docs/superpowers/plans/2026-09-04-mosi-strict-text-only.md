# MOSI strict Text-only diagnostic implementation plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框跟踪进度。

**目标：** 实现并运行冻结 DeBERTa feature 的严格 MOSI Text-only 五种子诊断。

**架构：** 独立目录只包含 Text projector、packed BiGRU 和 regression head；复用现有 official loader、训练指标和原子产物工具，不复制数据协议。

**技术栈：** Python、PyTorch、pytest、现有 GCNet/MOSI utilities、biggpu official `s0` environment。

---

### 任务 1：锁定 Text-only 模型合同

**文件：**
- 创建：`gcnet_mosi_text_only/model.py`
- 创建：`gcnet_mosi_text_only/tests/test_model.py`

- [x] 先写测试，要求改变 Audio/Visual 不改变输出、padding 不影响有效位置、输出形状为 `[L,B,1]` 且 backward 有限。
- [x] 运行测试并确认因模块不存在而失败。
- [x] 实现 LayerNorm--Linear--GELU、packed one-layer BiGRU 和 regression head。
- [x] 运行模型测试并确认通过。

### 任务 2：训练入口与五种子 runner

**文件：**
- 创建：`gcnet_mosi_text_only/train_mosi.py`
- 创建：`gcnet_mosi_text_only/run_mosi.py`
- 创建：`gcnet_mosi_text_only/tests/test_train.py`

- [x] 先写测试，锁定 validation W-F1 checkpoint selection、五 seeds、唯一 text feature 和无多模态参数。
- [x] 运行测试并确认预期失败。
- [x] 复用 official loader/evaluate helpers，实现 100-epoch训练、轻量 JSON/NPZ产物与并发 runner。
- [x] 运行全部新测试、编译和 diff 检查。

### 任务 3：正式实验与结论

**文件：**
- 创建：`gcnet_mosi_text_only/results/RESULT.md`
- 生成：`gcnet_mosi_text_only/results/formal/seed_*/{config,history,metrics}.json`

- [x] 同步到 biggpu，使用健康 GPU 2--3（排除 GPU4）运行 seeds 66--70。
- [x] 核验每份 history 100 epochs、validation 选优、test 非坍塌与五种子均值。
- [x] 只同步轻量结果，不提交 checkpoint。
- [x] 写结论、提交并推送 `feature/missing-m3-target-ple`。
