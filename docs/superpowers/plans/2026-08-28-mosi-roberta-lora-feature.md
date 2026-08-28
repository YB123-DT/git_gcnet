# MOSI RoBERTa-Large LoRA 文本特征实现计划

**目标：** 用 train-only MOSI sentiment supervision 适配本地 RoBERTa-large 的 query/value LoRA，导出无泄漏的 1024D text bank，并以 all-rates Missing-M3 seed66 判断是否突破 87。

## 任务 1：数据与核心模块 TDD

- [ ] 新建 `gcnet_missing_m3/text_lora.py` 与 focused tests；
- [ ] 先写 split/UID、pooling、trainable-parameter、导出契约红灯测试；
- [ ] 实现不依赖 transformers import 的数据、pooling、hash 核心；
- [ ] targeted tests 转绿。

## 任务 2：远程 RoBERTa/PEFT 集成

- [ ] 新建 `gcnet_missing_m3/train_text_lora.py` CLI；
- [ ] lazy import transformers/peft；
- [ ] 实现 train-only DataLoader、SmoothL1、val-MAE checkpoint、early stop；
- [ ] 实现 trainable-only best state、adapter/head保存、无标签导出；
- [ ] real checkpoint integration 验证 hidden=1024、LoRA targets=48、一次 forward/backward finite。

## 任务 3：特征生成与下游 seed66

- [ ] biggpu GPU smoke；
- [ ] 正式训练 LoRA 并导出 2,199 个 features；
- [ ] 审计 UID/shape/dtype/finite/hash/split；
- [ ] 使用新 text bank 运行 all-rates Missing-M3 seed66；
- [ ] 按 87.0、+0.5 与 nonzero delta 门槛决策。

## 任务 4：交付

- [ ] 独立规格与代码质量审查；
- [ ] 更新实验报告；
- [ ] Lore commit 并推送 GitHub；
- [ ] 若通过，扩 downstream seeds 67–70；若失败，关闭 frozen-bank 上游路线。

