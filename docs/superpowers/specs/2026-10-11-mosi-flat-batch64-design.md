# MOSI cfg84 Flat Batch64 设计

## 目标

运行一个 CMU-MOSI cfg84 no-JEPA Flat seed66 配方，检验将 conversation batch size 从32增至64、并按线性缩放规则将学习率从1e-3增至2e-3后的表现。

## 固定协议

- 基线：当前 cfg84 causal forward-only OSRAM、Flat、output1600、8 heads、KV64、write step0.6。
- 只改变：`batch_size: 32 -> 64`、`learning_rate: 1e-3 -> 2e-3`、`epochs: 100 -> 150`。
- 150 epochs 是当前 MOSI 完整训练口径；不是纯 batch-size 单变量消融。
- 保持 cyclic missing0.0–0.7、emotion-only、MSE、Adam、constant LR、原 mask 和 per-rate Test-oracle checkpoint 选择。
- 不启用 Nested、JEPA、completion、Gate、persistent mix 或额外 loss。

## 执行

仅 seed66，一组，从头训练。服务器 `biggpu`，使用启动时确认空闲的健康 GPU0；GPU4禁用。输出、日志、恢复状态和8个 BEST checkpoint使用独立目录。若 Batch64 OOM，保存错误证据并停止，不自动改变 batch、梯度累积或模型结构。

## 报告

完成后报告每率 W-F1/ACC、8-rate mean、high-missing mean、BEST epoch和与同种子 cfg84 Flat150的差值。结果标记为内部 Test-oracle 诊断。

