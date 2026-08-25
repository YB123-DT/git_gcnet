# GPU 4 重试诊断

主文档：[README.md](README.md)。

这些目录是基础设施诊断证据，不是完成的科学实验。Phase A 中，分配到物理 GPU 4 的7次正式训练 attempt 均在37–59秒、记录0–5个epoch后以返回码 `-9` 退出，且没有 Python 或模型 traceback。该现象在 GPU 4 三进程和单进程条件下均复现；相同锁定 task identity 在 GPU 5、6 或 7 上成功完成。

每次重试前都先重命名并保留失败 attempt。`results/artifacts/phase_a` 下 canonical `fold_5` 只包含成功的100 epoch任务。任何失败 attempt 都未进入 `summary.json` 或报告指标。

这些证据只说明本轮运行期间 GPU 4 不适合承载这些正式训练进程，不能解释为 GenAgg 或 Soft Medoid 模型失败。
