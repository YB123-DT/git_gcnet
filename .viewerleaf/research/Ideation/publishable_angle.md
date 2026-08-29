# Publishable Angle

## 研究方向

Modality-Track GCNet：面向统一缺失率模型的模态保持型对话图推理。

## 核心创新点

现有 Single-View Missing-M3 在图推理前将 observed modalities 压缩成一个 node。新方向
让 Audio、Text、Visual 分别通过共享参数的 Temporal/Speaker GCNet，直到图推理完成后
才依据 availability pattern 进行 slot fusion，从而避免 early fusion 擦除模态身份和独立
上下文证据。

## 研究问题

在不生成缺失模态、不增加多个独立 backbone 的条件下，延迟跨模态融合是否能提高一个
mixed-rate checkpoint 对单模态和双模态缺失 pattern 的统一鲁棒性？

## 方法概要

- 三个 modality-specific Student projectors 产生同维 track input；
- 三条 track 共享同一套 pre-graph recurrent、Temporal graph、Speaker graph 和 post-graph
  recurrent 参数；
- missing utterance-track 保持零输入，并在最终融合时严格排除；
- 图后将三个 masked track hidden 与 availability pattern embedding 拼接并映射回 emotion
  hidden；
- 沿用 training-only Missing-M3/EMA objective，测试不执行 Predictor。

## 对标基线

- 当前 Slot Single-View Missing-M3；
- CaM-HG 的 restore-then-mine 路线；
- 已关闭的 training-only MMoE fidelity 与 test-time completion variants。

## 预期贡献

- 识别 incomplete multimodal conversational learning 中 early fusion 的信息瓶颈；
- 提供一种参数共享但表示分轨的 relational reasoning primitive；
- 在一个模型测试八个 missing rates 的协议下检验高缺失鲁棒性。

## 目标刊物

ACL/EMNLP Findings、AAAI 或 IEEE TMM/TAFFC 的完整实验版本。

## 选择记录

- 候选方向数：3；
- 用户选择：方向 1，Modality-Track GCNet；
- 选择理由：直接针对当前证据指向的 early-fusion bottleneck，避免继续依赖不稳定的
  missing-latent hallucination；
- 弃选方向：Pattern-private experts（增量性较强）、Pre-Graph Restore（与 CaM-HG 撞车
  风险高）。

