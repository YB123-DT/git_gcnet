# CMU-MOSI Slot Missing-M3 Cyclic Rate 诊断

## 状态

`COMPLETE — EFFICIENT COMPROMISE`。

## 研究问题

固定率训练比统一八率训练明显更差，但原 `all` 模式在每个 batch 内计算八个 missing-rate
view，而 fixed 每个 batch 只计算一个 view。因此 fixed 的下降同时可能来自：

1. 失去跨 rate 多样性；
2. 失去同一步八 view 的梯度平均和训练预算。

Cyclic 控制每个 batch 只计算一个 rate，同时按 batch 在 `0.0--0.7` 之间均衡轮换：

```text
batch 0 -> eta 0.0
batch 1 -> eta 0.1
...
batch 7 -> eta 0.7
然后循环
```

它保留跨 rate 学习，但与 fixed 一样每个 batch 只有一个 view。

## 锁定配置

- CMU-MOSI，fold 1，seeds 66--70；
- Slot Missing-M3、DualGate MMoE、EMA teacher；
- hidden 200、latent 256、window 2/2、time attention off；
- regression MSE、LR `5e-4`、weight decay `1e-5`、JEPA weight `0.1`；
- 100 epochs、batch size 32；
- 八率 validation W-F1 均值选模，一个 checkpoint 测试全部八率；
- Original、Fixed 和 All 结果均直接继承。

## 结果

| 协议 | 每 batch views | 跨 rate | 八率 W-F1 | High W-F1 |
|---|---:|---:|---:|---:|
| Fixed | 1 | 否 | 77.264 | 72.634 |
| Cyclic | 1 | 是 | 78.084 | 74.439 |
| All | 8 | 是 | 78.868 | 75.121 |

Cyclic 相对 Fixed 提高 `0.820` 点，高缺失提高 `1.806` 点，证明跨 rate 多样性具有明显
正迁移。All 相对 Cyclic 仍提高 `0.784` 点，高缺失提高 `0.681` 点，说明同 batch 八 view
平均和更大的 view 预算也有贡献。

因此：正式精度结果继续使用 All；Cyclic 可作为计算量显著更低的效率版本，但当前不能取代
All 主结果。

详细结果见 `results/SUMMARY.md`。
