# 与已有ComP的实际边界

2026-10-23；只读比较，无模型改动、推理或训练。

核对本地作者实现：`/data2/yb/paper/05_reproduction/external_repos/ComP`，
HEAD `28192d3a5683543d7383e40898f9a98d1f114a08`。

`ComP/model.py:10–18`将batch内所有query与原型展平后计算余弦相似度。
`:33–53`的PromptFormer把每段序列经proj_n压成10个prototype，
在batch*n_proto轴softmax并读取，再映射成Prompt；因此确有跨样本特征访问，
不是每段对话完全独立的静态learned prototypes。
`:245–289`跨模态传递、再次生成并插值Prompt。
这些Prompt读取步骤没有把邻居真实情绪标签编码成Value；
forward虽然接收label参数，不能据此误称标签参与了Prompt读取。

| 因素 | 已核对ComP实现 | 本轮TabR迁移草案 |
|---|---|---|
| 可访问集合 | 当前mini-batch特征生成的原型 | 固定TRAIN集合中的具体话语实例 |
| 读取 | 对batch原型的余弦softmax | 共享Key的负平方距离top-k与softmax |
| 内容 | 特征原型生成跨模态Prompt | TRAIN标签编码与query-neighbor差异Value |
| 评估依赖 | 同batch其他样本的特征会参与 | TRAIN支持库，不使用其他测试样本/标签 |
| 接口 | 两轮Prompt生成与跨模态传播 | 保留CHM/Flat，新增示例读取分支（尚未实施） |

修正候选边界：不能声称跨样本访问本身是ComP没有的能力。
真正待检验的差异是TRAIN-only固定实例库及标签条件化Value。
这足以构成不同的受控迁移实验，但不自动构成HAMR第二项原创贡献。
当前不启动训练；仍需确认是否接受推理携带TRAIN实例/标签库。
