# 二十候选锁定清单

INTERNAL DIAGNOSTIC ONLY

**用户已撤回此清单，未训练，不得启动。** 以下保留来源核查记录，不是新一轮推荐。

全部来源为视觉、视觉问答或推荐系统，而非 MSA/MERC。每项为显式的 OSRAM 迁移版本，不是整篇原论文复现，也不预设提分。每项 seed66/100epochs/一个固定配置；共同的128维安全输入包装＋零初始化输出残差，不改变原 Flat 内容路径、Memory 或 loss。

|序号|候选与原文|本轮具体机制|新增参数|
|---:|---|---|---:|
|1|[film](https://doi.org/10.1609/aaai.v32i1.11671)|Local 条件仿射调制历史|308680|
|2|[se](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)|跨有效槽位 squeeze/excitation|307528|
|3|[eca](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)|通道相邻坐标一维卷积|305349|
|4|[cbam](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)|通道门＋固定槽位门|307542|
|5|[gct](https://arxiv.org/abs/1909.11519)|L2 通道归一化变换|305728|
|6|[simam](https://proceedings.mlr.press/v139/yang21o.html)|槽位能量权重|305344|
|7|[sk](https://arxiv.org/abs/1903.06586)|双 dense 分支逐通道选择|350944|
|8|[mlb](https://arxiv.org/abs/1610.04325)|低秩 Hadamard 双线性|354880|
|9|[mfb](https://arxiv.org/abs/1708.01471)|4 因子乘积求和及归一化|453952|
|10|[mutan](https://arxiv.org/abs/1705.06676)|两级 Tucker，rank=4|486976|
|11|[block](https://arxiv.org/abs/1902.00038)|4 chunks×rank4 双线性|388672|
|12|[mcb](https://doi.org/10.18653/v1/D16-1044)|固定 CountSketch1024＋FFT|436544|
|13|[ban](https://arxiv.org/abs/1805.07932)|单 Local query 的双线性 evidence attention|388032|
|14|[dcnv2](https://arxiv.org/abs/2008.13535)|两层 rank32 显式 cross|470592|
|15|[cin](https://arxiv.org/abs/1803.05170)|两层16-map vector-wise交叉|311248|
|16|[autoint](https://arxiv.org/abs/1810.11921)|两层双头槽位交互|437440|
|17|[din](https://arxiv.org/abs/1706.06978)|Local-conditioned 未归一化激活池化|382561|
|18|[dlrm](https://arxiv.org/abs/1906.00091)|10 个 pair dot＋Local|323136|
|19|[aff](https://arxiv.org/abs/2009.14082)|Local/历史 pooled channel blend|322048|
|20|[nonlocal](https://arxiv.org/abs/1711.07971)|signed dot/active-count token interaction|338432|

原 Flat 参数13,509,793；新增参数包含共同投影、type embedding 和128→1600输出，不是仅论文算子的参数。原 Flat 不冻结。上述不是严格等参数比较；SimAM 算子本身无参数，但公共适配层有参数。

五槽顺序固定 Local/Base/Gap-A/Gap-T/Gap-V，Local 和 Memory 投影共享宽度128、Memory只取forward512；inactive/padding先后安全屏蔽，首个有效 utterance residual为零。type embedding初始化零。无额外随机dropout。非BAN双线性项使用active-history mean，BAN保留四路历史；各方法的结构差异和参数差异均保留。

来源核查：见 [paper_bank.json](paper_bank.json) 和三份 literature JSON，含方法位置和代码链接。SK 原作者仓库当前404，使用确认来源的原代码fork及合作者版本核对；DCNv2为官方Google实现，未确认文件由论文作者本人编写。部分代码无许可或许可不统一，全部按公开公式独立编写，没有复制/分发第三方实现。AFF和DIN的论文/代码差异已记录；AFF本轮为pooled适配，非完整空间MS-CAM。CBAM卷积仅跨固定槽位；SK用dense分支替代空间卷积。

这些适配与此前Gate有概念亲缘，不称20种全新原创机制；但没有重复换宽度、没有把AFF/iAFF算两项，也没有原样复跑已有实验。范围已锁定，不按测试结果再改超参数。逐率Test-oracle加20候选筛选会产生选择偏差，胜者仍须独立验证才能成为论文结论。
