"""Generate compact tables from verified frozen-parent results, without inference."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    completed=root/'completed'
    verification=json.loads((completed/'VERIFICATION.json').read_text())
    reload=json.loads((completed/'RELOAD.json').read_text())
    assert verification['status']==reload['status']=='passed'
    modes=('neumiss','signature','ridge','csdi')
    names=dict(neumiss='NeuMiss',signature='Deep Signature',ridge='R2-D2',csdi='Latent CSDI')
    runs={m:json.loads((completed/(m+'.json')).read_text()) for m in modes}
    assert all(r['status']=='complete' and r['epoch']==150 for r in runs.values())
    first=runs['neumiss'];base=first['baseline_aggregate']
    rows=['# MOSI 四方案冻结父模型筛选','','INTERNAL TEST-ORACLE SCREEN','',
          '四组已完成。父模型为同一份原Flat miss=0.3 BEST（epoch75），仅新增分支训练。',
          'MOSI seed66，cmerc-repro，原GCNet random-mask-v1，batch32、Adam、WD1e-5。',
          '任务分支150轮，前100轮3e-4、后50轮3e-5；CSDI先额外150轮去噪，随后冻结去噪器。',
          '逐率Test-best W-F1；同次预测计算ACC/MSE，不是单一checkpoint的部署成绩。','',
          '| 方法 | 八率ACC | 八率W-F1 | 高缺失W-F1 | 八率MSE | ΔW-F1 vs冻结父模型 |',
          '|---|---:|---:|---:|---:|---:|',
          f"| 冻结Flat | {base['mean']['ACC']:.3f} | {base['mean']['WF1']:.3f} | {base['high']['WF1']:.3f} | {base['mean']['MSE']:.4f} | 0 |"]
    for mode,r in runs.items():
        a=r['aggregate']
        rows.append(f"| {names[mode]} | {a['mean']['ACC']:.3f} | {a['mean']['WF1']:.3f} | {a['high']['WF1']:.3f} | {a['mean']['MSE']:.4f} | {a['mean']['WF1']-base['mean']['WF1']:+.3f} |")
    rows += ['', '八率W-F1最好的是NeuMiss，高缺失W-F1最好的是本轮Latent CSDI。',
             '四组相对同权重冻结父模型有正向Test-best差值，不能据此归因为特定机制有效。',
             '历史原Flat逐率分别选父权重的结果为ACC79.535、W-F179.460、高缺失W-F173.352；',
             '本轮四组都未超过该历史W-F1。历史行与本轮固定父权重的对照职责不同，不混成严格配对。','',
             '## 逐率W-F1与选中epoch','',
             '| Rate | 冻结Flat | NeuMiss | Signature | R2-D2 | CSDI |',
             '|---|---:|---:|---:|---:|---:|']
    for rate in first['baseline']:
        values=[f"{r['best_by_rate'][rate]['metrics']['WF1']:.3f} (e{r['best_by_rate'][rate]['epoch']})" for r in runs.values()]
        rows.append(f"| {rate} | {first['baseline'][rate]['WF1']:.3f} | "+' | '.join(values)+' |')
    rows += ['', '## 配对纠错/致错与资源','',
             '| 方法 | 纠错槽位 | 致错槽位 | 新增任务参数 | 去噪参数 | 墙钟秒 | 峰值分配MiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in runs.items():
        best=[b['metrics'] for b in r['best_by_rate'].values()]
        rows.append(f"| {names[mode]} | {sum(b['corrections'] for b in best)} | {sum(b['harms'] for b in best)} | {r['identity']['module_parameters']} | {r['identity']['extra_denoiser_parameters']} | {r['seconds']:.1f} | {r['peak_gpu_mib']:.1f} |")
    rows += ['', '纠错/致错相对各rate同一冻结父模型，跨rate累计会重复同一句话，不能视为独立样本。',
             '四进程同时在biggpu物理GPU0运行，时间受并发影响，不作为严格速度基准。','',
             '## CSDI同权重假设聚合控制','',
             '| 聚合 | 八率ACC | 八率W-F1 | 高缺失W-F1 |', '|---|---:|---:|---:|']
    for label,a in [('分别预测后平均',runs['csdi']['aggregate']),('先平均latent再预测',reload['csdi_latent_mean_aggregate'])]:
        rows.append(f"| {label} | {a['mean']['ACC']:.3f} | {a['mean']['WF1']:.3f} | {a['high']['WF1']:.3f} |")
    rows += ['', '同一假设、同一选中桥接权重的冻结干预；latent_mean没有重新训练，不等同独立模型消融。','',
             '## 实现与验证','',
             '- 原OSRAM/CHM、K/V、Query、Local Skip、Flat、Norm、任务头源码没有修改。',
             '- pre-LN anchor用只读hook捕获，原模型每个view前向一次；冻结Norm/head仍可对分支输入求梯度。',
             '- NeuMiss80维；Signature严格过去8个有效话语、零basepoint二阶Signature；',
             '  R2-D2仅TRAIN支持、同rate/组合、排除整段同对话、FP64求解；CSDI拟合/PCA仅TRAIN。',
             '- 15项合成测试及四组真实GPU smoke通过，初始零出口与父预测一致（Ridge有意非零）。',
             '- 四组150轮训练Mask/话语顺序hash全部一致；测试Mask/labels/ID/父预测逐值一致。',
             '- 32份选中预测重算分数并核对checkpoint/预测hash通过。',
             '- 四组选中rate0.3权重strict重载，预测最大绝对差均为0，极性完全一致。',
             '- 父权重摘要不变、父参数无梯度，优化器/进度/RNG完整保存。','',
             '## 范围与限制','',
             '用户描述的参考包并未提供或找到；按所贴公式实现前三项，其参数量与给定表吻合。',
             'CSDI内部去噪结构没有完整源码，因此本轮为自实现条件DDPM（310401参数），',
             '不是所述610753参数参考去噪器的精确复现。20步线性beta1e-4..0.02的终端',
             'alpha_bar较高，纯高斯初始先验近似是该配置的局限；未据测试结果改变日程。',
             '没有自动增加种子、调参、改变Missing或解冻父模型。全部为seed66的内部探索。','',
             '执行源码`d88495a`，分支`feature/ecerc-features`；后处理脚本独立保存。',
             '远端目录：`/data2/yb/remote_experiments/osram_four_block_mosi_20261025/execution_v5`。',
             '旧失败/作废尝试见INVALID_ATTEMPTS.json，未进入排名；所有旧权重、Mask及实验保留。',
             '机器可读逐率结果见RESULT.csv，完整配置/版本/环境/选点hash见completed/*.json。','']
    (root/'RESULT.md').write_text('\n'.join(rows))
    with (root/'RESULT.csv').open('w',newline='') as stream:
        fieldnames=['method','rate','epoch','ACC','WF1','MSE','corrections','harms','N']
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for rate,b in first['baseline'].items():writer.writerow(dict(method='Frozen Flat',rate=rate,epoch=75,**b))
        for mode,r in runs.items():
            for rate,b in r['best_by_rate'].items():writer.writerow(dict(method=names[mode],rate=rate,epoch=b['epoch'],**b['metrics']))


if __name__=='__main__':main()
