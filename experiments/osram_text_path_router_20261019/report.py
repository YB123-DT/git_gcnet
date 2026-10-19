"""Reports for one validation-selected Router, not per-rate test-best routing."""
import csv


def write_report(output,summary):
    case=summary['identity']['case'];dataset=summary['dataset']
    title='IEMOCAP-6' if dataset=='IEMOCAPSix' else dataset
    fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
    lines=[f'# {title} 条件化历史Text三路径 Router','','INTERNAL FROZEN-ANCHOR EXPERIMENT',
        '',f"Source `{summary['source_commit']}`。seed66 / fold{case['fold']}，原固定miss0.3 anchor epoch{case['epoch']}。",
        f"只训练 Router；新增{summary['router_parameters']:,}参数，验证集选中 Router epoch{summary['selected_epoch']}，所有rate共用此权重。",
        '原主干完全冻结；一次Memory扫描，完整/B保留/G保留三个概率候选；不平均logits。',
        '仅当前Text缺失且严格过去见过Text时路由，其他样本精确旁路原完整预测。',
        '原anchor曾按Test-oracle选中，本轮Router不按测试标签选epoch或调参；不是独立测试泛化成绩。',
        '', '## 全测试集八率等权均值','','| 模型 | ACC (%) | UA (%) | W-F1 (%) | CE |',
        '|---|---:|---:|---:|---:|']
    names=dict(full='冻结原Flat',candidate_B='固定B候选（只删Gap-T←T）',candidate_G='固定G候选（只删Base←T）',
               uniform='固定三候选均匀概率混合',router='学习Router')
    for tag,label in names.items():
        m=summary['aggregate'][tag]['mean']
        lines.append('| '+label+' | '+' | '.join(fmt(m[k]) for k in ('ACC','UA','WF1','CE'))+' |')
    lines+=['','## 高缺失0.5/0.6/0.7','','| 模型 | ACC (%) | UA (%) | W-F1 (%) | CE |','|---|---:|---:|---:|---:|']
    for tag,label in names.items():
        m=summary['aggregate'][tag]['high']
        lines.append('| '+label+' | '+' | '.join(fmt(m[k]) for k in ('ACC','UA','WF1','CE'))+' |')
    lines+=['','## 逐rate：完整测试集','','| Rate | N | Eligible | Flat ACC | Router ACC | ΔACC pp | Flat UA | Router UA | ΔUA pp | Flat WF1 | Router WF1 | Flat CE | Router CE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for rate,r in summary['evaluation'].items():
        f,g=r['full'],r['router']
        values=[f['ACC'],g['ACC'],g['ACC']-f['ACC'],f['UA'],g['UA'],g['UA']-f['UA'],f['WF1'],g['WF1'],f['CE'],g['CE']]
        lines.append('| '+rate+' | '+str(f['N'])+' | '+str(r['eligible_N'])+' | '+' | '.join(fmt(v) for v in values)+' |')
        for tag in names:
            rows.append(dict(rate=rate,model=tag,eligible_N=r['eligible_N'],**r[tag]))
    lines+=['','## 触发子集及纠错/致错','','只在同一批触发话语内比较；N=0为N/A。纠错/致错方向均为原Flat→Router。',
        '', '| Rate | N | Flat ACC | Router ACC | Flat UA | Router UA | Flat CE | Router CE | 错→对 | 对→错 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rate,r in summary['evaluation'].items():
        f,g=r['eligible']['full'],r['eligible']['router'];p=r['paired_eligible']
        values=[f['ACC'],g['ACC'],f['UA'],g['UA'],f['CE'],g['CE']]
        lines.append('| '+rate+' | '+str(r['eligible_N'])+' | '+' | '.join(fmt(v) for v in values)+' | '+
                     ('N/A' if p['corrections'] is None else str(p['corrections']))+' | '+
                     ('N/A' if p['harms'] is None else str(p['harms']))+' |')
    lines+=['','## 触发子集平均权重','','| Rate | Eligible | w_BG | w_B | w_G |','|---|---:|---:|---:|---:|']
    for rate,r in summary['evaluation'].items():
        mean=r['weights']['mean']
        lines.append('| '+rate+' | '+str(r['eligible_N'])+' | '+' | '.join(fmt(v) for v in (mean if mean is not None else [None]*3))+' |')
    lines+=['','## 资源、复现及解释边界','',
        f"- 含初始化、训练、验证和评估耗时{summary['seconds']:.2f}s；峰值allocated显存{summary['peak_memory_mib']:.2f}MiB。",
        '- 预测/mask/权重分布存独立远程NPZ；checkpoint保存Router、Adam、RNG、epoch和不可变source/config标识，不复制或修改原权重。',
        '- 所有rate使用固定同一主干和同一Router，不能与原三种子per-rate Test-best混作同一口径。',
        '- 固定均匀混合是初始化/不学习对照，不是额外训练实验；零初始化输出层使触发位置初始均匀，不保证原Flat恒等。',
        '- IEMOCAP仅fold5；Router-validation来自原训练对话，冻结主干曾见过这些对话，不声称干净的主干验证划分。',
        '- Persistent整段Text可见或从未出现Text均不满足触发条件，本结构必然旁路，不声称改善Persistent。',
        '- 观察到收益也不能直接宣称纯模态语义归因、情绪转折检测或结构新颖性；单种子结果需如实保留指标冲突。','']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    with (output/'METRICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
