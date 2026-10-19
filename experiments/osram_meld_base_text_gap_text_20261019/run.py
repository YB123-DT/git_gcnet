"""MELD same-cohort four-condition Text-source readout replay."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=Path('/home/qinjinghui/remote_experiments/osram_12_source_evidence_audit_20261019/runs_v1/MELD')


def collect(bundle, cached):
    import torch
    from .audit import CONDITIONS,replay_four
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat,require_same_rng

    model,trainer=bundle['model'],bundle['trainer']
    loader=bundle['test_loader'];loader.sampler.set_epoch(0)
    chunks={};cursor=scans=0
    with torch.no_grad():
        for raw in loader:
            data=trainer._move_batch(raw,next(model.parameters()).device)
            valid=data[7].bool();meta=helper.metadata(data);n=len(meta['labels'])
            for key,value in meta.items():
                np.testing.assert_array_equal(value,cached[key][cursor:cursor+n])
            batch_av=torch.zeros((*valid.shape,3),device=valid.device)
            batch_av[valid]=torch.as_tensor(cached['availability'][cursor:cursor+n],device=valid.device,dtype=batch_av.dtype)
            av=batch_av.transpose(0,1)
            view=trainer._prepare_view_from_primary_masks(data,av,av,bundle['dimensions'])
            torch.testing.assert_close(view['availability'],av,rtol=0,atol=0)
            rng=helper.rng_state()
            with capture_flat(model) as captured:
                full=helper.forward(model,view)
            trace=model.osram.last_source_trace
            selected=valid.T & ~av[...,1].bool() & trace['trajectory'][...,1,3].bool()
            sources=torch.cat((trace['source_base'].unsqueeze(2),trace['source_gap']),2)
            predictions=replay_four(model.osram,model.smax_fc,captured['local'],
                captured['emotion_input'],sources,data[7],selected,full)
            require_same_rng(rng,helper.rng_state())
            flatten=lambda value:value.transpose(0,1)[valid].detach().cpu().numpy()
            values=dict(**meta,availability=flatten(av),trajectory=flatten(trace['trajectory']),
                        **{name:flatten(predictions[name]) for name in CONDITIONS})
            chosen=flatten(selected)
            np.testing.assert_array_equal(values['full'],cached['logits_full'][cursor:cursor+n])
            np.testing.assert_array_equal(values['trajectory'],cached['trajectory'][cursor:cursor+n])
            for name,row in (('without_base_text',0),('without_gap_text',2)):
                np.testing.assert_array_equal(values[name][chosen],cached['logits_removed'][cursor:cursor+n,row,1][chosen])
            for key,value in values.items():chunks.setdefault(key,[]).append(value)
            cursor+=n;scans+=1
    if cursor!=len(cached['labels']):raise RuntimeError('Incomplete original cached cohort')
    return {k:np.concatenate(v) for k,v in chunks.items()},dict(scans=scans,
        full_logits_cached_exact=True,single_removal_selected_logits_cached_exact=True,
        current_and_trajectory_identity_exact=True,rng_unchanged=True,
        same_cohort_for_all_four=True)


def write_report(output,report):
    from .audit import CONDITIONS
    labels={'full':'① 原模型','without_base_text':'② 删Base←T',
            'without_gap_text':'③ 删Gap-T←T','without_both':'④ 同时删除'}
    fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
    lines=['# MELD Base←T / Gap-T←T 同样本四条件分析','','INTERNAL DIAGNOSTIC ONLY',
        '',f"Source: `{report['source_commit']}`。seed66，固定原miss0.3 checkpoint，未训练。",
        '只纳入当前Text缺失且严格过去见过Text的有效话语。每个rate四条件用完全相同的样本。',
        '只移除指定Value来源项，不是删除整条Base/Gap，也不是删除原始历史Text；其他Gap的Text来源保留。',
        '原Local/Skip、Query、共享Memory读写和分类器不变。前三种预测在合格样本上逐位复现旧缓存。',
        '', '## ACC（%）','','| Rate | N | ① 原模型 | ② 删Base←T | ③ 删Gap-T←T | ④ 同时删除 |',
        '|---|---:|---:|---:|---:|---:|']
    rows=[]
    for rate,result in report['rates'].items():
        lines.append('| '+rate+' | '+str(result['N'])+' | '+' | '.join(fmt(result['metrics'][name]['ACC']) for name in CONDITIONS)+' |')
        for name in CONDITIONS:
            pair=result['paired_vs_full'].get(name,{})
            rows.append(dict(rate=rate,condition=name,**result['metrics'][name],
                             corrections=pair.get('corrections'),harms=pair.get('harms')))
    for metric in ('UA','WF1','CE'):
        lines+=['','## '+metric,'','| Rate | N | ① 原模型 | ② 删Base←T | ③ 删Gap-T←T | ④ 同时删除 |','|---|---:|---:|---:|---:|---:|']
        for rate,result in report['rates'].items():
            lines.append('| '+rate+' | '+str(result['N'])+' | '+' | '.join(fmt(result['metrics'][name][metric]) for name in CONDITIONS)+' |')
    lines+=['','## 条件边际贡献','','正数表示保留该项改善指标。CE列是“缺该项CE−有该项CE”；ACC列是“有该项ACC−缺该项ACC”。',
        '两项的非线性有限差分不是纯语义冗余或因果归因。','','| Rate | Gap有Base时ΔACC | Gap无Base时ΔACC | Gap有Base时CE改善 | Gap无Base时CE改善 | CE有限差分 |','|---|---:|---:|---:|---:|---:|']
    for rate,result in report['rates'].items():
        g=result['conditional_effects']
        values=[g['gap_with_base']['ACC_improvement'],g['gap_without_base']['ACC_improvement'],
                g['gap_with_base']['CE_improvement'],g['gap_without_base']['CE_improvement'],result['interaction_CE']]
        lines.append('| '+rate+' | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in values)+' |')
    lines+=['','## 纠错/致错（相对于①原模型）','','| Rate | 条件 | 错→对 | 对→错 |','|---|---|---:|---:|']
    for rate,result in report['rates'].items():
        for name,pair in result['paired_vs_full'].items():
            lines.append('| '+rate+' | '+labels[name]+' | '+str(pair['corrections'])+' | '+str(pair['harms'])+' |')
    lines+=['','## 口径边界','','- miss0没有合格话语，N/A不算零分；Persistent模式同样没有满足此筛选的样本，不增加空条件推理。',
        '- 不是完整测试集成绩，不替换三种子baseline；没有根据这些测试结果训练Gate或选择新超参数。',
        '- 不同rate可以选择不同目标话语，只保证每个rate内部严格配对。',
        '- 输入移除可能分布外；不把交互差值直接叫作纯Text信息冗余或新方法有效性。',
        '- 四组逐类Recall/support、当前A/V/AV分组、条件贡献和原始身份/Mask哈希保留在SUMMARY.json。',
        '- 逐样本预测/logits/Mask保留于独立远程NPZ，不上传权重、数据或大型预测。','']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    with (output/'METRICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    from .audit import CONDITIONS,eligible_mask,summarize_arrays
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    from experiments.osram_tasr_20261018.protocol import RATES
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'SUMMARY.json'
    if path.exists():raise RuntimeError('Independent output already used; inspect before any rerun')
    report=dict(status='running',source_commit=args.source_commit,no_training=True,rates={},error=None)
    helper.write_json(path,report);tick=time.monotonic()
    try:
        previous=json.loads((PREVIOUS/'SUMMARY.json').read_text())
        if previous['status']!='complete' or previous['source_commit']!='cb02830e794297e9d3654b5f0c9a4900c6ac0cdf':
            raise ValueError('Require completed original12-source audit')
        bundle=helper.setup_case('MELD',args.gpu_uuid,args.output)
        if previous['anchor']['checkpoint_sha256']!=bundle['case']['checkpoint_sha256']:
            raise ValueError('Wrong original fixed anchor')
        model=bundle['model'];digest=helper.backbone_digest(model)
        model.osram.osram_source_tracking=True;model.osram.osram_source_audit=True
        if model.osram.osram_tasr:raise ValueError('TASR must remain disabled')
        report.update(anchor=bundle['case'],environment=bundle['environment'],gpu=bundle['gpu'],
                      previous_summary_sha256=helper.sha(PREVIOUS/'SUMMARY.json'),
                      source_files={str(p.relative_to(ROOT)):helper.sha(p) for p in
                          (Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'gcnet_missing_m3/osram.py',ROOT/'gcnet_missing_m3/tasr.py')})
        torch.cuda.reset_peak_memory_stats()
        for rate in RATES:
            key='random_'+rate.replace('.','p')
            old=json.loads((PREVIOUS/(key+'.json')).read_text());old_npz=PREVIOUS/(key+'.npz')
            if helper.sha(old_npz)!=old['checks']['output_sha256']:raise ValueError('Old NPZ hash mismatch')
            with np.load(old_npz,allow_pickle=False) as z:
                cached={name:z[name].copy() for name in ('labels','speakers','conversation_ids','utterance_indices',
                    'availability','trajectory','logits_full','logits_removed')}
            chosen=eligible_mask(cached['availability'],cached['trajectory'])
            if not chosen.any():
                arrays={k:v for k,v in cached.items() if not k.startswith('logits_')}
                arrays.update({name:cached['logits_full'].copy() for name in CONDITIONS})
                checks=dict(scans=0,no_eligible_targets=True,same_cohort_for_all_four=True)
            else:
                arrays,checks=collect(bundle,cached)
            arrays['eligible']=chosen
            np.savez_compressed(args.output/(key+'.npz'),**arrays)
            result=summarize_arrays(arrays)
            result['checks']=dict(checks,cached_npz_sha256=old['checks']['output_sha256'],
                                 availability_sha256=old['checks']['availability_sha256'],
                                 output_sha256=helper.sha(args.output/(key+'.npz')))
            report['rates'][rate]=result;helper.write_json(path,report)
            print(json.dumps(dict(rate=rate,N=result['N'],ACC={n:result['metrics'][n]['ACC'] for n in CONDITIONS})),flush=True)
        if digest!=helper.backbone_digest(model) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen original model changed')
        report.update(status='complete',backbone_unchanged=True,seconds=time.monotonic()-tick,
                      peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        helper.write_json(path,report);write_report(args.output,report)
    except Exception as error:
        report.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        helper.write_json(path,report);raise


if __name__=='__main__':main()
