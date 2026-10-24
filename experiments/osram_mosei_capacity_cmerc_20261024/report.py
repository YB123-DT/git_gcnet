"""Matched cmerc control only; never import s0 scores into this comparison."""
import csv
import hashlib
import json
import math
from pathlib import Path

CONTROL=Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none')
_CACHE={}


def validate_pair(s,history,baseline,baseline_history):
    cfg,old=s['config'],baseline['config']
    for key in set(cfg)|set(old):
        if key not in ('osram_adapter_hidden_dim','osram_meaningful_block') and cfg.get(key)!=old.get(key):
            raise ValueError('Configuration not paired: '+key)
    if (cfg['learning_rate']!=1e-4 or cfg['epochs']!=60
            or cfg['missing_mask_protocol']!='gcnet-random-mask-v1'):
        raise ValueError('Require original cmerc1e-4 protocol')
    if s['environment']!=baseline['environment']:
        raise ValueError('Environment mismatch; s0 scores are not controls')
    if s['identity']['rng_repair']!=baseline['identity']['rng_repair']:
        raise ValueError('CUDA RNG handling differs')
    for key in ('split','mask_reference'):
        if s['identity'][key]!=baseline['identity'][key]:raise ValueError('Pair differs: '+key)
    if s['identity']['data']['files']!=baseline['identity']['data']['files']:
        raise ValueError('Actual feature files differ')
    for key,value in baseline['identity']['source_hashes'].items():
        if key.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')) and key!='gcnet_missing_m3/meaningful_new40_registry.py':
            if s['identity']['source_hashes'].get(key)!=value:raise ValueError('Numerical source differs: '+key)
    if len(history)!=60 or len(baseline_history)!=60:raise ValueError('Incomplete training budget')
    for row,reference in zip(history,baseline_history):
        for key in ('epoch','mask_trace_sha256','optimizer_steps','learning_rates'):
            if row[key]!=reference[key]:raise ValueError('Training trace differs: '+key)
        if row['learning_rates']!=[1e-4]:raise ValueError('No local LR/schedule change authorized')


def summarize(output,state,*,final=False):
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import write
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    baseline=json.loads((CONTROL/'SUMMARY.json').read_text())
    baseline_history=json.loads((CONTROL/'history.json').read_text())
    paths={j['mode']:Path(j['output']) for j in state['jobs']};paths['f1600']=CONTROL
    complete,states,rows={}, {}, []
    for name,path in paths.items():
        file=path/'SUMMARY.json';s=json.loads(file.read_text()) if file.exists() else dict(status='pending')
        states[name]={k:s.get(k) for k in ('status','epoch','error','total_parameters','nested_parameters','seconds','peak_memory_mib')}
        if s['status']!='complete':continue
        signature=(file.stat().st_mtime_ns,file.stat().st_size)
        if not final and _CACHE.get(str(path),{}).get('signature')==signature:
            item,per_rate=_CACHE[str(path)]['item'],_CACHE[str(path)]['rows']
        else:
            if s.get('epoch')!=60:raise ValueError('Incomplete formal result')
            validate_selected(path,s['best_by_rate'])
            history=json.loads((path/'history.json').read_text())
            validate_pair(s,history,baseline,baseline_history)
            agg=s['aggregate']['router']
            for scope,rates in (('mean',[f'{i/10:.1f}' for i in range(8)]),('high',['0.5','0.6','0.7'])):
                for metric in ('ACC','WF1','MSE'):
                    value=sum(s['evaluation'][r]['router'][metric] for r in rates)/len(rates)
                    if not math.isclose(value,agg[scope][metric],abs_tol=1e-9):raise ValueError('Aggregate mismatch')
            item=dict(path=str(path),aggregate=agg,config=s['config'],initial=s['initial_backbone_sha256'],
                total_parameters=s['total_parameters'],nested_parameters=s['nested_parameters'],
                summary_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            per_rate=[dict(model=name,rate=r,epoch=b['epoch'],**s['evaluation'][r]['router']) for r,b in s['best_by_rate'].items()]
            _CACHE[str(path)]=dict(signature=signature,item=item,rows=per_rate)
        complete[name]=item;rows.extend(per_rate)
    if final and len(complete)!=9:raise ValueError('Require8newcmerc arms plus1pairedFlat1600 control')
    lines=['# MOSEI cmerc capacity comparison','','INTERNAL TEST-ORACLE SCREEN',
        'All scores use cmerc-repro, constantLR1e-4,60epochs,seed66,newGCNetMissing and matching CUDA RNG handling.',
        'Eight fresh runs; no s0 results or weights reused. Flat1600 is the completed matched cmerc control.',
        'No matching cmerc plainFlat512 was authorized; its same-width delta is N/A, not borrowed from s0.',
        'Per-rate TEST-best W-F1 can use different epochs; not a single-checkpoint deployment score.','',
        '| Model | ACC | W-F1 | High W-F1 | Delta WF1 vs Flat1600 | Delta WF1 vs same width |',
        '|---|---:|---:|---:|---:|---:|']
    for name,item in complete.items():
        a=item['aggregate'];plain=name.split('_n')[0]
        delta=a['mean']['WF1']-baseline['aggregate']['router']['mean']['WF1']
        same=None
        if '_n' in name and plain in complete:
            if item['initial']!=complete[plain]['initial']:raise ValueError('Same-width original initialization differs')
            same=a['mean']['WF1']-complete[plain]['aggregate']['mean']['WF1']
        cell=f'{same:+.6f}' if same is not None else 'N/A'
        lines.append(f"| {name} | {a['mean']['ACC']:.6f} | {a['mean']['WF1']:.6f} | {a['high']['WF1']:.6f} | {delta:+.6f} | {cell} |")
    lines+=['','## Progress','']+[f'- {name}: {row}' for name,row in states.items()]
    output=Path(output);(output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    result=dict(label='INTERNAL TEST-ORACLE SCREEN',complete=final,states=states,runs=complete)
    write(output/'REPORT.json',result)
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','rate','epoch','ACC','WF1','MSE']);writer.writeheader();writer.writerows(rows)
    if final:write(output/'RESULT.json',result)
