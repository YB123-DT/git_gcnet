"""Keep completed metrics separate from partial progress and reuse protected controls."""
import csv
import hashlib
import json
import math
from pathlib import Path


def summarize(output,state,*,final=False):
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import write
    from experiments.osram_mosei_v3_capacity_20261022.worker import REFERENCE
    paths={j['mode']:Path(j['output']) for j in state['jobs']}
    paths.update(f256=Path(state['existing_flat256']),f1600=REFERENCE,
        f512=Path('/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f512_plain'),
        f1024=Path('/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/small_flat'))
    states,completed,rows={}, {}, []
    reference=json.loads((REFERENCE/'SUMMARY.json').read_text())
    for name,path in paths.items():
        file=path/'SUMMARY.json';s=json.loads(file.read_text()) if file.exists() else dict(status='pending')
        states[name]={k:s.get(k) for k in ('status','epoch','error','parameters','seconds','peak_memory_mib')}
        if s['status']!='complete':continue
        if s.get('epoch')!=60:raise ValueError('Incomplete epoch budget')
        validate_selected(path,s['best_by_rate'])
        cfg=s.get('config') or s['identity']['config']
        if cfg['learning_rate']!=.001 or cfg['missing_mask_protocol']!='gcnet-random-mask-v1':raise ValueError('Wrong control protocol')
        for key in ('split','seed_protocol','data_files'):
            if s['identity'][key]!=reference['identity'][key]:raise ValueError('Pair differs: '+key)
        masks=s['identity']['mask_hashes'];ref_masks=reference['identity']['mask_hashes']
        if masks.get('test',masks)!=ref_masks.get('test',ref_masks):raise ValueError('Different test masks')
        agg={scope:{metric:sum(s['evaluation'][rate]['router'][metric] for rate in rates)/len(rates)
             for metric in ('ACC','WF1','MSE')} for scope,rates in
             (('mean',[f'{i/10:.1f}' for i in range(8)]),('high',['0.5','0.6','0.7']))}
        for scope in agg:
            for metric in agg[scope]:
                if not math.isclose(agg[scope][metric],s['aggregate']['router'][scope][metric],abs_tol=1e-9):raise ValueError('Aggregate mismatch')
        completed[name]=dict(path=str(path),aggregate=agg,parameters=s.get('parameters'),
            summary_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        for rate,b in s['best_by_rate'].items():
            rows.append(dict(model=name,rate=rate,epoch=b['epoch'],**s['evaluation'][rate]['router']))
    if final and len(completed)!=11:raise ValueError('Require all8approved arms plus3reused controls')
    result=dict(label='INTERNAL TEST-ORACLE SCREEN',complete=final,states=states,runs=completed)
    output=Path(output);write(output/'REPORT.json',result)
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        fields=['model','rate','epoch','ACC','WF1','MSE'];writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    lines=['# MOSEI small Flat / Nested capacity screen','','INTERNAL TEST-ORACLE SCREEN',
        'Seed66,s0,LR1e-3,60epochs,new GCNet Missing. Only hidden width/Nested switch change.',
        'Seven new workers; Flat256 already running is observed, never duplicated. Three completed plain controls reused.',
        'Only completed runs appear in scores. Per-rate Test-best is not one-checkpoint deployment performance.','',
        '| Model | ACC | W-F1 | High W-F1 | Delta W-F1 vs same-width plain |', '|---|---:|---:|---:|---:|']
    for name,d in completed.items():
        a=d['aggregate'];plain=name.split('_n')[0]
        delta=a['mean']['WF1']-completed[plain]['aggregate']['mean']['WF1'] if '_n' in name and plain in completed else None
        cell=f'{delta:+.6f}' if delta is not None else 'N/A'
        lines.append(f"| {name} | {a['mean']['ACC']:.6f} | {a['mean']['WF1']:.6f} | {a['high']['WF1']:.6f} | {cell} |")
    lines+=['','## Progress','']+[f'- {name}: {row}' for name,row in states.items()]
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    if final:write(output/'RESULT.json',result)
