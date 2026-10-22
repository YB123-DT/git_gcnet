"""Completed-only capacity comparison; reporting never runs model inference."""
import csv
import json
from pathlib import Path

from .worker import MODES, RATES, REFERENCE

OLD_V3 = Path('/data2/yb/remote_experiments/osram_source_role_v3_20261021/execution_v1/runs/CMUMOSEI')


def summarize(root):
    root = Path(root); states = {}; complete = {}; rows = []; aggregate = []
    paths = dict(large_flat=REFERENCE, original_v3=OLD_V3,
                 **{mode:root/'runs'/mode for mode in MODES})
    for name,directory in paths.items():
        p = directory/'SUMMARY.json'
        s = json.loads(p.read_text()) if p.exists() else dict(status='not_started')
        states[name] = {k:s.get(k) for k in ('status','epoch','error','seconds','peak_memory_mib','parameters')}
        if s.get('status')!='complete':
            continue
        if s.get('epoch')!=60 or set(s.get('evaluation',{}))!=set(RATES):
            raise ValueError('Incomplete final evaluation: '+str(p))
        complete[name]=s
        for rate in RATES:
            metric=s['evaluation'][rate]['router']
            rows.append(dict(model=name,rate=rate,epoch=s['best_by_rate'][rate]['epoch'],
                             **{k:metric[k] for k in ('ACC','WF1','MSE')}))
        for scope,rates in [('mean',RATES),('high',RATES[5:])]:
            aggregate.append(dict(model=name,scope=scope,
                **{k:sum(s['evaluation'][r]['router'][k] for r in rates)/len(rates)
                   for k in ('ACC','WF1','MSE')}))
    if set(MODES)<=complete.keys():
        a,b=[complete[k]['identity'] for k in MODES]
        for key in ('source_commit','mask_hashes','split','seed_protocol','data_files','small_flat_initial_sha256'):
            if a[key]!=b[key]:
                raise ValueError('Same-round capacity pair differs: '+key)
        aa,bb=dict(a['config']),dict(b['config'])
        for key in ('osram_source_role_v3','osram_source_role_v3_dim','osram_source_role_v3_pair_hidden'):
            aa.pop(key);bb.pop(key)
        if aa!=bb:
            raise ValueError('Same-round configs differ beyond V3 capacity')
    result=dict(label='INTERNAL TEST-ORACLE SCREEN',states=states,aggregate=aggregate,per_rate=rows)
    (root/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (root/'PER_RATE.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# MOSEI smaller Flat / wider V3, seed66','', 'INTERNAL TEST-ORACLE SCREEN','',
        'Two fresh joint60epoch runs, Flat hidden1024, unchanged output1600/CHM8x64.',
        'Wide V3: width256, PairMLP512, same four heads and original hierarchy.',
        'Historical large Flat/default V3 are references, not fresh same-source training.',
        'Only completed runs enter score tables. No inference is performed by reporting.','',
        '| Model | Scope | ACC | W-F1 | MSE |','|---|---|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['model']} | {r['scope']} | {r['ACC']:.3f} | {r['WF1']:.3f} | {r['MSE']:.6f} |")
    lines+=['','## State','']+[f'- {name}: {s}' for name,s in states.items()]
    lines+=['','## Per rate','', '| Model | Rate | Epoch | ACC | W-F1 | MSE |',
            '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['rate']} | {r['epoch']} | {r['ACC']:.3f} | {r['WF1']:.3f} | {r['MSE']:.6f} |")
    lines+=['','No multi-seed/independent-generalization claim. Per-rate Test-best may select different epochs.',
            'Shrinking Flat alone and adding wider V3 are distinguished by the new small_flat control.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
