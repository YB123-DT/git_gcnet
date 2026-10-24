"""Complete-only MOSEI combination report against preserved matched controls."""
import csv
import hashlib
import json
from pathlib import Path

from experiments.osram_nested_cmerc_tuning_20261023.worker import MOSEI_COMBINATIONS, RATES

CONTROL_ROOT = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66')
CONTROL_METHODS = ('none','nested_dim32','mosei_nested_mlp128')
REGISTRY = 'gcnet_missing_m3/meaningful_new40_registry.py'
OLD_SOURCE = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/source_v2')


def validate_source_extension(root, old_root=OLD_SOURCE):
    root, old_root = Path(root), Path(old_root)
    removed = []
    lines = []
    for line in (root / REGISTRY).read_text().splitlines(keepends=True):
        if any("'"+name+"':" in line for name in MOSEI_COMBINATIONS):
            removed.append(line)
        else:
            lines.append(line)
    if len(removed) != 4 or ''.join(lines) != (old_root / REGISTRY).read_text():
        raise ValueError('Registry differs beyond the four authorized definitions')
    checked = 0
    for package in ('gcnet_missing_m3','gcnet_modality_jepa'):
        old_files = {p.name for p in (old_root/package).glob('*.py')}
        if old_files != {p.name for p in (root/package).glob('*.py')}:
            raise ValueError('Unexpected numerical source file addition/removal')
        for name in old_files:
            relative = f'{package}/{name}'
            if relative == REGISTRY:
                continue
            if (root/relative).read_bytes() != (old_root/relative).read_bytes():
                raise ValueError('Numerical source changed: '+relative)
            checked += 1
    return dict(unchanged_numerical_files=checked, registry_change='four named configuration entries only',
        old_registry_sha256=hashlib.sha256((old_root/REGISTRY).read_bytes()).hexdigest(),
        new_registry_sha256=hashlib.sha256((root/REGISTRY).read_bytes()).hexdigest())


def validate_pair(candidate, history, baseline, base_history):
    if candidate['initial_backbone_sha256'] != baseline['initial_backbone_sha256']:
        raise ValueError('Original initialization differs')
    normalize = lambda cfg: {k:v for k,v in cfg.items() if k != 'osram_meaningful_block'}
    if normalize(candidate['config']) != normalize(baseline['config']):
        raise ValueError('Training configuration differs beyond module switch')
    for key in ('mask_reference','split'):
        if candidate['identity'][key] != baseline['identity'][key]:
            raise ValueError('Pair differs: '+key)
    sources = lambda s: {k:v for k,v in s['identity']['source_hashes'].items()
        if k.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')) and k != REGISTRY}
    if sources(candidate) != sources(baseline):
        raise ValueError('Numerical model/sampler source differs')
    if len(history) != len(base_history):
        raise ValueError('Training budget differs')
    for old,new in zip(base_history,history):
        for name in ('epoch','mask_trace_sha256','optimizer_steps','learning_rates'):
            if old[name] != new[name]:
                raise ValueError('Training trace differs: '+name)


def write_report(output, state):
    jobs = state['jobs']
    if len(jobs)!=4 or {j['method'] for j in jobs}!=set(MOSEI_COMBINATIONS) or any(j['status']!='complete' for j in jobs):
        raise ValueError('Require all four complete authorized combinations; no partial ranking')
    from .run import ROOT
    from experiments.osram_nested_cmerc_tuning_20261023.run import validate_summary
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import validate_selected
    proof = validate_source_extension(ROOT)
    rows, runs = [], []
    baseline, base_history = None, None
    for method, path, job in (
        [(m,CONTROL_ROOT/m,None) for m in CONTROL_METHODS] +
        [(j['method'],Path(j['output']),j) for j in jobs]):
        s = validate_summary(job) if job else json.loads((path/'SUMMARY.json').read_text())
        if (s['status'],s['epoch'],s['seed'],s['dataset'],s['fold']) != ('complete',60,66,'CMUMOSEI',1):
            raise ValueError('Require complete same-protocol controls')
        validate_selected(path,s['best_by_rate'])
        history = json.loads((path/'history.json').read_text())
        if len(history)!=60:
            raise ValueError('Require60epochs of history')
        if method == 'none':
            baseline,base_history = s,history
        validate_pair(s,history,baseline,base_history)
        for scope,rates in [('mean',RATES),('high',RATES[5:])]:
            for metric in ('ACC','WF1','MSE'):
                actual = sum(s['evaluation'][r]['router'][metric] for r in rates)/len(rates)
                if abs(actual-s['aggregate']['router'][scope][metric]) > 1e-9:
                    raise ValueError('Aggregate differs from selected predictions')
        for rate in RATES:
            row=s['best_by_rate'][rate]
            if not row['same_evaluation_call'] or any(
                s['evaluation'][rate].get(key) != value for key,value in row['metrics'].items()
            ):
                raise ValueError('Selected metrics do not match publication')
            rows.append(dict(method=method,rate=rate,epoch=row['epoch'],**{
                k:s['evaluation'][rate]['router'][k] for k in ('ACC','WF1','MSE')}))
        runs.append(dict(method=method,path=str(path),aggregate=s['aggregate']['router'],
            total_parameters=s['total_parameters'],nested_parameters=s['nested_parameters'],
            seconds=s['seconds'],peak_memory_mib=s['peak_memory_mib'],
            source_commit=s['source_commit'],summary_sha256=hashlib.sha256((path/'SUMMARY.json').read_bytes()).hexdigest()))
    lines=['# MOSEI Nested combinations','','INTERNAL TEST-ORACLE SCREEN; seed66 only.',
        'Same cmerc-repro, fixed accepted masks,60epochs,constant1e-4,batch32,WD1e-5,MSE.',
        'Controls are reused read-only. Numerical sources identical except verified registry-only additions.',
        'No independent generalization or statistical-significance claim. No automatic promotion.','',
        '| Method | ACC | ΔACC pp | W-F1 | ΔWF1 pp | High WF1 | Nested params |',
        '|---|---:|---:|---:|---:|---:|---:|']
    base = runs[0]['aggregate']['mean']
    for r in runs:
        m,h=r['aggregate']['mean'],r['aggregate']['high']
        lines.append(f"| {r['method']} | {m['ACC']:.3f} | {m['ACC']-base['ACC']:+.3f} | {m['WF1']:.3f} | {m['WF1']-base['WF1']:+.3f} | {h['WF1']:.3f} | {r['nested_parameters']} |")
    output=Path(output)
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (output/'RESULT.json').write_text(json.dumps(dict(status='complete',label='INTERNAL TEST-ORACLE SCREEN',
        source_commit=state['source_commit'],source_extension_proof=proof,runs=runs),indent=2,sort_keys=True)+'\n')
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
