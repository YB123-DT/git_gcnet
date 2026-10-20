"""Final direct-source report includes the completed residual method control."""
import hashlib
import json
from pathlib import Path

from experiments.osram_coupled_source_role_20261020.report import write_report as common_report

RESIDUAL=Path('/data2/yb/remote_experiments/osram_coupled_source_role_20261020/execution_v1/CMUMOSI/runs/CMUMOSI/seed_66/joint')


def write_report(output,queue):
    common_report(output,queue,source_role_mode='direct',method_label='Direct Source Weighting')
    reference=RESIDUAL/'SUMMARY.json'
    residual=json.loads(reference.read_text())
    run=Path(queue['jobs'][0]['output'])
    new=json.loads((run/'SUMMARY.json').read_text())
    if residual['status']!='complete' or residual['epoch']!=150:raise ValueError('Require completed residual reference')
    for key in ('split','seed_protocol','mask_hashes'):
        if residual['identity'][key]!=new['identity'][key]:raise ValueError('Residual comparison '+key+' differs')
    if residual['initial_backbone_sha256']!=new['initial_backbone_sha256']:raise ValueError('Residual initialization differs')
    expected=dict(residual['identity']['config'])
    if expected.get('osram_source_role_mode','residual')!='residual':raise ValueError('Reference must be residual mode')
    expected['osram_source_role_mode']='direct'
    if expected!=new['identity']['config']:raise ValueError('Residual comparison training config differs')
    history_file=RESIDUAL/'history.json'
    old_history=json.loads(history_file.read_text())
    new_history=json.loads((run/'history.json').read_text())
    if len(old_history)!=150 or len(new_history)!=150:raise ValueError('Require complete paired histories')
    for a,b in zip(old_history,new_history):
        for key in ('epoch','mask_trace_sha256','optimizer_steps','learning_rates'):
            if a[key]!=b[key]:raise ValueError('Residual comparison training '+key+' differs')
    summary_path=Path(output)/'SUMMARY.json'
    summary=json.loads(summary_path.read_text())
    summary['residual_reference']=dict(path=str(RESIDUAL),
        summary_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(history_file.read_bytes()).hexdigest(),
        source_commit=residual.get('source_commit'),config_mode_only_differs=True,
        training_masks_updates_and_lr_exact=True,aggregate=residual['aggregate'])
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    lines=['','## Completed residual variant reference','',
        '| Method | 8-rate W-F1 | High W-F1 |','|---|---:|---:|']
    for name,s in (('Residual .25*tanh',residual),('Direct 2*sigmoid',new)):
        a=s['aggregate']['router'];lines.append(f"| {name} | {a['mean']['WF1']:.4f} | {a['high']['WF1']:.4f} |")
    lines+=['','This changes source-weight range (.75–1.25 versus0–2) AND numerical re-summation.',
        'It is NOT evidence that removing an algebraic residual alone causes any score change.']
    path=Path(output)/'RESULT.md';path.write_text(path.read_text()+'\n'.join(lines)+'\n')
