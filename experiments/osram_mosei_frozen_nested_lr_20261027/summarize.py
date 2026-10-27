"""Recompute only complete selected runs; default3e-4 remains protected."""
import csv
import json
from pathlib import Path
import statistics

from .protocol import EPOCHS,LRS,PARAMETERS,RATES,reference_dir,sha,validate_epoch_trace,validate_reference


def flat_row(nested,parent_epoch,predictions):
    row=dict(nested,config='flat',learning_rate=None,ACC=nested['Flat_ACC'],
        WF1=nested['Flat_WF1'],MSE=nested['Flat_MSE'],added_parameters=0,
        full_parameters=nested['full_parameters']-PARAMETERS,corrections=0,harms=0,
        selected_epoch=parent_epoch,predictions=str(predictions),seconds=None,peak_memory_mib=None)
    row.update({k:0. for k in row if k.startswith('delta_Flat_')})
    return row


def aggregates(rows):
    results={}
    for config in ('flat','default3e4',*LRS):
        chosen=[r for r in rows if r['config']==config]
        if len(chosen)!=8 or {r['rate'] for r in chosen}!=set(RATES):continue
        results[config]={name:{k:statistics.mean(r[k] for r in chosen if r['rate'] in rates)
            for k in ('ACC','WF1','MSE')} for name,rates in [('mean',RATES),('high',RATES[5:])]}
    return results


def summarize(root,source,jobs):
    from experiments.osram_frozen_nested_cross_20261025.summarize import selected_row
    root=Path(root);rows=[]
    for rate in RATES:
        output=reference_dir(rate)
        ref=json.loads((output/'SUMMARY.json').read_text())
        job=dict(output=str(output),dataset='CMUMOSEI',fold=1,rate=rate)
        row,_=selected_row(job,ref['identity']['source_commit'])
        rows.append(dict(row,config='default3e4',learning_rate=3e-4))
        rows.append(flat_row(row,ref['identity']['parent']['epoch'],output/ref['baseline_artifact']['path']))
    for job in jobs:
        if job['status']!='complete':raise ValueError('Cannot rank unfinished LR runs')
        output=Path(job['output'])
        summary=json.loads((output/'SUMMARY.json').read_text());identity=summary['identity']
        lr=LRS[job['config']]
        if identity['config']!=job['config'] or identity['learning_rate']!=lr:raise ValueError('Wrong branch LR identity')
        ref_path=reference_dir(job['rate']);ref=json.loads((ref_path/'SUMMARY.json').read_text())
        validate_reference(ref,identity['parent'],identity['effective_stage2_config'],identity['test_mask_sha256'],job['rate'])
        if identity['initial_block_sha256']!=ref['identity']['initial_block_sha256']:raise ValueError('Initial Nested differs')
        history=json.loads((output/'history.json').read_text())
        original=json.loads((ref_path/'history.json').read_text())
        if len(history)!=EPOCHS:raise ValueError('Incomplete history')
        for row,old in zip(history,original):validate_epoch_trace(row,old,lr)
        check=dict(job,dataset='CMUMOSEI',fold=1)
        row,_=selected_row(check,source)
        rows.append(dict(row,config=job['config'],learning_rate=lr))
    result=dict(status='complete',source_commit=source,completed_new_runs=len(jobs),per_rate=rows,
        aggregate=aggregates(rows),label='INTERNAL per-rate Test-oracle WF1; seed66 only',
        scope='Only branch LR changes; own-rate original Flat frozen, cyclic training,60epochs')
    if len(jobs)!=16 or not all(k in result['aggregate'] for k in ('flat','default3e4',*LRS)):
        raise ValueError('Finite sixteen-run comparison incomplete')
    lines=['# MOSEI66 frozen Nested learning-rate screen','',result['label'],
        'No new Flat training; each rate retains its own protected parent and exact test mask.',
        '', '| Model | Eight-rate ACC | Eight-rate W-F1 | High-missing ACC | High-missing W-F1 |',
        '|---|---:|---:|---:|---:|']
    for config,values in result['aggregate'].items():
        lines.append(f"| {config} | {values['mean']['ACC']:.3f} | {values['mean']['WF1']:.3f} | {values['high']['ACC']:.3f} | {values['high']['WF1']:.3f} |")
    lines+=['','| Rate | Config | ACC | W-F1 | Best epoch |','|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['rate']} | {row['config']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['selected_epoch'] if row['config']!='flat' else 'parent'} |")
    lines+=['','Same initialization, training sample/mask/update traces, and parents were verified.',
        'All new selected checkpoints/predictions and metrics were independently validated.',
        'Eight-rate means combine8 own-rate parent/branch pairs, not one deployment checkpoint.',
        'Single-seed test-oracle tuning is exploratory; no automatic seed or architecture expansion.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (root/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=sorted({k for r in rows for k in r if not isinstance(r[k],list)})
    with (root/'RESULT.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return result
