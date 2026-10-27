"""Independently verify selected variants, with actual architecture/counts."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .protocol import CONFIGS,EPOCHS,LR,RATES,reference_dir,sha,validate_epoch_trace,validate_reference


def aggregates(rows):
    result={}
    for config in ('flat','default3e4',*CONFIGS):
        group=[r for r in rows if r['config']==config]
        if len(group)!=8 or {r['rate'] for r in group}!=set(RATES):continue
        result[config]={name:{key:statistics.mean(r[key] for r in group if r['rate'] in rates)
            for key in ('ACC','WF1','MSE')} for name,rates in [('mean',RATES),('high',RATES[5:])]}
    return result


def validate_selected_metadata(summary,job,source):
    cfg=CONFIGS[job['config']];identity=summary['identity']
    expected=dict(source_commit=source,config=job['config'],method=cfg['method'],dataset='CMUMOSEI',seed=66,
        fold=1,stage='frozen',smoke=False,evaluation_rate=job['rate'],learning_rate=LR,
        architecture={k:cfg[k] for k in ('dim','depth','mlp_hidden','groups')})
    if (summary['status']!='complete' or summary['epoch']!=EPOCHS or summary['total_parameters']!=cfg['parameters']
        or not summary['checks'] or not all(summary['checks'].values())
        or any(identity.get(k)!=v for k,v in expected.items())):raise ValueError('Wrong selected variant/config/source/budget')


def selected_row(job,source):
    import numpy as np
    from experiments.osram_frozen_nested_cross_20261025.summarize import protected_arrays
    from experiments.osram_frozen_nested_cross_20261025.protocol import measured_metrics
    output=Path(job['output']).resolve();summary=json.loads((output/'SUMMARY.json').read_text())
    validate_selected_metadata(summary,job,source)
    identity,best=summary['identity'],summary['best'];rate=job['rate']
    reference=json.loads((reference_dir(rate)/'SUMMARY.json').read_text())
    validate_reference(reference,identity['parent'],identity['effective_stage2_config'],identity['test_mask_sha256'],rate)
    history=json.loads((output/'history.json').read_text());old=json.loads((reference_dir(rate)/'history.json').read_text())
    if [r['epoch'] for r in history]!=list(range(1,EPOCHS+1)):raise ValueError('Incomplete formal history')
    for row,original in zip(history,old):validate_epoch_trace(row,original,LR)
    if max(history,key=lambda r:r['test']['WF1'])['epoch']!=best['epoch']:raise ValueError('Wrong earliest WF1 selected epoch')
    for key,digest in [('checkpoint','checkpoint_sha256'),('predictions','prediction_sha256')]:
        path=(output/best[key]).resolve()
        if not path.is_relative_to(output) or sha(path)!=best[digest]:raise ValueError('Selected artifact path/hash differs')
    artifact=summary['baseline_artifact'];basepath=(output/artifact['path']).resolve()
    if not basepath.is_relative_to(output) or sha(basepath)!=artifact['sha256']:raise ValueError('Baseline artifact differs')
    with np.load(output/best['predictions'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    with np.load(basepath,allow_pickle=False) as data:baseline={k:data[k].copy() for k in data.files}
    protected=protected_arrays(identity['parent'],rate)
    for key in ('labels','availability','conversation_ids','utterance_indices'):
        np.testing.assert_array_equal(arrays[key],baseline[key]);np.testing.assert_array_equal(arrays[key],protected[key])
    if hashlib.sha256(arrays['availability'].tobytes()).hexdigest()!=identity['test_mask_sha256']:raise ValueError('Protected mask differs')
    original=protected['prediction'] if 'prediction' in protected else protected['logits']
    np.testing.assert_array_equal(arrays['base_prediction'],original.reshape(-1))
    np.testing.assert_array_equal(baseline['prediction'],arrays['base_prediction'])
    score=measured_metrics('CMUMOSEI',arrays['labels'],arrays['prediction'],arrays['base_prediction'])
    flat=measured_metrics('CMUMOSEI',arrays['labels'],arrays['base_prediction'],arrays['base_prediction'])
    for key,value in score.items():np.testing.assert_allclose(value,best['metrics'][key],rtol=0,atol=1e-5)
    for key,value in flat.items():np.testing.assert_allclose(value,summary['baseline'][key],rtol=0,atol=1e-5)
    row=dict(config=job['config'],rate=rate,seed=66,selected_epoch=best['epoch'],learning_rate=LR,
        added_parameters=summary['total_parameters'],full_parameters=summary['full_parameters'],
        seconds=summary['seconds'],peak_memory_mib=summary['peak_memory_mib'],
        predictions=str(output/best['predictions']),parent=identity['parent']['checkpoint'],
        corrections=score['corrections'],harms=score['harms'],N=score['N'])
    row.update({k:score[k] for k in ('ACC','WF1','MSE')})
    return row


def summarize(root,source,jobs):
    from experiments.osram_frozen_nested_cross_20261025.summarize import selected_row as default_row
    from experiments.osram_mosei_frozen_nested_lr_20261027.summarize import flat_row
    root=Path(root);rows=[]
    for rate in RATES:
        output=reference_dir(rate);reference=json.loads((output/'SUMMARY.json').read_text())
        row,_=default_row(dict(output=str(output),dataset='CMUMOSEI',fold=1,rate=rate),reference['identity']['source_commit'])
        rows.append(dict(row,config='default3e4',learning_rate=LR))
        rows.append(flat_row(row,reference['identity']['parent']['epoch'],output/reference['baseline_artifact']['path']))
    if len(jobs)!=24 or any(j['status']!='complete' for j in jobs):raise ValueError('Cannot rank unfinished capacity screen')
    rows.extend(selected_row(job,source) for job in jobs)
    aggregate=aggregates(rows)
    if set(aggregate)!={'flat','default3e4',*CONFIGS}:raise ValueError('Missing/duplicated eight-rate groups')
    result=dict(status='complete',source_commit=source,completed_new_runs=24,per_rate=rows,aggregate=aggregate,
        label='INTERNAL per-rate Test-oracle WF1; seed66 only',scope='One Nested structure factor changes; parent frozen, LR3e-4 constant60epochs')
    lines=['# MOSEI66 frozen Nested capacity screen','',result['label'],'',
        '| Variant | Added params | Eight-rate ACC | Eight-rate W-F1 | High ACC | High W-F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for config,values in aggregate.items():
        parameters=next(r['added_parameters'] for r in rows if r['config']==config)
        lines.append(f"| {config} | {parameters} | {values['mean']['ACC']:.3f} | {values['mean']['WF1']:.3f} | {values['high']['ACC']:.3f} | {values['high']['WF1']:.3f} |")
    lines+=['','| Rate | Config | ACC | W-F1 | Selected epoch |','|---|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['rate']} | {row['config']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['selected_epoch']} |")
    lines+=['','Only Nested architecture changes; CHM,Flat,mask and training budget are fixed.',
        'dim32 explicitly keeps MLP64; registered plain dim32 would also shrink MLP and is not used.',
        'Initial logits equal the protected parent, but changed-shape parameter tensors are not claimed identical to default.',
        'Default-dimension Sweep and original Nested were verified numerically equivalent with opened decoders.',
        'Eight-rate means combine8 own-rate parent/branch pairs. No independent generalization or significance claim.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (root/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=sorted({k for r in rows for k in r if not isinstance(r[k],list)})
    with (root/'RESULT.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return result
