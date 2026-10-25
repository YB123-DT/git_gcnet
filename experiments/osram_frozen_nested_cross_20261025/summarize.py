"""Recompute selected metrics, keeping rate/fold and pooled OOF distinct."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from experiments.osram_three_block_joint_mosi_20261025.scheduler import save
from .protocol import budget, measured_metrics, selection_metric

RATES = tuple(str(i/10) for i in range(8))
DATASETS = ('CMUMOSEI','MELD','IEMOCAPSix')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_arrays(parent,rate):
    """Return the protected parent selection, never a newly sampled mask."""
    import numpy as np
    if digest(parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('Protected parent checkpoint changed')
    if 'selected_prediction' in parent:
        path,expected = parent['selected_prediction'],parent['selected_prediction_sha256']
    else:
        root = Path(parent['parent_path'])
        if digest(root/'SUMMARY.json') != parent['summary_sha256']:
            raise ValueError('Protected parent summary changed')
        selected = json.loads((root/'SUMMARY.json').read_text())['best_by_rate'][rate]
        path,expected = root/selected['predictions'],selected['prediction_sha256']
    if digest(path) != expected:raise ValueError('Protected selected prediction changed')
    with np.load(path,allow_pickle=False) as stored:return {k:stored[k].copy() for k in stored.files}


def selected_row(job,source):
    import numpy as np
    output = Path(job['output']).resolve()
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    dataset,fold,rate = job['dataset'],job['fold'],job['rate']
    if (summary['status'] != 'complete' or summary['epoch'] != budget(dataset)
        or identity['source_commit'] != source or identity['method'] != 'nested_gnn_rooted_evidence'
        or identity['dataset'] != dataset or identity['seed'] != 66 or identity['fold'] != fold
        or identity['stage'] != 'frozen' or identity['smoke'] or identity['evaluation_rate'] != rate
        or summary['total_parameters'] != 159235 or not summary['checks'] or not all(summary['checks'].values())):
        raise ValueError('Incomplete/mismatched epoch, source, parent or correctness evidence')
    history = json.loads((output/'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1,budget(dataset)+1)):
        raise ValueError('Require full contiguous training history')
    best = summary['best'];metric = selection_metric(dataset)
    if max(history,key=lambda r:r['test'][metric])['epoch'] != best['epoch']:
        raise ValueError('Selected epoch differs from earliest strict own-rate Test-best')
    for name,sha in [('checkpoint','checkpoint_sha256'),('predictions','prediction_sha256')]:
        path = (output/best[name]).resolve()
        if not path.is_relative_to(output) or digest(path) != best[sha]:
            raise ValueError('Selected artifact hash/path differs')
    baseline = summary['baseline_artifact']
    if digest(output/baseline['path']) != baseline['sha256']:
        raise ValueError('Saved baseline changed')
    with np.load(output/best['predictions'],allow_pickle=False) as stored:
        arrays = {k:stored[k].copy() for k in stored.files}
    with np.load(output/baseline['path'],allow_pickle=False) as stored:
        original = {k:stored[k].copy() for k in stored.files}
    protected = protected_arrays(identity['parent'],rate)
    for key in ('labels','availability','conversation_ids','utterance_indices'):
        np.testing.assert_array_equal(arrays[key],original[key])
        np.testing.assert_array_equal(arrays[key],protected[key])
    if digest(output/best['predictions']) != best['prediction_sha256']:
        raise ValueError('Selected arrays changed while reading')
    if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != identity['test_mask_sha256']:
        raise ValueError('Selected test mask hash differs')
    np.testing.assert_array_equal(arrays['base_prediction'],original['base_prediction'])
    np.testing.assert_array_equal(original['prediction'],original['base_prediction'])
    prediction = protected['prediction'] if 'prediction' in protected else protected['logits']
    if dataset == 'CMUMOSEI':prediction = prediction.reshape(-1)
    np.testing.assert_array_equal(arrays['base_prediction'],prediction)
    score = measured_metrics(dataset,arrays['labels'],arrays['prediction'],arrays['base_prediction'])
    flat = measured_metrics(dataset,arrays['labels'],arrays['base_prediction'],arrays['base_prediction'])
    for key,value in score.items():
        np.testing.assert_allclose(value,best['metrics'][key],rtol=0,atol=1e-5)
    for key,value in flat.items():
        np.testing.assert_allclose(value,summary['baseline'][key],rtol=0,atol=1e-5)
    row = dict(dataset=dataset,seed=66,fold=fold,rate=rate,selected_epoch=best['epoch'],
        added_parameters=159235,full_parameters=summary['full_parameters'],
        seconds=summary['seconds'],peak_memory_mib=summary['peak_memory_mib'],
        corrections=score['corrections'],harms=score['harms'],N=score['N'],
        predictions=str(output/best['predictions']),parent=identity['parent']['checkpoint'])
    for key in ('ACC','UA','WF1','CE','MSE'):
        if key in score:row.update({key:score[key],'Flat_'+key:flat[key],'delta_Flat_'+key:score[key]-flat[key]})
    for key in ('recall','support'):
        if key in score:row[key] = score[key]
    return row,arrays


def summarize(root,source,jobs):
    import numpy as np
    root,rows,errors = Path(root),[],[]
    arrays_by_key = {}
    for job in jobs:
        try:
            if job['status'] != 'complete':raise ValueError('Job not complete')
            row,arrays = selected_row(job,source)
            rows.append(row);arrays_by_key[(row['dataset'],row['fold'],row['rate'])] = arrays
        except Exception as error:errors.append(dict(id=job['id'],error=str(error)))
    expected = {(d,f,r) for d in DATASETS for f in (range(1,6) if d == 'IEMOCAPSix' else (1,)) for r in RATES}
    actual = {(r['dataset'],r['fold'],r['rate']) for r in rows}
    complete = len(jobs) == len(rows) == 56 and actual == expected and not errors
    aggregate,pooled = [],[]
    for dataset in DATASETS:
        folds = tuple(range(1,6)) if dataset == 'IEMOCAPSix' else (1,)
        group = [r for r in rows if r['dataset'] == dataset]
        keys = [k for k in (group[0] if group else {}) if k.endswith(('ACC','UA','WF1','CE','MSE'))]
        for fold in folds:
            chosen = [r for r in group if r['fold'] == fold]
            if len(chosen) != 8 or {r['rate'] for r in chosen} != set(RATES):continue
            for name,rates in [('mean',RATES),('high',RATES[5:])]:
                subset = [r for r in chosen if r['rate'] in rates]
                aggregate.append(dict(dataset=dataset,seed=66,fold=fold,rate=name,
                    aggregation='within_fold_equal_rate',**{k:statistics.mean(r[k] for r in subset) for k in keys}))
        if len(group) != len(folds)*8 or {(r['fold'],r['rate']) for r in group} != {(f,r) for f in folds for r in RATES}:continue
        for name,rates in [(r,(r,)) for r in RATES]+[('mean',RATES),('high',RATES[5:])]:
            subset = [r for r in group if r['rate'] in rates]
            aggregate.append(dict(dataset=dataset,seed=66,fold='all',rate=name,
                aggregation='equal_fold_equal_rate',**{k:statistics.mean(r[k] for r in subset) for k in keys}))
        if dataset == 'IEMOCAPSix':
            for rate in RATES:
                combined = {k:np.concatenate([arrays_by_key[(dataset,f,rate)][k] for f in folds])
                    for k in ('labels','prediction','base_prediction')}
                score = measured_metrics(dataset,combined['labels'],combined['prediction'],combined['base_prediction'])
                flat = measured_metrics(dataset,combined['labels'],combined['base_prediction'],combined['base_prediction'])
                pooled.append(dict(dataset=dataset,seed=66,rate=rate,aggregation='pooled_out_of_fold',
                    metrics=score,Flat=flat))
    result = dict(status='complete' if complete else 'incomplete',source_commit=source,expected_jobs=56,
        validated_rate_results=len(rows),per_rate=rows,aggregate=aggregate,pooled_oof=pooled,failures=errors,
        protocol='INTERNAL own-rate Test-best; MOSEI WF1, classification ACC; same-prediction UA/WF1',
        limitations=['Seed66 only; extra frozen-adapter training budget',
            'Separate own-rate parent/block pairs, not one checkpoint',
            'IEMOCAP headline uses equal fivefold means; pooled OOF is separate'])
    save(root/'RESULT.json',result)
    records = aggregate+rows
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (root/'RESULT.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields or ['dataset','rate']);writer.writeheader();writer.writerows(records)
    lines = ['# Frozen original Nested, seed66: MOSEI / IEMOCAP-6 / MELD','',
        f'Status {result["status"]}; {len(rows)}/56 validated formal runs.','',result['protocol'],'',
        '| Dataset | Fold | Rate | Best epoch | ACC | UA | W-F1 | Flat ACC | Flat UA | Flat W-F1 | ΔACC | ΔUA | ΔWF1 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        cells = [row['dataset'],str(row['fold']),row['rate'],str(row.get('selected_epoch','—'))]
        cells += [f'{row[k]:.4f}' if k in row else '—' for k in
            ('ACC','UA','WF1','Flat_ACC','Flat_UA','Flat_WF1','delta_Flat_ACC','delta_Flat_UA','delta_Flat_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['',*['- '+note for note in result['limitations']]]
    if errors:lines.extend(['',*['- '+str(e) for e in errors]])
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
