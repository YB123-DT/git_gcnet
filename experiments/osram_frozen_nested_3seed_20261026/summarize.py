"""Independently validate 128 new selections plus 64 protected seed66 results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .protocol import DATASETS, RATES, budget, measured_metrics, selection_metric
from experiments.osram_frozen_nested_cross_20261025.summarize import digest, protected_arrays
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

REMOTE = Path('/data2/yb/remote_experiments')


def aggregate_rows(rows):
    results = []
    for dataset in DATASETS:
        folds = tuple(range(1, 6)) if dataset == 'IEMOCAPSix' else (1,)
        for seed in (66, 67, 68):
            group = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed]
            keys = [k for k in (group[0] if group else {}) if k.endswith(('ACC','UA','WF1','CE','MSE'))]
            for fold in folds:
                chosen = [r for r in group if r['fold'] == fold]
                if len(chosen) != 8 or {r['rate'] for r in chosen} != set(RATES):
                    continue
                for name, rates in [('mean', RATES), ('high', RATES[5:])]:
                    subset = [r for r in chosen if r['rate'] in rates]
                    results.append(dict(dataset=dataset, seed=seed, fold=fold, rate=name,
                        aggregation='within_fold_equal_rate',
                        **{k:statistics.mean(r[k] for r in subset) for k in keys}))
            if len(group) != len(folds)*8 or {(r['fold'],r['rate']) for r in group} != {(f,r) for f in folds for r in RATES}:
                continue
            for name, rates in [(r, (r,)) for r in RATES]+[('mean', RATES), ('high', RATES[5:])]:
                subset = [r for r in group if r['rate'] in rates]
                results.append(dict(dataset=dataset, seed=seed, fold='all', rate=name,
                    aggregation='equal_fold_equal_rate',
                    **{k:statistics.mean(r[k] for r in subset) for k in keys}))
        for rate in list(RATES)+['mean','high']:
            chosen = [r for r in results if r['dataset'] == dataset and r['fold'] == 'all'
                and r['rate'] == rate and r['seed'] in (66, 67, 68)]
            if len(chosen) != 3 or {r['seed'] for r in chosen} != {66, 67, 68}:
                continue
            keys = [k for k in chosen[0] if k.endswith(('ACC','UA','WF1','CE','MSE'))]
            results.append(dict(dataset=dataset, seed='66/67/68', fold='all', rate=rate,
                aggregation='mean_three_seed_equal_fold_equal_rate',
                **{k:statistics.mean(r[k] for r in chosen) for k in keys},
                **{k+'_std':statistics.stdev(r[k] for r in chosen) for k in keys}))
    return results


def selected_row(job, source):
    import numpy as np
    output = Path(job['output']).resolve()
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    dataset, seed, fold, rate = (job[k] for k in ('dataset','seed','fold','rate'))
    if (summary['status'] != 'complete' or summary['epoch'] != budget(dataset)
        or identity['source_commit'] != source or identity['method'] != 'nested_gnn_rooted_evidence'
        or identity['dataset'] != dataset or identity['seed'] != seed or identity['fold'] != fold
        or identity['stage'] != 'frozen' or identity['smoke'] or identity['evaluation_rate'] != rate
        or summary['total_parameters'] != 159235 or not summary['checks'] or not all(summary['checks'].values())):
        raise ValueError('Incomplete/mismatched source, seed, fold, epoch or correctness evidence')
    history = json.loads((output/'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, budget(dataset)+1)):
        raise ValueError('Require full contiguous training history')
    best = summary['best']
    if max(history, key=lambda r:r['test'][selection_metric(dataset)])['epoch'] != best['epoch']:
        raise ValueError('Selected epoch differs from earliest strict Test-best')
    for name, sha in [('checkpoint','checkpoint_sha256'), ('predictions','prediction_sha256')]:
        path = (output/best[name]).resolve()
        if not path.is_relative_to(output) or digest(path) != best[sha]:
            raise ValueError('Selected artifact hash/path differs')
    baseline = summary['baseline_artifact']
    if digest(output/baseline['path']) != baseline['sha256']:
        raise ValueError('Saved baseline changed')
    with np.load(output/best['predictions'], allow_pickle=False) as stored:
        arrays = {k:stored[k].copy() for k in stored.files}
    with np.load(output/baseline['path'], allow_pickle=False) as stored:
        original = {k:stored[k].copy() for k in stored.files}
    protected = protected_arrays(identity['parent'], rate)
    for key in ('labels','availability','conversation_ids','utterance_indices'):
        np.testing.assert_array_equal(arrays[key], original[key])
        np.testing.assert_array_equal(arrays[key], protected[key])
    if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != identity['test_mask_sha256']:
        raise ValueError('Test mask hash differs')
    np.testing.assert_array_equal(arrays['base_prediction'], original['base_prediction'])
    np.testing.assert_array_equal(original['prediction'], original['base_prediction'])
    predicted = protected['prediction'] if 'prediction' in protected else protected['logits']
    if dataset in ('CMUMOSI', 'CMUMOSEI'):
        predicted = predicted.reshape(-1)
    np.testing.assert_array_equal(arrays['base_prediction'], predicted)
    score = measured_metrics(dataset, arrays['labels'], arrays['prediction'], arrays['base_prediction'])
    flat = measured_metrics(dataset, arrays['labels'], arrays['base_prediction'], arrays['base_prediction'])
    for key, value in score.items():
        np.testing.assert_allclose(value, best['metrics'][key], rtol=0, atol=1e-5)
    for key, value in flat.items():
        np.testing.assert_allclose(value, summary['baseline'][key], rtol=0, atol=1e-5)
    row = dict(dataset=dataset, seed=seed, fold=fold, rate=rate, selected_epoch=best['epoch'],
        added_parameters=159235, full_parameters=summary['full_parameters'], seconds=summary['seconds'],
        peak_memory_mib=summary['peak_memory_mib'], corrections=score['corrections'], harms=score['harms'],
        N=score['N'], predictions=str(output/best['predictions']), parent=identity['parent']['checkpoint'])
    for key in ('ACC','UA','WF1','CE','MSE'):
        if key in score:
            row.update({key:score[key], 'Flat_'+key:flat[key], 'delta_Flat_'+key:score[key]-flat[key]})
    for key in ('recall','support'):
        if key in score:
            row[key] = score[key]
    return row


def protected_seed66():
    from experiments.osram_frozen_nested_cross_20261025.summarize import selected_row as cross_row
    from experiments.osram_frozen_nested_gcnet_20261025.summarize import selected_row as mosi_row
    rows, provenance = [], []
    roots = [REMOTE/'osram_frozen_nested_gcnet_20261025/execution_v1',
             REMOTE/'osram_frozen_nested_cross_20261025/execution_v1']
    for root in roots:
        queue = json.loads((root/'QUEUE.json').read_text())
        if queue['status'] != 'complete':
            raise ValueError('Protected seed66 queue is not complete')
        source = queue['source_commit']
        provenance.append(dict(queue=str(root/'QUEUE.json'), sha256=digest(root/'QUEUE.json'), source_commit=source))
        for job in queue['train']:
            if root == roots[0]:
                row = dict(mosi_row(job, source), dataset='CMUMOSI', fold=1)
            else:
                row, _ = cross_row(job, source)
            rows.append(row)
    if len(rows) != 64:
        raise ValueError('Need all64 protected seed66 selections')
    return rows, provenance


def summarize(root, source, jobs):
    root = Path(root)
    rows, errors = [], []
    for job in jobs:
        try:
            if job['status'] != 'complete':
                raise ValueError('Job not complete')
            rows.append(selected_row(job, source))
        except Exception as error:
            errors.append(dict(id=job['id'], error=str(error)))
    provenance = []
    try:
        previous, provenance = protected_seed66()
        rows.extend(previous)
    except Exception as error:
        errors.append(dict(id='protected_seed66', error=str(error)))
    expected = {(d,s,f,r) for d in DATASETS for s in (66,67,68)
        for f in (range(1,6) if d == 'IEMOCAPSix' else (1,)) for r in RATES}
    complete = len(jobs) == 128 and len(rows) == 192 and not errors and {
        (r['dataset'],r['seed'],r['fold'],r['rate']) for r in rows} == expected
    aggregate = aggregate_rows(rows)
    result = dict(status='complete' if complete else 'incomplete', source_commit=source,
        expected_new_jobs=128, expected_total_rate_results=192, validated_rate_results=len(rows),
        reused_seed66=provenance, per_rate=rows, aggregate=aggregate, failures=errors,
        protocol='INTERNAL Test-oracle: own-rate WF1 for MOSI/MOSEI, ACC for MELD/IEMOCAP; UA same prediction',
        limitations=['Eight separate own-rate frozen parents/blocks, not one deployed checkpoint',
            'Extra stage2 training; IEMOCAP headline is equal fivefold mean, not pooled OOF',
            'Seeds67/68 replicate original159235parameter Nested without model or HPO changes',
            'MELD67/68 have no completed same-seed Signature trace; no seed66 trace substitution'])
    save(root/'RESULT.json', result)
    records = aggregate+rows
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['dataset','seed','rate'])
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Original Nested frozen-stage three-seed replication', '',
        f'Status {result["status"]}; validated {len(rows)}/192 rate/fold selections (64 protected seed66 +128 new).',
        '', result['protocol'], '',
        '| Dataset | Seed | Fold | Rate | Epoch | ACC | UA | W-F1 | ΔFlat ACC | ΔFlat UA | ΔFlat W-F1 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        cells = [row['dataset'], str(row['seed']), str(row['fold']), row['rate'], str(row.get('selected_epoch','—'))]
        cells += [f'{row[k]:.4f}' if k in row else '—' for k in
            ('ACC','UA','WF1','delta_Flat_ACC','delta_Flat_UA','delta_Flat_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', *['- '+note for note in result['limitations']]]
    if errors:
        lines += ['', *['- '+str(e) for e in errors]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
