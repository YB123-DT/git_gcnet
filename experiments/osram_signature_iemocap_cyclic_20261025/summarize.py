"""Validate selected-ACC predictions, then separate fivefold means from pooling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import digest
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
KEYS = ('ACC', 'UA', 'WF1', 'CE')


def confusion_metrics(confusion):
    import numpy as np
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (matrix < 0).any():
        raise ValueError('Expected square nonnegative confusion counts')
    support, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    count = support.sum()
    if not count:
        raise ValueError('Cannot score empty predictions')
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(2*correct, support+predicted, out=np.zeros_like(correct), where=support+predicted > 0)
    return dict(ACC=100*float(correct.sum()/count), UA=100*float(recall.mean()),
                WF1=100*float((f1*support).sum()/count), N=int(count),
                recall=(100*recall).tolist(), support=support.astype(int).tolist())


def prediction_metrics(labels, logits):
    import numpy as np
    labels, logits = np.asarray(labels).reshape(-1), np.asarray(logits)
    if logits.shape != (len(labels), 6) or not np.isfinite(logits).all():
        raise ValueError('Expected finite six-class logits')
    truth = labels.astype(np.int64)
    if not np.array_equal(truth, labels) or (truth < 0).any() or (truth >= 6).any():
        raise ValueError('Invalid IEMOCAP-6 labels')
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (truth, logits.argmax(-1)), 1)
    shifted = logits.astype(np.float64)-logits.max(-1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(-1))-shifted[np.arange(len(labels)), truth]
    metrics = confusion_metrics(confusion)
    metrics['CE'] = float(ce.mean())
    return metrics, confusion.tolist()


def selected_row(job, source_commit):
    import numpy as np
    output = Path(job['output'])
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    epochs = identity['stage2_epochs']
    if (not isinstance(epochs, int) or epochs <= 0 or summary.get('status') != 'complete'
            or summary.get('epoch') != epochs or identity['dataset'] != 'IEMOCAPSix'
            or identity['seed'] != job['seed'] or identity['fold'] != job['fold']
            or identity['evaluation_rate'] != job['rate']
            or identity['source_commit'] != source_commit
            or identity['cyclic_training'] is not True
            or identity['mask_protocol'] != 'gcnet-random-mask-v1'):
        raise ValueError(f'{output}: incomplete or mismatched identity')
    parent = identity['parent']
    if (digest(parent['checkpoint']) != parent['checkpoint_sha256']
            or summary['parent_id'] != parent['checkpoint_sha256']):
        raise ValueError(f'{output}: frozen parent checksum mismatch')
    required = ('parent_unchanged', 'parent_grad_none', 'checkpoint_replay_exact',
                'cyclic_train_rates_verified', 'only_requested_test_rate')
    if any(summary.get('checks', {}).get(key) is not True for key in required):
        raise ValueError(f'{output}: missing frozen/cyclic invariance checks')
    best = summary['best']
    for field, hash_field in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
        if digest(output/best[field]) != best[hash_field]:
            raise ValueError(f'{output}: selected {field} checksum mismatch')
    history = json.loads((output/'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, epochs+1)):
        raise ValueError(f'{output}: incomplete/noncontiguous epoch history')
    for index, entry in enumerate(history):
        expected = [float(RATES[(index+batch) % 8]) for batch in range(entry['optimizer_steps'])]
        if not expected or entry['training_rates'] != expected or not entry.get('mask_trace_sha256'):
            raise ValueError(f'{output}: cyclic rate/mask trace mismatch')
    if not any(r.get('gradients', {}).get('nonzero', 0) for r in history):
        raise ValueError(f'{output}: no observed nonzero Signature gradient')
    selected = max(history, key=lambda row: row['test']['ACC'])
    if selected['epoch'] != best['epoch']:
        raise ValueError(f'{output}: not earliest strict Test-ACC maximum')
    with np.load(output/best['predictions'], allow_pickle=False) as arrays:
        metrics, confusion = prediction_metrics(arrays['labels'], arrays['prediction'])
        baseline, parent_confusion = prediction_metrics(arrays['labels'], arrays['base_prediction'])
        mask_hash = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    if mask_hash != identity['test_mask_sha256']:
        raise ValueError(f'{output}: selected prediction mask mismatch')
    row = dict(dataset='IEMOCAPSix', seed=job['seed'], fold=job['fold'], rate=job['rate'],
               selection_metric='ACC', selected_epoch=selected['epoch'], N=metrics['N'])
    for key in KEYS:
        values = (metrics[key], baseline[key], selected['test'][key], best['metrics'][key], summary['baseline'][key])
        if (not all(math.isfinite(value) for value in values)
                or abs(metrics[key]-selected['test'][key]) > 1e-5
                or abs(metrics[key]-best['metrics'][key]) > 1e-5
                or abs(baseline[key]-summary['baseline'][key]) > 1e-5):
            raise ValueError(f'{output}: {key} disagrees with selected prediction/parent')
        row[key], row['parent_'+key], row['delta_'+key] = metrics[key], baseline[key], metrics[key]-baseline[key]
    row.update(confusion=confusion, parent_confusion=parent_confusion, recall=metrics['recall'],
               parent_recall=baseline['recall'], support=metrics['support'],
               corrections=best['metrics'].get('corrections'), harms=best['metrics'].get('harms'),
               parent_id=summary['parent_id'], test_mask_sha256=mask_hash,
               training_trace_sha256=hashlib.sha256(json.dumps([
                   (r['epoch'], r['optimizer_steps'], r['mask_trace_sha256']) for r in history]).encode()).hexdigest(),
               initial_block_sha256=identity['initial_block_sha256'], summary=str(output/'SUMMARY.json'))
    return row


def mean_row(rows, scope, **identity):
    return dict(scope=scope, **identity, **{prefix+key: statistics.mean(r[prefix+key] for r in rows)
                for prefix in ('', 'parent_', 'delta_') for key in KEYS})


def aggregate_rows(rows):
    import numpy as np
    aggregate, pooled = [], []
    for seed in (66, 67, 68):
        all_seed = [r for r in rows if r['seed'] == seed]
        for fold in range(1, 6):
            group = [r for r in all_seed if r['fold'] == fold]
            if len(group) == 8 and {r['rate'] for r in group} == set(RATES):
                for label, rates in (('mean', RATES), ('high', RATES[5:])):
                    aggregate.append(mean_row([r for r in group if r['rate'] in rates],
                        f'fold_eight_rate_{label}' if label == 'mean' else 'fold_high_rate_mean',
                        dataset='IEMOCAPSix', seed=seed, fold=fold, rate=label))
        if len(all_seed) != 40 or {(r['fold'], r['rate']) for r in all_seed} != {
                (fold, rate) for fold in range(1, 6) for rate in RATES}:
            continue
        for rate in RATES:
            group = [r for r in all_seed if r['rate'] == rate]
            aggregate.append(mean_row(group, 'fivefold_arithmetic_mean', dataset='IEMOCAPSix', seed=seed, rate=rate))
            if all('confusion' in r and 'parent_confusion' in r for r in group):
                combined = confusion_metrics(np.sum([r['confusion'] for r in group], axis=0))
                original = confusion_metrics(np.sum([r['parent_confusion'] for r in group], axis=0))
                n = sum(r['N'] for r in group)
                combined['CE'] = sum(r['CE']*r['N'] for r in group)/n
                original['CE'] = sum(r['parent_CE']*r['N'] for r in group)/n
                record = dict(scope='fivefold_pooled_predictions', dataset='IEMOCAPSix', seed=seed, rate=rate, N=n)
                for key in KEYS:
                    record.update({key:combined[key], 'parent_'+key:original[key], 'delta_'+key:combined[key]-original[key]})
                pooled.append(record)
        for label, rates in (('mean', RATES), ('high', RATES[5:])):
            aggregate.append(mean_row([r for r in all_seed if r['rate'] in rates],
                'fivefold_eight_rate_mean' if label == 'mean' else 'fivefold_high_rate_mean',
                dataset='IEMOCAPSix', seed=seed, rate=label))
    for label in ('mean', 'high'):
        group = [r for r in aggregate if r['scope'] == (
            'fivefold_eight_rate_mean' if label == 'mean' else 'fivefold_high_rate_mean')]
        if len(group) == 3:
            record = mean_row(group, 'three_seed_fivefold_mean', dataset='IEMOCAPSix', rate=label)
            for prefix in ('', 'parent_', 'delta_'):
                for key in KEYS:
                    record[prefix+key+'_std'] = statistics.stdev(r[prefix+key] for r in group)
            aggregate.append(record)
    return aggregate, pooled


def summarize(root, source_commit, jobs):
    root = Path(root)
    rows, failures, invalid_groups = [], [], set()
    for job in jobs:
        try:
            if job['status'] != 'complete':
                raise ValueError(job.get('error', f'Job status {job["status"]}'))
            rows.append(selected_row(job, source_commit))
        except Exception as error:
            failures.append(dict(id=job.get('id'), status='invalid', error=str(error)))
    for seed in (66, 67, 68):
        for fold in range(1, 6):
            group = [r for r in rows if r['seed'] == seed and r['fold'] == fold]
            if len(group) == 8 and (len({r['training_trace_sha256'] for r in group}) != 1
                                   or len({r['initial_block_sha256'] for r in group}) != 1):
                invalid_groups.add((seed, fold))
                failures.append(dict(id=f'{seed}_fold{fold}', status='invalid',
                    error='Eight per-rate parents differ in stage2 initialization or training masks'))
    aggregate, pooled = aggregate_rows([r for r in rows if (r['seed'], r['fold']) not in invalid_groups])
    expected = {(s, f, r) for s in (66, 67, 68) for f in range(1, 6) for r in RATES}
    result = dict(status='complete' if not failures and len(rows) == 120 and {
        (r['seed'], r['fold'], r['rate']) for r in rows} == expected else 'incomplete',
        protocol='INTERNAL per-rate Test-ACC oracle; UA/WF1/CE use same selected epoch and exact mask',
        source_commit=source_commit, per_rate=rows, aggregate=aggregate, pooled=pooled, incomplete=failures)
    save(root/'RESULT.json', result)
    csv_rows = aggregate+pooled+rows
    fields = list(dict.fromkeys(k for row in csv_rows for k in row))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['seed', 'fold', 'rate'])
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ['# IEMOCAP-6 frozen Signature cyclic stage two', '',
        f'Source `{source_commit}`; validated {len(rows)}/120. Internal Test-oracle exploration.', '',
        'Frozen own-rate Flat parent; cyclic random training; current-rate test only. '
        'All UA/WF1/CE correspond to the earliest maximum-ACC epoch. '
        'Fivefold arithmetic means and pooled-prediction metrics are separately labelled. '
        'Incomplete seeds never receive a fivefold summary. Std is sample std over three seeds.', '',
        '| Scope | Seed | Fold | Rate | Epoch | ACC | UA | W-F1 | CE | ΔACC | ΔUA | ΔW-F1 |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in csv_rows:
        cells = [str(row.get(k, '—')) for k in ('scope', 'seed', 'fold', 'rate', 'selected_epoch')]
        cells += [f'{row[k]:.4f}' for k in ('ACC', 'UA', 'WF1', 'CE', 'delta_ACC', 'delta_UA', 'delta_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    if failures:
        lines += ['', 'Invalid/incomplete jobs:', '', *[f'- {r}' for r in failures]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    state = json.loads((args.root/'QUEUE.json').read_text())
    result = summarize(args.root, state['source_commit'], state.get('train', []))
    print(json.dumps(dict(status=result['status'], completed=len(result['per_rate']))))


if __name__ == '__main__':
    main()
