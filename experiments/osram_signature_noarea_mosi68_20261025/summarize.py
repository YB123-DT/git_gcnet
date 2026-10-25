"""Paired NoArea/full Signature/Flat metrics from their exact selected predictions."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from experiments.osram_signature_cyclic_multi_20261025.summarize import selected_row, digest
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

NORMAL_SOURCE = '4d5aa09d999b49d7194f72f6bf75c8ab75edbef1'
NORMAL_ROOT = Path('/data2/yb/remote_experiments/osram_signature_cyclic68_20261025/execution_v1/train')
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def prediction_metrics(labels, prediction):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    labels, prediction = np.asarray(labels).reshape(-1), np.asarray(prediction).reshape(-1)
    if len(labels) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('Invalid saved prediction')
    nonzero = labels != 0
    truth, guess = labels[nonzero] > 0, prediction[nonzero] > 0
    return dict(ACC=100*float(accuracy_score(truth, guess)),
        WF1=100*float(f1_score(truth, guess, average='weighted', zero_division=0)),
        MSE=float(((labels-prediction)**2).mean()), N=int(nonzero.sum()))


def validate_pair(identity, normal, source, rate):
    expected = dict(dataset='CMUMOSI', seed=68, stage2_epochs=150,
        evaluation_rate=rate, cyclic_training=True, mask_protocol='gcnet-random-mask-v1')
    if (identity.get('source_commit') != source or normal.get('source_commit') != NORMAL_SOURCE
        or identity.get('ablation') != 'degree2_no_area'
        or any(identity.get(k) != v or normal.get(k) != v for k, v in expected.items())):
        raise ValueError('Paired run scope/source/protocol mismatch')
    for key in ('initial_block_sha256', 'test_mask_sha256', 'parent', 'effective_stage2_config'):
        if identity[key] != normal[key]:
            raise ValueError('Paired run differs in '+key)


def paired_row(job, source):
    import numpy as np
    row = selected_row(job)
    normal_job = dict(job, output=str(NORMAL_ROOT/job['rate']))
    normal_row = selected_row(normal_job)
    output, normal_dir = Path(job['output']), Path(normal_job['output'])
    records = [json.loads((p/'SUMMARY.json').read_text()) for p in (output, normal_dir)]
    validate_pair(records[0]['identity'], records[1]['identity'], source, job['rate'])
    if any(record['total_parameters'] != 281856 for record in records):
        raise ValueError('Trainable parameter budget changed')
    for key in ('parent_id', 'test_mask_sha256', 'initial_block_sha256', 'training_trace_sha256'):
        if row[key] != normal_row[key]:
            raise ValueError('Paired trace differs in '+key)
    for record, path, evaluated in zip(records, (output, normal_dir), (row, normal_row)):
        with np.load(path/record['best']['predictions']) as arrays:
            for prefix, field in (('', 'prediction'), ('parent_', 'base_prediction')):
                measured = prediction_metrics(arrays['labels'], arrays[field])
                for key in ('ACC', 'WF1', 'MSE'):
                    if abs(measured[key]-evaluated[prefix+key]) > 1e-5:
                        raise ValueError('Saved prediction metrics do not match selected metrics')
            import hashlib
            if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != record['identity']['test_mask_sha256']:
                raise ValueError('Saved availability hash differs')
    histories = [json.loads((p/'history.json').read_text()) for p in (output, normal_dir)]
    for new, old in zip(*histories):
        for key in ('epoch', 'N', 'optimizer_steps', 'training_rates', 'mask_trace_sha256', 'learning_rate'):
            if new.get(key) != old.get(key):
                raise ValueError('Full per-epoch training trace differs: '+key)
    for key in ('ACC', 'WF1', 'MSE'):
        if abs(row['parent_'+key]-normal_row['parent_'+key]) > 1e-5:
            raise ValueError('Different frozen parent predictions')
        row['normal_'+key] = normal_row[key]
        row['delta_signature_'+key] = row[key]-normal_row[key]
    row['normal_selected_epoch'] = normal_row['selected_epoch']
    row['normal_summary'] = str(normal_dir/'SUMMARY.json')
    return row


def summarize(root, source, jobs):
    root, rows, errors = Path(root), [], []
    for job in jobs:
        try:
            if job['status'] != 'complete':
                raise ValueError('Job status '+job['status'])
            rows.append(paired_row(job, source))
        except Exception as error:
            errors.append(dict(id=job['id'], error=str(error)))
    complete = (len(rows) == 8 and {row['rate'] for row in rows} == set(RATES) and not errors)
    if complete and (len({r['initial_block_sha256'] for r in rows}) != 1
                     or len({r['training_trace_sha256'] for r in rows}) != 1):
        complete = False
        errors.append(dict(id='eight-rate-group', error='Per-rate initialization or training masks differ'))
    aggregates = []
    if complete:
        keys = [k for k in rows[0] if k.endswith(('ACC', 'WF1', 'MSE'))]
        for label, rates in (('mean', RATES), ('high', RATES[5:])):
            group = [row for row in rows if row['rate'] in rates]
            aggregates.append(dict(dataset='CMUMOSI', seed=68, rate=label,
                **{key:statistics.mean(row[key] for row in group) for key in keys}))
    result = dict(status='complete' if complete else 'incomplete', source_commit=source,
        normal_source_commit=NORMAL_SOURCE, normal_reference=str(NORMAL_ROOT),
        expected_jobs=8, validated_jobs=len(rows), per_rate=rows, aggregate=aggregates, incomplete=errors,
        protocol='INTERNAL Test-WF1 oracle; matched frozen-parent stage2; seed68 only',
        limitation='Retrained no-antisymmetric-area control, not frozen inference ablation or one-stage training')
    save(root/'RESULT.json', result)
    records = aggregates+rows
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['rate'])
        writer.writeheader()
        writer.writerows(records)
    lines = ['# MOSI68 matched retrained NoArea Signature', '',
        f'Status {result["status"]}; validated {len(rows)}/8. Source `{source}`.', '',
        'Original same-rate frozen Flat, same initialization, cyclic masks, 150 epochs and exact test masks. '
        'Only antisymmetric second-level area is removed. Internal Test-best WF1; ACC from the same predictions.', '',
        '| Rate/scope | Epoch | Flat ACC | Flat WF1 | Full ACC | Full WF1 | NoArea ACC | NoArea WF1 | ΔFull ACC | ΔFull WF1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        cells = [str(row['rate']), str(row.get('selected_epoch', '—'))]
        cells += [f'{row[key]:.4f}' for key in ('parent_ACC', 'parent_WF1', 'normal_ACC', 'normal_WF1',
            'ACC', 'WF1', 'delta_signature_ACC', 'delta_signature_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    if errors:
        lines += ['', 'Incomplete/invalid: '+json.dumps(errors)]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads((args.root/'QUEUE.json').read_text())
    print(json.dumps(summarize(args.root, queue['source_commit'], queue.get('train', []))))
