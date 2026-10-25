"""Same-selected-epoch metrics, never aggregate incomplete eight-rate groups."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import digest
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
EPOCHS = {'CMUMOSI': 150, 'CMUMOSEI': 60, 'MELD': 50}


def selected_row(job):
    output = Path(job['output'])
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    dataset, seed, rate = job['dataset'], job['seed'], job['rate']
    if (summary.get('status') != 'complete' or summary.get('epoch') != EPOCHS[dataset]
            or identity['dataset'] != dataset or identity['seed'] != seed
            or identity['evaluation_rate'] != rate or identity['cyclic_training'] is not True):
        raise ValueError(f'{output}: incomplete or mismatched run identity')
    parent = identity['parent']
    if digest(parent['checkpoint']) != parent['checkpoint_sha256'] or summary['parent_id'] != parent['checkpoint_sha256']:
        raise ValueError(f'{output}: frozen parent checksum mismatch')
    if not identity['test_mask_sha256']:
        raise ValueError(f'{output}: missing exact mask identity')
    if identity.get('mask_protocol') != 'gcnet-random-mask-v1':
        raise ValueError(f'{output}: incorrect missing protocol')
    required_checks = ('parent_unchanged', 'parent_grad_none', 'checkpoint_replay_exact',
                       'cyclic_train_rates_verified', 'only_requested_test_rate')
    if any(summary.get('checks', {}).get(key) is not True for key in required_checks):
        raise ValueError(f'{output}: incomplete frozen-parent/cyclic/replay invariants')
    best = summary['best']
    for field, hash_field in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
        if digest(output/best[field]) != best[hash_field]:
            raise ValueError(f'{output}: selected {field} integrity mismatch')
    history = json.loads((output/'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, EPOCHS[dataset]+1)):
        raise ValueError(f'{output}: incomplete/noncontiguous history')
    for index, entry in enumerate(history):
        consumed = entry['training_rates']
        expected = [float(RATES[(index+batch) % 8]) for batch in range(entry['optimizer_steps'])]
        if not consumed or consumed != expected or not entry.get('mask_trace_sha256'):
            raise ValueError(f'{output}: incorrect cyclic training rates or absent mask trace')
    if not any(entry.get('gradients', {}).get('nonzero', 0) for entry in history):
        raise ValueError(f'{output}: no observed nonzero branch gradients')
    metric = 'ACC' if dataset == 'MELD' else 'WF1'
    # max preserves the first epoch on ties, matching strict-improvement workers.
    selected = max(history, key=lambda row: row['test'][metric])
    if selected['epoch'] != summary['best']['epoch']:
        raise ValueError(f'{output}: selected epoch is not earliest Test-{metric} maximum')
    keys = ('ACC', 'UA', 'WF1', 'CE') if dataset == 'MELD' else ('ACC', 'WF1', 'MSE')
    row = dict(dataset=dataset, seed=seed, rate=rate, selection_metric=metric,
               selected_epoch=selected['epoch'])
    for key in keys:
        value = selected['test'][key]
        if abs(value-summary['best']['metrics'][key]) > 1e-6:
            raise ValueError(f'{output}: {key} not from selected epoch')
        row[key] = value
        row['parent_'+key] = summary['baseline'][key]
        row['delta_'+key] = value-summary['baseline'][key]
    import hashlib
    training_trace = [(entry['epoch'], entry['optimizer_steps'], entry['mask_trace_sha256'])
                      for entry in history]
    row.update(parent_id=summary['parent_id'], test_mask_sha256=identity['test_mask_sha256'],
               training_trace_sha256=hashlib.sha256(json.dumps(training_trace).encode()).hexdigest(),
               initial_block_sha256=identity['initial_block_sha256'],
               summary=str(output/'SUMMARY.json'))
    return row


def summarize(root, source_commit, jobs):
    root = Path(root)
    rows, failures = [], []
    for job in jobs:
        if job['status'] != 'complete':
            failures.append(dict(id=job.get('id'), status=job['status'], error=job.get('error')))
            continue
        try:
            rows.append(selected_row(job))
        except Exception as error:
            failures.append(dict(id=job.get('id'), status='invalid', error=str(error)))
    aggregate = []
    for dataset, seed in sorted({(r['dataset'], r['seed']) for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed]
        if len(group) != 8 or {r['rate'] for r in group} != set(RATES):
            continue
        if (len({r['training_trace_sha256'] for r in group}) != 1
                or len({r['initial_block_sha256'] for r in group}) != 1):
            failures.append(dict(id=f'{dataset}_{seed}', status='invalid',
                error='Eight per-rate parents did not share stage2 initialization/training masks'))
            continue
        for label, subset in (('mean', RATES), ('high', RATES[5:])):
            selected = [r for r in group if r['rate'] in subset]
            numerical = [key for key in group[0] if key not in ('seed', 'selected_epoch')
                         and isinstance(group[0][key], (float, int))]
            aggregate.append(dict(dataset=dataset, seed=seed, rate=label,
                selection_metric=group[0]['selection_metric'],
                **{key: statistics.mean(r[key] for r in selected) for key in numerical}))
    result = dict(status='complete' if len(rows) == 48 and not failures else 'incomplete',
        protocol='INTERNAL per-rate TEST-oracle; same selected epoch/mask metrics',
        source_commit=source_commit, per_rate=rows, aggregate=aggregate, incomplete=failures)
    save(root/'RESULT.json', result)
    fields = list(dict.fromkeys(key for row in rows+aggregate for key in row))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['dataset', 'seed', 'rate'])
        writer.writeheader()
        writer.writerows(aggregate+rows)
    lines = ['# Frozen Signature cyclic multi-dataset stage two', '',
        f'Source `{source_commit}`; completed {len(rows)}/48. Internal Test-oracle exploration.', '',
        'Each row pairs its own frozen Flat parent and exact test mask. MELD selects ACC; '
        'sentiment selects WF1. UA/MSE/CE are never independently maximized. '
        'Incomplete eight-rate groups are excluded from means.', '',
        '| Dataset | Seed | Rate/scope | Epoch | ACC | UA | WF1 | MSE/CE | ΔACC | ΔUA | ΔWF1 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def display(row, key):
        return f'{row[key]:.4f}' if key in row else '—'
    for row in aggregate+rows:
        loss_key = 'CE' if row['dataset'] == 'MELD' else 'MSE'
        lines.append(f'| {row["dataset"]} | {row["seed"]} | {row["rate"]} | '
            f'{row.get("selected_epoch", "—")} | '+ ' | '.join(display(row, key) for key in
                ('ACC', 'UA', 'WF1', loss_key, 'delta_ACC', 'delta_UA', 'delta_WF1'))+' |')
    if failures:
        lines += ['', 'Incomplete/failed jobs:', '', *[f'- {r}' for r in failures]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    state = json.loads((args.root/'QUEUE.json').read_text())
    result = summarize(args.root, state['source_commit'], state.get('train', []))
    print(json.dumps({'status': result['status'], 'completed': len(result['per_rate'])}))


if __name__ == '__main__':
    main()
