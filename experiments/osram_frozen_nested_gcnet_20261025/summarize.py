"""Validate every selected Nested prediction against its own frozen parent."""
import csv
import json
from pathlib import Path
import statistics

from experiments.osram_recent_noarea_serial_20261025.summarize import check_prediction
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

RATES = tuple(str(i/10) for i in range(8))


def selected_row(job, source):
    output = Path(job['output'])
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    history = json.loads((output/'history.json').read_text())
    if (summary['status'] != 'complete' or summary['epoch'] != 150
        or summary['total_parameters'] != 159235 or summary['full_parameters'] != 13509793+159235
        or identity['source_commit'] != source or identity['method'] != 'nested_gnn_rooted_evidence'
        or identity['seed'] != 66 or identity['stage'] != 'frozen' or identity['smoke']
        or identity['evaluation_rate'] != job['rate'] or not summary['checks'] or not all(summary['checks'].values())):
        raise ValueError('Incomplete/mismatched Nested source, parent, epoch or correctness evidence')
    if [r['epoch'] for r in history] != list(range(1,151)):
        raise ValueError('Need150contiguous epochs')
    best = summary['best']
    if max(history, key=lambda r:r['test']['WF1'])['epoch'] != best['epoch']:
        raise ValueError('Selected epoch differs from earliest strict own-rate Test-best')
    score, off, parent, pair = check_prediction(output, best, identity, job['rate'], 'stage2')
    row = dict(rate=job['rate'], seed=66, selected_epoch=best['epoch'], added_parameters=159235,
        seconds=summary['seconds'], peak_memory_mib=summary['peak_memory_mib'], **pair)
    for key in score:
        row.update({key:score[key], 'Flat_'+key:parent[key], 'branch_off_'+key:off[key],
            'delta_Flat_'+key:score[key]-parent[key]})
    return row


def summarize(root, source, jobs):
    root, rows, errors = Path(root), [], []
    for job in jobs:
        try:
            if job['status'] != 'complete': raise ValueError('Job not complete')
            rows.append(selected_row(job, source))
        except Exception as error:
            errors.append(dict(id=job['id'], error=str(error)))
    complete = len(jobs) == len(rows) == 8 and {r['rate'] for r in rows} == set(RATES) and not errors
    aggregate = []
    if complete:
        keys = [k for k in rows[0] if k.endswith(('ACC','WF1','MSE'))]
        for name,rates in [('mean',RATES),('high',RATES[5:])]:
            selected = [r for r in rows if r['rate'] in rates]
            aggregate.append(dict(rate=name, **{k:statistics.mean(r[k] for r in selected) for k in keys}))
    result = dict(status='complete' if complete else 'incomplete', source_commit=source,
        expected_jobs=8, validated_rate_results=len(rows), method='nested_gnn_rooted_evidence',
        per_rate=rows, aggregate=aggregate, failures=errors,
        protocol='INTERNAL Test-best W-F1, same prediction ACC; eight own-rate frozen parents, cyclic training')
    save(root/'RESULT.json', result)
    records = aggregate+rows
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (root/'RESULT.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields or ['rate'])
        writer.writeheader();writer.writerows(records)
    lines = ['# MOSI66 original Nested: frozen own-rate Flat parents', '',
        f'Status {result["status"]}; {len(rows)}/8 validated150epoch runs.', '',
        result['protocol'], '',
        '| Rate | Best epoch | ACC | W-F1 | Flat ACC | Flat W-F1 | ΔACC(pp) | ΔW-F1(pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        cells = [row['rate'], str(row.get('selected_epoch','—'))]
        cells += [f'{row[k]:.4f}' for k in ('ACC','WF1','Flat_ACC','Flat_WF1','delta_Flat_ACC','delta_Flat_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Seed66 only. No statistical or independent-generalization claim.',
        'Eight parents/blocks; the mean is not a single deployable checkpoint.',
        'Nested159235parameters, NoArea parallel281856 and serial238798; not capacity matched.']
    if errors: lines.extend(['',*['- '+str(e) for e in errors]])
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
