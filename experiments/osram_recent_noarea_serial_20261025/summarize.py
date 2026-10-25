"""Validate saved predictions; never mix joint and per-parent frozen training."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

RATES = tuple(f'{i/10:.1f}' for i in range(8))
ACCEPTED = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(labels, prediction):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    labels, prediction = np.asarray(labels).reshape(-1), np.asarray(prediction).reshape(-1)
    if labels.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Malformed saved prediction')
    active = labels != 0
    truth, guess = labels[active] > 0, prediction[active] > 0
    return dict(ACC=100*float(accuracy_score(truth, guess)),
        WF1=100*float(f1_score(truth, guess, average='weighted', zero_division=0)),
        MSE=float(((labels-prediction)**2).mean()))


def check_prediction(output, best, identity, rate, stage):
    import numpy as np
    for field, checksum in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')):
        path = (output/best[field]).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != best[checksum]:
            raise ValueError('Selected artifact hash/path mismatch')
    selected = best['metrics']['router'] if stage == 'stage1' else best['metrics']
    with np.load(output/best['predictions'], allow_pickle=False) as arrays:
        score = measured(arrays['labels'], arrays['prediction'])
        branch_off = measured(arrays['labels'], arrays['base_prediction'])
        for key in score:
            if abs(score[key]-selected[key]) > 1e-5:
                raise ValueError('Selected metric does not match saved predictions')
        flat = json.loads((ACCEPTED/'SUMMARY.json').read_text())
        old = flat['best_by_rate'][rate]
        if digest(ACCEPTED/old['predictions']) != old['prediction_sha256']:
            raise ValueError('Accepted original prediction changed')
        with np.load(ACCEPTED/old['predictions'], allow_pickle=False) as baseline:
            for key in ('labels', 'availability', 'conversation_ids', 'utterance_indices'):
                np.testing.assert_array_equal(arrays[key], baseline[key])
            parent = measured(baseline['labels'], baseline['prediction'])
            if stage == 'stage2':
                np.testing.assert_allclose(arrays['base_prediction'].reshape(-1),
                    baseline['prediction'].reshape(-1), rtol=1e-5, atol=1e-6)
        nonzero = arrays['labels'].reshape(-1) != 0
        truth = arrays['labels'].reshape(-1)[nonzero] > 0
        old = arrays['base_prediction'].reshape(-1)[nonzero] > 0
        new = arrays['prediction'].reshape(-1)[nonzero] > 0
        pair = dict(corrections=int(((old != truth) & (new == truth)).sum()),
                    harms=int(((old == truth) & (new != truth)).sum()))
    return score, branch_off, parent, pair


def selected_rows(job, source):
    output = Path(job['output'])
    summary = json.loads((output/'SUMMARY.json').read_text())
    identity = summary['identity']
    stage, mode = job['stage'], job['mode']
    expected_stage = 'joint' if stage == 'stage1' else 'frozen'
    if (summary['status'] != 'complete' or summary['epoch'] != 150
        or identity['source_commit'] != source or identity['seed'] != 66
        or identity['stage'] != expected_stage or identity['mode'] != mode or identity['smoke']):
        raise ValueError('Completed stage/mode/source identity mismatch')
    history = json.loads((output/'history.json').read_text())
    if [row['epoch'] for row in history] != list(range(1, 151)):
        raise ValueError('Require150contiguous epochs')
    if stage == 'stage1':
        if not all(summary.get(k) is True for k in ('initial_parameters_exact', 'initial_real_batch_exact',
            'full_checkpoint_restore_verified', 'selected_all_rates_replayed', 'training_inputs_exact')):
            raise ValueError('Missing joint invariance/replay evidence')
        bests = summary['best_by_rate']
        if set(bests) != set(RATES):
            raise ValueError('Joint model must preserve all8rate selections')
        added, total = summary['added_parameters'], summary['total_parameters']
    else:
        if not summary['checks'] or not all(summary['checks'].values()):
            raise ValueError('Missing frozen invariance/replay evidence')
        if identity['evaluation_rate'] != job['rate']:
            raise ValueError('Frozen own-rate selection differs')
        bests = {job['rate']:summary['best']}
        added, total = summary['total_parameters'], summary['full_parameters']
    if added != (281856 if mode == 'parallel' else 238798) or total != 13509793+added:
        raise ValueError('Unexpected module parameter budget')
    rows = []
    for rate, best in bests.items():
        metric = lambda row: row['test'][rate]['router']['WF1'] if stage == 'stage1' else row['test']['WF1']
        if max(history, key=metric)['epoch'] != best['epoch']:
            raise ValueError('Selected epoch is not earliest Test-best WF1')
        score, off, parent, pair = check_prediction(output, best, identity, rate, stage)
        row = dict(stage=stage, mode=mode, seed=66, rate=rate, selected_epoch=best['epoch'],
            added_parameters=added, full_parameters=total, seconds=summary['seconds'],
            peak_memory_mib=summary['peak_memory_mib'], summary=str(output/'SUMMARY.json'), **pair)
        for key in score:
            row[key] = score[key]
            row['branch_off_'+key] = off[key]
            row['Flat_'+key] = parent[key]
            row['delta_Flat_'+key] = score[key]-parent[key]
            row['delta_branch_off_'+key] = score[key]-off[key]
        rows.append(row)
    return rows


def summarize(root, source, jobs):
    root, rows, errors = Path(root), [], []
    for job in jobs:
        try:
            if job['status'] != 'complete':
                raise ValueError('Job status '+job['status'])
            rows += selected_rows(job, source)
        except Exception as error:
            errors.append(dict(id=job['id'], error=str(error)))
    aggregate = []
    for stage in ('stage1', 'stage2'):
        for mode in ('parallel', 'serial'):
            group = [r for r in rows if r['stage'] == stage and r['mode'] == mode]
            if len(group) != 8 or {r['rate'] for r in group} != set(RATES):
                continue
            for name, rates in (('mean', RATES), ('high', RATES[5:])):
                subset = [r for r in group if r['rate'] in rates]
                keys = [k for k in group[0] if k.endswith(('ACC', 'WF1', 'MSE'))]
                aggregate.append(dict(stage=stage, mode=mode, seed=66, rate=name,
                    added_parameters=group[0]['added_parameters'],
                    **{k:statistics.mean(r[k] for r in subset) for k in keys}))
    expected = {(stage, mode, rate) for stage in ('stage1', 'stage2')
                for mode in ('parallel', 'serial') for rate in RATES}
    complete = len(jobs) == 18 and len(rows) == 32 and {(r['stage'], r['mode'], r['rate']) for r in rows} == expected and not errors
    result = dict(status='complete' if complete else 'incomplete', source_commit=source,
        expected_jobs=18, validated_rate_results=len(rows), per_rate=rows, aggregate=aggregate, failures=errors,
        descriptor='NoArea: S1 plus symmetric S2, NOTfirst-order-only',
        protocol='INTERNAL Test-best WF1; ACCsameprediction/mask; stagesreportedseparately',
        limitations=['Joint and frozen stages have different budgets and parent selection; no pooled ranking',
            'Parallel281856 versus serial238798parameters; not equal-capacity wiring isolation',
            'Seed66 only; no novelty or independent generalization claim'])
    save(root/'RESULT.json', result)
    records = aggregate+rows
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['stage', 'mode', 'rate'])
        writer.writeheader(); writer.writerows(records)
    lines = ['# MOSI66 NoArea: parallel versus serial, two training stages', '',
        f'Status {result["status"]}; {len(rows)}/32 validated rate-results from18jobs. Source `{source}`.', '',
        'NoArea retains symmetric second-level terms. Stage1 trains the full model from scratch; '
        'stage2 freezes own-rate accepted Flat and trains only a new block. Never pool stages.', '',
        '| Stage | Mode | Rate/scope | Epoch | ACC | W-F1 | Flat ACC | Flat W-F1 | ΔFlat ACC | ΔFlat W-F1 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        cells = [str(row[k]) for k in ('stage', 'mode', 'rate')]+[str(row.get('selected_epoch', '—'))]
        cells += [f'{row[k]:.4f}' for k in ('ACC', 'WF1', 'Flat_ACC', 'Flat_WF1', 'delta_Flat_ACC', 'delta_Flat_WF1')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', *['- '+note for note in result['limitations']]]
    if errors:
        lines += ['', 'Incomplete/invalid jobs:', *['- '+str(e) for e in errors]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result
