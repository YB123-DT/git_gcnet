"""Strict paired comparison: retrained NoArea versus original Signature and Flat."""
import argparse
import csv
import json
from pathlib import Path

from experiments.osram_signature_iemocap_cyclic_20261025.summarize import KEYS, aggregate_rows, selected_row
from experiments.osram_three_block_joint_mosi_20261025.scheduler import save

NORMAL_SOURCE = '21e46f2ecd092345818659694db8ff372672c35e'
NORMAL_ROOT = Path('/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1/train')
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def paired_row(job, source_commit):
    row = selected_row(job, source_commit)
    path = Path(job['output'])/'SUMMARY.json'
    summary = json.loads(path.read_text())
    identity = summary['identity']
    if identity.get('ablation') != 'degree2_no_area' or identity['stage2_epochs'] != 100 or job['seed'] != 68:
        raise ValueError(f'{path}: expected 100-epoch seed68 degree2_no_area run')
    reference_job = dict(job, output=str(NORMAL_ROOT/job['id']), status='complete')
    normal = selected_row(reference_job, NORMAL_SOURCE)
    reference_path = Path(reference_job['output'])/'SUMMARY.json'
    reference = json.loads(reference_path.read_text())
    for field in ('parent_id', 'test_mask_sha256', 'initial_block_sha256', 'training_trace_sha256'):
        if row[field] != normal[field]:
            raise ValueError(f'{path}: NoArea/Signature pair differs in {field}')
    if reference['identity']['stage2_epochs'] != 100 or summary['total_parameters'] != reference['total_parameters']:
        raise ValueError(f'{path}: NoArea/Signature training/parameter budget differs')
    for key in KEYS:
        if abs(row['parent_'+key]-normal['parent_'+key]) > 1e-5:
            raise ValueError(f'{path}: NoArea/Signature frozen Flat predictions differ')
        row['normal_'+key] = normal[key]
        row['delta_signature_'+key] = row[key]-normal[key]
    row.update(normal_summary=str(reference_path), normal_selected_epoch=normal['selected_epoch'],
               total_parameters=summary['total_parameters'], ablation='degree2_no_area')
    return row


def aggregate_paired(rows):
    aggregate, pooled = aggregate_rows(rows)
    # Reuse the exact arithmetic grouping; swap only the comparison baseline.
    compare = []
    for row in rows:
        adjusted = dict(row)
        for key in KEYS:
            adjusted['parent_'+key] = row['normal_'+key]
            adjusted['delta_'+key] = row['delta_signature_'+key]
        # Pooled normal results require normal confusion; omit this optional view.
        adjusted.pop('confusion', None)
        adjusted.pop('parent_confusion', None)
        compare.append(adjusted)
    comparison, _ = aggregate_rows(compare)
    index = {(r['scope'], r.get('seed'), r.get('fold'), r.get('rate')):r for r in comparison}
    for row in aggregate:
        normal = index[(row['scope'], row.get('seed'), row.get('fold'), row.get('rate'))]
        for key in KEYS:
            row['normal_'+key] = normal['parent_'+key]
            row['delta_signature_'+key] = normal['delta_'+key]
    return aggregate, pooled


def summarize(root, source_commit, jobs):
    root = Path(root)
    rows, failures, invalid_folds = [], [], set()
    for job in jobs:
        try:
            if job['status'] != 'complete':
                raise ValueError(job.get('error', f'Job status {job["status"]}'))
            rows.append(paired_row(job, source_commit))
        except Exception as error:
            failures.append(dict(id=job.get('id'), status='invalid', error=str(error)))
    for fold in range(1, 6):
        group = [r for r in rows if r['fold'] == fold]
        if len(group) == 8 and (len({r['initial_block_sha256'] for r in group}) != 1
                               or len({r['training_trace_sha256'] for r in group}) != 1):
            invalid_folds.add(fold)
            failures.append(dict(id=f'68_fold{fold}', status='invalid', error='Per-rate initialization/training mask traces differ'))
    valid = [row for row in rows if row['fold'] not in invalid_folds]
    aggregate, pooled = aggregate_paired(valid)
    expected = {(68, fold, rate) for fold in range(1, 6) for rate in RATES}
    complete = not failures and len(rows) == 40 and {(r['seed'], r['fold'], r['rate']) for r in rows} == expected
    result = dict(status='complete' if complete else 'incomplete', source_commit=source_commit,
        normal_source_commit=NORMAL_SOURCE, normal_reference=str(NORMAL_ROOT), ablation='degree2_no_area',
        protocol='INTERNAL per-rate Test-ACC oracle; same-epoch same-mask UA/WF1/CE; seed68 only',
        per_rate=rows, aggregate=aggregate, pooled_vs_flat=pooled, incomplete=failures,
        expected_jobs=40, validated_jobs=len(rows),
        limitation='Frozen-parent branch training; no three-seed claim. This is a retrained ablation, not the earlier frozen sensitivity test.')
    save(root/'RESULT.json', result)
    records = aggregate+rows
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (root/'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['seed', 'fold', 'rate'])
        writer.writeheader()
        writer.writerows(records)
    lines = ['# IEMOCAP-6 seed68: retrained NoArea Signature', '',
        f'Validated {len(rows)}/40; status **{result["status"]}**. Source `{source_commit}`.', '',
        'Same frozen own-rate Flat, initial weights, exact evaluation masks, cyclic training mask traces, '
        '100-epoch Adam budget and parameter count as original Signature. '
        'Only second-level antisymmetric area is removed. First level and symmetric second level remain. '
        'Internal Test-ACC oracle; UA/W-F1 use the same selected predictions. No three-seed claim.', '',
        '| Scope | Fold | Rate | Epoch | Flat ACC | Flat UA | Signature ACC | Signature UA | NoArea ACC | NoArea UA | ΔFlat ACC | ΔFlat UA | ΔSignature ACC | ΔSignature UA |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        cells = [str(row.get(k, '—')) for k in ('scope', 'fold', 'rate', 'selected_epoch')]
        cells += [f'{row[key]:.4f}' for key in ('parent_ACC', 'parent_UA', 'normal_ACC', 'normal_UA',
            'ACC', 'UA', 'delta_ACC', 'delta_UA', 'delta_signature_ACC', 'delta_signature_UA')]
        lines.append('| '+' | '.join(cells)+' |')
    if failures:
        lines += ['', 'Incomplete/invalid jobs:', '', *[f'- {item}' for item in failures]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    state = json.loads((args.root/'QUEUE.json').read_text())
    result = summarize(args.root, state['source_commit'], state.get('train', []))
    print(json.dumps(dict(status=result['status'], validated=result['validated_jobs'], expected=40)))


if __name__ == '__main__':
    main()
