"""Assemble paired frozen diagnostic results without treating probes as gains."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ('A', 'B_local', 'B_donor', 'B_real', 'C_donor', 'C_real')


def assemble(root):
    jobs = []
    missing = []
    for i in range(8):
        path = root / f'rate_0p{i}' / 'SUMMARY.json'
        if not path.is_file():
            missing.append(dict(rate=i / 10, status='missing'))
            continue
        value = json.loads(path.read_text())
        if value['status'] != 'complete':
            missing.append(dict(rate=i / 10, status=value['status'], error=value.get('error')))
            continue
        jobs.append(value)
    if jobs:
        reference = jobs[0]
        for job in jobs:
            if (job['source_config'] != reference['source_config'] or job['source'] != reference['source']
                    or job['seed'] != reference['seed'] or job['git_commit'] != reference['git_commit']):
                raise ValueError('Cannot mix incompatible source/config/seed/code diagnostic jobs')
            for name in MODELS[1:]:
                if job['stages'][name]['settings'] != reference['stages'][name]['settings']:
                    raise ValueError('Cannot mix different stage budgets/optimizer/selection protocols')
    rows = []
    for job in jobs:
        for name in MODELS:
            score = job['A_paired'] if name == 'A' else job['stages'][name]['best']
            rows.append(dict(rate=job['rate'], model=name, **{k: score[k] for k in ('N', 'ACC', 'UA', 'WF1', 'CE')},
                             epoch=None if name == 'A' else job['stages'][name]['best_epoch']))
    aggregate = {}
    if not missing:
        for name in MODELS:
            subset = [r for r in rows if r['model'] == name]
            aggregate[name] = {k + '_8rate': float(np.mean([r[k] for r in subset])) for k in ('ACC', 'UA', 'WF1', 'CE')}
            aggregate[name].update({k + '_high': float(np.mean([r[k] for r in subset if r['rate'] >= .5]))
                                    for k in ('ACC', 'UA', 'WF1', 'CE')})
    return dict(status='partial' if missing else 'complete', missing=missing, completed_rates=len(jobs),
                epochs=None if not jobs else jobs[0]['stages']['B_real']['settings']['epochs'],
                rows=rows, aggregate=aggregate, protocol='INTERNAL DIAGNOSTIC ONLY; per-rate test ACC',
                source_A_full={k: float(np.mean([j['A_full'][k] for j in jobs])) for k in ('ACC', 'UA', 'WF1')} if jobs else {},
                total_seconds=sum(j['seconds'] for j in jobs),
                paired_changes={str(j['rate']): j['paired_changes'] for j in jobs},
                availability_groups={str(j['rate']): j['availability_groups'] for j in jobs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'AGGREGATE.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'PER_RATE.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('rate', 'model', 'N', 'ACC', 'UA', 'WF1', 'CE', 'epoch'))
        writer.writeheader()
        writer.writerows(result['rows'])
    lines = ['# Frozen Query versus Decision — MELD seed66', '', 'INTERNAL DIAGNOSTIC ONLY', '',
             f"Completed rates: {result['completed_rates']}/8; status: {result['status']}.", '',
             'Original CHM/Local/Flat are frozen. B fits a new small decoder; C freezes that selected B decoder and changes only Query.',
             'Donors come from a different conversation in the same split, matched on current availability and history presence without labels.',
             'Paired scores use the common donor-matched population; the full A reference is separate. Test masks replay the source winning epoch.',
             f"Each fit uses {result.get('epochs', 'the recorded')} epochs (epoch0 included), test-best ACC; associated UA/W-F1 are not separately selected.", '',
             '| Model | 8-rate ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in result['aggregate'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[k]:.3f}' for k in
                     ('ACC_8rate', 'UA_8rate', 'WF1_8rate', 'ACC_high', 'UA_high', 'WF1_high')) + ' |')
    lines += ['', '## Interpretation boundaries', '',
              '- B_real versus B_local/B_donor tests usable history signal under this decoder class. B versus A also changes head capacity, regularization and fitting protocol.',
              '- C_real versus its exact frozen B_real decoder tests whether this small Query correction can improve retrieval for that decoder. C_donor controls added current-conditioned function capacity.',
              '- No gain does not prove Memory has no information or that Query is globally optimal. A gain on test-selected probes is not a deployable model result.',
              '- No D raw-history fit, backbone training, auxiliary loss, multi-seed extension or architecture search was performed.', '',
              'See PER_RATE.csv for per-rate scores, AGGREGATE.json for corrections/harms and current-availability splits.']
    if result['missing']:
        lines += ['', 'Missing/unfinished runs: ' + json.dumps(result['missing'])]
    (args.output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'completed_rates', 'aggregate')}, indent=2))


if __name__ == '__main__':
    main()
