"""Equal-rate means, paired increments and explicit diagnostic boundaries."""
import csv
import json
from pathlib import Path
import statistics

from .analysis import MODELS, EDGES


def summarize(output, records, arrays):
    if set(records) != {f'{i/10:.1f}' for i in range(8)} or len(arrays) != 8:
        raise ValueError('Require all eight rates')
    output = Path(output); aggregate = {}; rows = []; changes = []
    for scope, rates in [('mean', tuple(records)), ('high', ('0.5', '0.6', '0.7'))]:
        aggregate[scope] = dict(models={}, comparisons={})
        for m in MODELS:
            aggregate[scope]['models'][m] = {metric: statistics.mean(
                records[r]['statistics']['models'][m][metric] for r in rates) for metric in ('ACC', 'WF1', 'MSE')}
        for a, b in EDGES:
            edge = a + '_to_' + b
            values = [records[r]['statistics']['comparisons'][edge] for r in rates]
            aggregate[scope]['comparisons'][edge] = dict(
                **{k: statistics.mean(v[k] for v in values) for k in ('gain_ACC_pp', 'gain_WF1_pp', 'delta_MSE')},
                **{k: sum(v[k] for v in values) for k in ('corrections', 'harms', 'N_polarity')})
    for rate, record in records.items():
        for m, v in record['statistics']['models'].items(): rows.append(dict(rate=rate, model=m, **v))
        for edge, v in record['statistics']['comparisons'].items():
            changes.append(dict(rate=rate, comparison=edge, **{k: v[k] for k in
                ('N_polarity', 'gain_ACC_pp', 'gain_WF1_pp', 'delta_MSE', 'corrections', 'harms', 'both_correct', 'both_wrong')}))
    result = dict(label='INTERNAL DIAGNOSTIC ONLY', aggregate=aggregate, per_rate=records,
        protocol='Within-rate identical Direct checkpoint, learned gamma, exact saved masks, one Memory scan. '
        'Across rates: existing W-F1 Test-oracle BEST epochs. Equal-rate metric means; summed counts are repeated exposures, not unique utterances.')
    (output/'REPORT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for name, values in [('METRICS.csv', rows), ('CHANGES.csv', changes)]:
        with (output/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ['# MOSEI seed66 Direct: Local / Base / Gap readout audit', '', 'INTERNAL DIAGNOSTIC ONLY', '',
        'L = original Local adapter input plus original Local Skip; LB = L+Base; LG = L+active Gap; Full = L+Base+active Gap.',
        'Learned Direct source weights remain unchanged. Only final Flat evidence inputs are zeroed; no training or new masks.',
        'Same checkpoint within each rate, different existing W-F1 Test-best checkpoints across rates. No single-checkpoint deployment claim.', '',
        '| Rate/scope | L ACC | LB ACC | LG ACC | Full ACC | L W-F1 | LB W-F1 | LG W-F1 | Full W-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope, models in [(r, v['statistics']['models']) for r, v in records.items()] + [(s, v['models']) for s, v in aggregate.items()]:
        lines.append('| '+scope+' | '+' | '.join(f'{models[m][metric]:.3f}' for metric in ('ACC', 'WF1') for m in MODELS)+' |')
    lines += ['', '## Paired increments (eight-rate equal means; counts sum repeated exposures)', '',
        '| Comparison | Delta ACC(pp) | Delta W-F1(pp) | Corrections | Harms |', '|---|---:|---:|---:|---:|']
    for edge, v in aggregate['mean']['comparisons'].items():
        lines.append(f"| {edge} | {v['gain_ACC_pp']:+.3f} | {v['gain_WF1_pp']:+.3f} | {v['corrections']} | {v['harms']} |")
    lines += ['', 'CE/UA are not MOSEI regression/polarity metrics; ACC/W-F1 use nonzero labels and prediction>0. MSE includes zero labels.',
        'Readout removal can be out-of-distribution. This is not a separately trained architecture ablation, nor proof of a writing/query bottleneck.',
        'CHANGES.csv gives per-rate corrections/harms; positive gain means adding the named evidence helped under this intervention.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
