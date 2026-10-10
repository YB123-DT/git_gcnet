"""Compare cached Flat/Nested predictions for current no-Text utterances only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def wf1(y, pred):
    truth, predicted = y > 0, pred > 0
    total = 0.
    for value in (False, True):
        tp = np.sum((truth == value) & (predicted == value))
        fp = np.sum((truth != value) & (predicted == value))
        fn = np.sum((truth == value) & (predicted != value))
        denominator = 2 * tp + fp + fn
        total += (2 * tp / denominator if denominator else 0.) * np.sum(truth == value)
    return float(total / len(y)) if len(y) else None


def summarize(rows):
    y = np.array([r['label'] for r in rows])
    f = np.array([r['flat_pred'] for r in rows])
    n = np.array([r['nested_pred'] for r in rows])
    valid = y != 0
    fc, nc = (f[valid] > 0) == (y[valid] > 0), (n[valid] > 0) == (y[valid] > 0)
    a, b = wf1(y[valid], f[valid]), wf1(y[valid], n[valid])
    return dict(n=len(rows), nonneutral_n=int(valid.sum()), flat_wf1=100*a,
                nested_wf1=100*b, delta_wf1_pp=100*(b-a),
                flat_acc=100*float(fc.mean()), nested_acc=100*float(nc.mean()),
                corrections=int((~fc & nc).sum()), harms=int((fc & ~nc).sum()),
                both_correct=int((fc & nc).sum()), both_wrong=int((~fc & ~nc).sum()),
                flip_percent=100*float((fc != nc).mean()),
                mean_abs_prediction_change=float(np.abs(n-f).mean()),
                mean_signed_prediction_change=float((n-f).mean()),
                flat_mse=float(((f-y)**2).mean()), nested_mse=float(((n-y)**2).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    models = dict(flat={}, nested={})
    with args.input.open() as stream:
        for row in csv.DictReader(stream):
            # Each utterance occurs once per slot. Use Local to avoid sixfold counting.
            if row['slot'] != 'Local':
                continue
            key = int(row['seed']), float(row['rate']), row['conversation'], int(row['utterance'])
            index = models[row['model']]
            if key in index:
                raise AssertionError('Duplicate utterance')
            index[key] = row
    if models['flat'].keys() != models['nested'].keys():
        raise AssertionError('Unmatched sample IDs')
    pairs = []
    for key, f in models['flat'].items():
        n = models['nested'][key]
        if f['pattern'] != n['pattern'] or float(f['label']) != float(n['label']):
            raise AssertionError('Unmatched mask or label')
        if 'T' in f['pattern']:
            continue
        if f['pattern'] not in ('A', 'V', 'AV') or key[1] == 0:
            raise AssertionError('Unexpected no-Text condition')
        y, fp, npred = float(f['label']), float(f['prediction']), float(n['prediction'])
        outcome = ('neutral' if y == 0 else
                   'correction' if (fp > 0) != (y > 0) and (npred > 0) == (y > 0) else
                   'harm' if (fp > 0) == (y > 0) and (npred > 0) != (y > 0) else
                   'both_correct' if (fp > 0) == (y > 0) else 'both_wrong')
        pairs.append(dict(seed=key[0], rate=key[1], conversation=key[2], utterance=key[3],
                          pattern=f['pattern'], label=y, flat_pred=fp, nested_pred=npred,
                          prediction_change=npred-fp, outcome=outcome))
    per_rate = []
    for seed in (66, 67, 68):
        for rate in (i/10 for i in range(1, 8)):
            for pattern in ('all_no_Text', 'A', 'V', 'AV', 'all_no_Text_history_only'):
                selected = [r for r in pairs if r['seed'] == seed and r['rate'] == rate
                            and (pattern.startswith('all_no_Text') or r['pattern'] == pattern)
                            and (pattern != 'all_no_Text_history_only' or r['utterance'] > 0)]
                if selected:
                    per_rate.append(dict(seed=seed, rate=rate, pattern=pattern, **summarize(selected)))
    macro = []
    for seed in ('all', 66, 67, 68):
        for pattern in ('all_no_Text', 'A', 'V', 'AV', 'all_no_Text_history_only'):
            for rates in ('all_nonempty_rates', 'high_missing'):
                selected = [r for r in per_rate if r['pattern'] == pattern
                            and (seed == 'all' or r['seed'] == seed)
                            and (rates != 'high_missing' or r['rate'] >= .5)]
                record = dict(seed=seed, pattern=pattern, rates=rates, groups=len(selected))
                for name in ('flat_wf1', 'nested_wf1', 'delta_wf1_pp', 'flat_acc', 'nested_acc',
                             'flip_percent', 'mean_abs_prediction_change',
                             'mean_signed_prediction_change', 'flat_mse', 'nested_mse'):
                    record[name] = statistics.mean(r[name] for r in selected)
                for name in ('n', 'nonneutral_n', 'corrections', 'harms', 'both_correct', 'both_wrong'):
                    record[name + '_sum'] = sum(r[name] for r in selected)
                macro.append(record)
    for name, rows in (('missing_text_pairs', pairs), ('missing_text_per_rate', per_rate),
                       ('missing_text_macro', macro)):
        with (args.output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    summary = dict(label='INTERNAL DIAGNOSTIC ONLY', source_sha256=digest,
                   source=str(args.input), matched_all_utterances=len(models['flat']),
                   missing_text_pairs=len(pairs), inference_runs=0, training_runs=0,
                   aggregation='equal mean of nonempty seed/rate groups; rate0.0 has no no-Text rows',
                   macro=macro, per_rate=per_rate)
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps([r for r in macro if r['seed'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
