"""ACC/UA from existing IEMOCAP six-class predictions; no new inference."""
import hashlib
import json
from pathlib import Path
import statistics


def classification_scores(labels, predictions, *, classes=6):
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError('empty or mismatched predictions')
    count, correct = [0] * classes, [0] * classes
    for y, p in zip(labels, predictions):
        if y not in range(classes) or p not in range(classes):
            raise ValueError('invalid class index')
        count[int(y)] += 1
        correct[int(y)] += int(y == p)
    if not all(count):
        raise ValueError('class absent; do not silently change UA class set')
    return sum(correct) / len(labels), statistics.mean(c / n for c, n in zip(correct, count)), count


def main():
    import numpy as np
    remote = Path('/data2/yb/remote_experiments')
    output = remote / 'osram_nested_cross_dataset_20261006'
    roots = {'Flat': remote / 'osram_iemocap_cfg84_nojepa_5session_20260919/iemocap6/seed_66',
             'Nested': output / 'IEMOCAPSix/seed_66'}
    result = dict(label='INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
                  seed=66, classes=6, aggregation='equal fold mean then equal rate mean',
                  ua_definition='mean recall across all six classes; NOT macro F1',
                  caveat='historical Flat selection_metric unrecorded; Nested accuracy selection',
                  models={})
    rates = [str(i / 10) for i in range(8)]
    for name, root in roots.items():
        folds = []
        for f in range(1, 6):
            folder = root / f'fold_{f}'
            metrics = json.loads((folder / 'metrics.json').read_text())
            row = dict(fold=f, rates={}, selection_metric=metrics.get('selection_metric'))
            for rate in rates:
                filename = f'predictions_miss_{rate.replace(".", "p")}.npz'
                path = folder / filename
                with np.load(path) as saved:
                    y, p, mask = saved['labels'], saved['predictions'], saved['availability']
                if name == 'Nested':
                    with np.load(roots['Flat'] / f'fold_{f}' / filename) as baseline:
                        if not np.array_equal(y, baseline['labels']) or not np.array_equal(mask, baseline['availability']):
                            raise ValueError('sample label or availability order differs')
                acc, ua, support = classification_scores(y, p)
                if abs(acc - metrics['test'][rate]['accuracy']) > 1e-12:
                    raise ValueError('saved predictions do not match selected accuracy')
                row['rates'][rate] = dict(acc=100*acc, ua=100*ua, class_support=support,
                     prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            folds.append(row)
        mean_rates = {r: {key: statistics.mean(f['rates'][r][key] for f in folds)
                         for key in ('acc', 'ua')} for r in rates}
        result['models'][name] = dict(root=str(root), folds=folds, rates=mean_rates,
            mean8={key: statistics.mean(mean_rates[r][key] for r in rates) for key in ('acc', 'ua')},
            high={key: statistics.mean(mean_rates[r][key] for r in rates[5:]) for key in ('acc', 'ua')})
    result['difference_pp'] = {group: {key: result['models']['Nested'][group][key] -
        result['models']['Flat'][group][key] for key in ('acc', 'ua')} for group in ('mean8', 'high')}
    path = output / 'ACC_UA_SUMMARY.json'
    if path.exists():
        raise FileExistsError('Do not overwrite a previous metric audit')
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({name: {k:v[k] for k in ('mean8','high','rates')} for name,v in result['models'].items()}, indent=2))
    print('difference_pp', result['difference_pp'])


if __name__ == '__main__':
    main()
