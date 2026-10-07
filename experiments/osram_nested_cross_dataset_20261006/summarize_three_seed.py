"""Complete fixed-scope report from existing checkpoints/predictions only."""
import json
from pathlib import Path
import statistics

from .acc_ua import classification_scores
from .run import ROOT, sha, read, write, LABEL


def main():
    import numpy as np
    rates = [str(i/10) for i in range(8)]
    report = dict(label=LABEL, aggregation='fold mean, rate mean, seed mean; sample SD across 3 seeds',
                  seed_list=[66, 67, 68], no_new_inference=True, datasets={})
    for dataset, classes, folds in [('IEMOCAPFour', 4, range(1, 6)),
                                     ('IEMOCAPSix', 6, range(1, 6)), ('CMUMOSEI', None, [1])]:
        seeds = []
        for seed in (66, 67, 68):
            rows = []
            for fold in folds:
                folder = ROOT / dataset / f'seed_{seed}/fold_{fold}'
                provenance = read(folder / 'PROVENANCE.json')
                metrics, history = read(folder / 'metrics.json'), read(folder / 'history.json')
                if (provenance.get('status') != 'complete' or not provenance.get('outputs_verified')
                        or [r['epoch'] for r in history] != list(range(1, 101))):
                    raise ValueError(f'incomplete run {folder}')
                expected = 'accuracy' if classes else 'weighted_f1'
                if metrics.get('selection_metric') != expected:
                    raise ValueError('selection metric mismatch')
                cfg = read(folder / 'config.json')
                if (cfg['dataset'], cfg['seed'], cfg['fold'], cfg['osram_meaningful_block']) != (
                        dataset, seed, fold, 'nested_gnn_rooted_evidence'):
                    raise ValueError('run config mismatch')
                values = {}
                for i, rate in enumerate(rates):
                    prediction = folder / f'predictions_miss_0p{i}.npz'
                    best = folder / f'best_miss_0p{i}.pt'
                    if not best.is_file() or not (folder / 'last_training.pt').is_file():
                        raise ValueError('checkpoint missing')
                    if sha(prediction) != provenance['artifact_sha256'][prediction.name]:
                        raise ValueError('prediction integrity mismatch')
                    selected = metrics['test'][rate]
                    values[rate] = dict(wf1=100*selected['weighted_f1'], acc=100*selected['accuracy'])
                    if classes:
                        with np.load(prediction) as stored:
                            acc, ua, support = classification_scores(stored['labels'], stored['predictions'], classes=classes)
                        if abs(acc-selected['accuracy']) > 1e-12:
                            raise ValueError('accuracy mismatch')
                        values[rate].update(ua=100*ua)
                rows.append(dict(fold=fold, rates=values, provenance_sha256=sha(folder / 'PROVENANCE.json'),
                                 metrics_sha256=sha(folder / 'metrics.json'), history_sha256=sha(folder / 'history.json')))
            keys = ('wf1', 'acc', 'ua') if classes else ('wf1', 'acc')
            per_rate = {r:{k:statistics.mean(row['rates'][r][k] for row in rows) for k in keys} for r in rates}
            seeds.append(dict(seed=seed, folds=rows, rates=per_rate,
                mean8={k:statistics.mean(per_rate[r][k] for r in rates) for k in keys},
                high={k:statistics.mean(per_rate[r][k] for r in rates[5:]) for k in keys}))
        mean = {group:{k:dict(mean=statistics.mean(s[group][k] for s in seeds),
                             sd=statistics.stdev(s[group][k] for s in seeds)) for k in keys}
                for group in ('mean8', 'high')}
        report['datasets'][dataset] = dict(seeds=seeds, summary=mean)
        print(dataset, json.dumps(mean), flush=True)
    path = ROOT / 'THREE_SEED_SUMMARY.json'
    if path.exists():
        raise FileExistsError('Keep existing report; reconcile before regenerating')
    write(path, report)


if __name__ == '__main__':
    main()
