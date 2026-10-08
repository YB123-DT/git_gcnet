"""Read existing selected predictions; no training, inference or reselection."""
import hashlib
import json
from pathlib import Path
import statistics as st

import numpy as np


ROOT = Path('/data2/yb/remote_experiments')
RATES = [str(i / 10) for i in range(8)]


def folder(dataset, model, seed, fold):
    if model == 'Nested':
        if dataset == 'CMUMOSI':
            suffix = f'runs/nested_gnn_rooted_evidence/seed_{seed}'
            prefix = ('osram_new40_gpu0123_20261004/attempt2' if seed == 66
                      else 'osram_readout_top3_3seed_20261005')
            return ROOT / prefix / suffix
        return ROOT / f'osram_nested_cross_dataset_20261006/{dataset}/seed_{seed}/fold_{fold}'
    if dataset == 'CMUMOSI':
        return ROOT / f'osram_mosi_memory_gap_ablation_20260920/full/seed_{seed}'
    if dataset == 'CMUMOSEI':
        return ROOT / f'osram_mosei_cfg84_nojepa_20260919/seed_{seed}'
    classes = 4 if dataset == 'IEMOCAPFour' else 6
    return ROOT / f'osram_iemocap_cfg84_nojepa_5session_20260919/iemocap{classes}/seed_{seed}/fold_{fold}'


def main():
    report = {'label': 'INTERNAL DIAGNOSTIC ONLY',
              'protocol': 'Existing per-rate BEST; equal folds, rates, seeds66/67/68; sample SD across seeds.',
              'caveat': 'Historical Flat IEMOCAP selection_metric unrecorded; Nested selects accuracy.',
              'datasets': {}}
    for dataset in ('CMUMOSI', 'CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix'):
        classification = dataset.startswith('IEMOCAP')
        keys = ('ACC', 'UA') if classification else ('ACC', 'WF1')
        models = {}
        for model in ('Flat', 'Nested'):
            seeds = []
            for seed in (66, 67, 68):
                folds = []
                for fold in (range(1, 6) if classification else [1]):
                    root = folder(dataset, model, seed, fold)
                    metrics = json.loads((root / 'metrics.json').read_text())
                    rates = {}
                    hashes = {}
                    for rate in RATES:
                        path = root / f'predictions_miss_{rate.replace(".", "p")}.npz'
                        with np.load(path) as saved:
                            y, p = saved['labels'].reshape(-1), saved['predictions'].reshape(-1)
                        if classification:
                            classes = 4 if dataset == 'IEMOCAPFour' else 6
                            assert all(np.sum(y == c) > 0 for c in range(classes))
                            acc = float(np.mean(y == p))
                            ua = float(np.mean([np.mean(p[y == c] == c) for c in range(classes)]))
                            rates[rate] = {'ACC': 100 * acc, 'UA': 100 * ua}
                            assert abs(acc - metrics['test'][rate]['accuracy']) < 1e-10
                        else:
                            # Keep the original stored sentiment filtering/threshold metrics.
                            rates[rate] = {'ACC': 100 * metrics['test'][rate]['accuracy'],
                                           'WF1': 100 * metrics['test'][rate]['weighted_f1']}
                        hashes[rate] = hashlib.sha256(path.read_bytes()).hexdigest()
                    folds.append({'fold': fold, 'source': str(root), 'rates': rates,
                                  'prediction_sha256': hashes})
                per_rate = {r: {k: st.mean(f['rates'][r][k] for f in folds) for k in keys} for r in RATES}
                seeds.append({'seed': seed, 'folds': folds, 'rates': per_rate,
                              'mean8': {k: st.mean(per_rate[r][k] for r in RATES) for k in keys},
                              'high': {k: st.mean(per_rate[r][k] for r in RATES[5:]) for k in keys}})
            models[model] = {'seeds': seeds, 'rates': {r: {k: st.mean(s['rates'][r][k] for s in seeds) for k in keys} for r in RATES}}
            for group in ('mean8', 'high'):
                models[model][group] = {k: {'mean': st.mean(s[group][k] for s in seeds),
                                          'sd': st.stdev(s[group][k] for s in seeds)} for k in keys}
        report['datasets'][dataset] = models
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
