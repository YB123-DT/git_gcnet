"""Read saved MOSI predictions only; no torch/model forward or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PATTERNS = [('A', 4), ('T', 2), ('V', 1), ('AT', 6), ('AV', 5), ('TV', 3), ('ATV', 7)]


def score(y, p):
    return float(f1_score(y, p, average='weighted', zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', type=Path, default=Path('/data2/yb/remote_experiments'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, overall, provenance = [], [], []
    for seed in range(66, 71):
        sources = [args.remote_root / 'osram_forward_only_mosi_20260908' / f'seed_{seed}',
                   args.remote_root / 'osram_write_step_train_20260909/mosi' / f'seed_{seed}']
        arrays, configs, metrics = [], [], []
        for source in sources:
            file = source / 'predictions_miss_0p5.npz'
            with np.load(file, allow_pickle=False) as data:
                arrays.append({key: data[key].copy() for key in ('predictions', 'labels', 'availability')})
            configs.append(json.loads((source / 'config.json').read_text()))
            metrics.append(json.loads((source / 'metrics.json').read_text()))
            provenance.append(dict(seed=seed, source=str(source), npz_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                epoch=metrics[-1]['best_epoch'], selection_protocol=metrics[-1]['selection_protocol'],
                write_step=configs[-1].get('osram_write_step', 1.),
                full_mask_sha256=metrics[-1]['mask_sha256']['0.5']))
        for config in configs:
            config.setdefault('osram_forward_slot_reuse', False)
            config.setdefault('osram_write_step', 1.)
            assert config['mosi_task_mode'] == 'regression' and config['seed'] == seed
        assert configs[0]['osram_write_step'] == 1. and configs[1] == dict(configs[0], osram_write_step=.6)
        assert all(m['selection_protocol'] == '8-rate-mean-test-oracle' for m in metrics)
        assert metrics[0]['mask_sha256']['0.5'] == metrics[1]['mask_sha256']['0.5']
        old, new = arrays
        for key in ('labels', 'availability'):
            assert np.array_equal(old[key], new[key]), f'Unpaired {key}: seed{seed}'
        for array in arrays:
            assert all(np.isfinite(value).all() for value in array.values())
        labels, availability = old['labels'], old['availability']
        assert availability.shape == (labels.size, 3)
        assert np.isin(availability, [0, 1]).all() and (availability.sum(1) > 0).all()
        valid = labels != 0  # Same exclusion and >0 sign threshold as original _metrics.
        y = labels > 0
        pa, pc = old['predictions'] > 0, new['predictions'] > 0
        assert pa.shape == pc.shape == labels.shape
        for pred, metric in zip((pa, pc), metrics):
            assert abs(score(y[valid], pred[valid]) - metric['test']['0.5']['weighted_f1']) < 1e-12
        code = availability @ np.array([4, 2, 1])
        masks = PATTERNS + [('ALL', None), ('NO_TEXT', -1), ('TEXT_PRESENT', -2)]
        for pattern, bit in masks:
            selected = np.ones(labels.size, dtype=bool) if bit is None else (
                availability[:, 1] == (0 if bit == -1 else 1) if bit in (-1, -2) else code == bit)
            mask = selected & valid
            n = int(mask.sum())
            assert n > 0, 'Report missing patterns as NA, not zero'
            fa, fc = score(y[mask], pa[mask]), score(y[mask], pc[mask])
            regressions = int((mask & (pa == y) & (pc != y)).sum())
            repairs = int((mask & (pa != y) & (pc == y)).sum())
            row = dict(seed=seed, pattern=pattern, n_all=int(selected.sum()), n_scored=n,
                       n_zero_excluded=int((selected & ~valid).sum()), true_positive=int(y[mask].sum()),
                       old_wf1=fa, new_wf1=fc, delta_wf1=fc-fa,
                       old_accuracy=float(accuracy_score(y[mask], pa[mask])),
                       new_accuracy=float(accuracy_score(y[mask], pc[mask])),
                       old_correct_to_new_wrong=regressions, old_wrong_to_new_correct=repairs,
                       net_extra_errors=regressions-repairs)
            for prefix, pred in [('old', pa), ('new', pc)]:
                tn, fp, fn, tp = confusion_matrix(y[mask], pred[mask], labels=[False, True]).ravel()
                row.update({f'{prefix}_{key}': int(value) for key, value in zip(('tn','fp','fn','tp'), (tn,fp,fn,tp))})
            (overall if bit is None or bit in (-1, -2) else rows).append(row)
        assert sum(r['n_scored'] for r in rows if r['seed'] == seed) == valid.sum()
        assert sum(r['net_extra_errors'] for r in rows if r['seed'] == seed) == overall[-3]['net_extra_errors']
    summaries = []
    for pattern in [x[0] for x in PATTERNS] + ['ALL','NO_TEXT','TEXT_PRESENT']:
        group = [r for r in rows + overall if r['pattern'] == pattern]
        summaries.append(dict(pattern=pattern, n_seeds=len(group),
            n_scored_mean=statistics.mean(r['n_scored'] for r in group),
            old_mean=statistics.mean(r['old_wf1'] for r in group),
            old_sd=statistics.stdev(r['old_wf1'] for r in group),
            new_mean=statistics.mean(r['new_wf1'] for r in group),
            new_sd=statistics.stdev(r['new_wf1'] for r in group),
            delta_mean=statistics.mean(r['delta_wf1'] for r in group),
            delta_sd=statistics.stdev(r['delta_wf1'] for r in group),
            positive_seeds=sum(r['delta_wf1']>0 for r in group),
            negative_seeds=sum(r['delta_wf1']<0 for r in group),
            net_extra_errors_sum=sum(r['net_extra_errors'] for r in group)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in [('per_seed_pattern.csv',rows), ('overall_and_text_groups.csv',overall),
                              ('summary.csv',summaries), ('provenance.csv',provenance)]:
        with (args.output_dir / filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n')
            writer.writeheader(); writer.writerows(records)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
