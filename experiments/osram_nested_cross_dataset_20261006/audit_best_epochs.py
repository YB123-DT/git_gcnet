"""Read selected checkpoint epochs only; no training, inference or reselection."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median, mean


def folder(root, dataset, model, seed, fold):
    if model == 'Nested':
        return root / f'osram_nested_cross_dataset_20261006/{dataset}/seed_{seed}/fold_{fold}'
    if dataset == 'CMUMOSEI':
        return root / f'osram_mosei_cfg84_nojepa_20260919/seed_{seed}'
    classes = 4 if dataset == 'IEMOCAPFour' else 6
    return root / f'osram_iemocap_cfg84_nojepa_5session_20260919/iemocap{classes}/seed_{seed}/fold_{fold}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data2/yb/remote_experiments'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, aggregates = [], []
    for dataset in ('CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix'):
        for model in ('Flat', 'Nested'):
            epochs = []
            for seed in (66, 67, 68):
                for fold in ([1] if dataset == 'CMUMOSEI' else range(1, 6)):
                    path = folder(args.root, dataset, model, seed, fold)
                    metrics = json.loads((path / 'metrics.json').read_text())
                    config = json.loads((path / 'config.json').read_text())
                    assert config['epochs'] == 100
                    assert metrics['selection_protocol'] == 'per-rate-test-oracle'
                    selected = metrics['selected_epoch_by_rate']
                    values = [selected[str(i / 10)] for i in range(8)]
                    assert all(isinstance(e, int) and 1 <= e <= 100 for e in values)
                    epochs.extend(values)
                    records.append(dict(dataset=dataset, model=model, seed=seed, fold=fold,
                                        selected_epoch_by_rate=selected,
                                        selection_metric=metrics.get('selection_metric', 'unrecorded'),
                                        source=str(path), metrics_sha256=hashlib.sha256((path / 'metrics.json').read_bytes()).hexdigest()))
            aggregates.append(dict(dataset=dataset, model=model, n=len(epochs), median=median(epochs),
                                   mean=mean(epochs), minimum=min(epochs), maximum=max(epochs),
                                   epoch90_to100=sum(e >= 90 for e in epochs),
                                   epoch100=sum(e == 100 for e in epochs)))
    summary = dict(label='INTERNAL DIAGNOSTIC ONLY', training_runs=0, inference_runs=0,
                   caveat='Historical Flat selection metric unrecorded; Nested IEMOCAP selects ACC, MOSEI selects W-F1. These are stored selected epochs, not reselection by UA.',
                   aggregation='Descriptive distribution of selected epochs over seed/fold/rate; not a shared global BEST epoch or optimal stopping recommendation',
                   records=records, aggregates=aggregates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Flat versus original Nested: other datasets BEST epochs', '', 'INTERNAL DIAGNOSTIC ONLY', '',
             summary['caveat'], '', summary['aggregation'], '',
             '| Dataset | Model | N | Median | Range | Epoch90–100 | Epoch100 |',
             '|---|---|---:|---:|---|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['n']} | {row['median']} | {row['minimum']}–{row['maximum']} | {row['epoch90_to100']} | {row['epoch100']} |")
    for dataset in ('CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix'):
        lines.extend(['', f'## {dataset}: exact stored epochs', '',
                      '| Seed | Fold | Model | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 |',
                      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in sorted((r for r in records if r['dataset'] == dataset), key=lambda r:(r['seed'], r['fold'], r['model'])):
            values = ' | '.join(str(row['selected_epoch_by_rate'][str(i / 10)]) for i in range(8))
            lines.append(f"| {row['seed']} | {row['fold']} | {row['model']} | {values} |")
    lines.extend(['', 'Metrics and config files from all66 runs were read. No training, inference or checkpoint reselection.',
                  'IEMOCAP: five folds per seed, so120 selected epochs per model; MOSEI:24 per model.',
                  'Source paths and metric hashes are saved in SUMMARY.json.', ''])
    (args.output / 'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
