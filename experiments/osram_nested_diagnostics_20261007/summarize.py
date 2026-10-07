"""Summarize saved Nested diagnostics, requiring all requested seeds/folds."""
import argparse
import json
from pathlib import Path
import statistics

DATASETS = ('CMUMOSI', 'CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix')
VARIANTS = ('full', 'local_only', 'local_base', 'gap_off', 'base_off', 'no_local')
PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')


def aggregate(rows, folds):
    values = {}
    for row in rows:
        key = (row['seed'], row['fold'])
        if key in values:
            raise ValueError('Duplicate seed/fold observation')
        values[key] = row['value']
    missing = [[s, f] for s in (66, 67, 68) for f in folds if (s, f) not in values]
    seeds = {str(s): statistics.mean(values[s, f] for f in folds) for s in (66, 67, 68)
             if all((s, f) in values for f in folds)}
    return dict(mean=statistics.mean(seeds.values()) if not missing else None,
                sd=statistics.stdev(seeds.values()) if not missing else None,
                seeds=seeds, missing=missing, complete_three_seeds=not missing)


def build(root):
    report = dict(label='INTERNAL DIAGNOSTIC ONLY; existing per-rate Test-oracle; no training',
                  aggregation='equal folds within seed, equal rates, equal seeds; sample SD across seeds',
                  datasets={})
    rows = []
    for path in sorted((root / 'results').glob('*/SUMMARY.json')):
        status = path.parent / 'STATUS.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'complete':
            continue
        row = json.loads(path.read_text())
        row['summary_path'] = str(path)
        rows.append(row)
    for dataset in DATASETS:
        selected = [r for r in rows if r['dataset'] == dataset]
        folds = (1, 2, 3, 4, 5) if dataset.startswith('IEMOCAP') else (1,)
        keys = ('weighted_f1', 'accuracy', 'unweighted_accuracy') if dataset.startswith('IEMOCAP') else ('weighted_f1', 'accuracy')
        result = dict(completed_runs=len(selected), expected_runs=3*len(folds), random_rates={}, fixed_patterns={})
        for mode, cells in [('random_rates', [str(i/10) for i in range(8)]), ('fixed_patterns', PATTERNS)]:
            for cell in cells:
                result[mode][cell] = {}
                for variant in VARIANTS:
                    result[mode][cell][variant] = {}
                    for key in keys:
                        observations = [dict(seed=r['seed'], fold=r['fold'], value=100*r[mode][cell]['metrics'][variant][key])
                                        for r in selected]
                        result[mode][cell][variant][key] = aggregate(observations, folds)
        for group, rates in [('mean8', [str(i/10) for i in range(8)]), ('high', ['0.5', '0.6', '0.7'])]:
            result[group] = {}
            for variant in VARIANTS:
                result[group][variant] = {}
                for key in keys:
                    observations = [dict(seed=r['seed'], fold=r['fold'], value=statistics.mean(
                        100*r['random_rates'][rate]['metrics'][variant][key] for rate in rates)) for r in selected]
                    result[group][variant][key] = aggregate(observations, folds)
        report['datasets'][dataset] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.root)
    (args.root / 'DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Nested 四数据集冻结诊断', '', 'INTERNAL DIAGNOSTIC ONLY', '',
             '只复用旧版 Nested checkpoint；没有训练、Flat 对照或选点变更。单位为百分数。', '',
             '完成全部指定 folds/seeds 后才生成三种子均值；未齐全显示 pending。', '']
    def fmt(value):
        return 'pending' if value['mean'] is None else f"{value['mean']:.3f} ± {value['sd']:.3f}"
    for dataset, row in report['datasets'].items():
        lines += [f'## {dataset}', '', f"完成 {row['completed_runs']}/{row['expected_runs']} runs。", '',
                  '| Readout | 8-rate W-F1 | High W-F1 |', '|---|---:|---:|']
        for variant in VARIANTS:
            lines.append(f"|{variant}|{fmt(row['mean8'][variant]['weighted_f1'])}|{fmt(row['high'][variant]['weighted_f1'])}|")
        lines += ['', '| Fixed observed | Full W-F1 | Gap off | Base off | No Local |', '|---|---:|---:|---:|---:|']
        for pattern in PATTERNS:
            metrics = row['fixed_patterns'][pattern]
            lines.append('|' + '|'.join([pattern] + [fmt(metrics[v]['weighted_f1']) for v in
                                                   ('full', 'gap_off', 'base_off', 'no_local')]) + '|')
        lines += ['', 'ACC/UA 和每率指标见 DIAGNOSTIC_SUMMARY.json；Query/Addressing/Read 明细见各 run 的 query CSV。', '']
    (args.root / 'RESULT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
