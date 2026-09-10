"""Summarize complete 100-epoch histories, never the mean-selected checkpoint."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.osram_causal_readout_20260910.run import (
    FULL, ROOT, RATES, SEEDS, VARIANTS, training_settings, write_json,
)


def extract_best(history):
    if len(history) != 100 or [row.get('epoch') for row in history] != list(range(1, 101)):
        raise ValueError('Expected exactly 100 consecutive epochs')
    best = {}
    for row in history:
        metrics = row.get('test_oracle', {})
        if set(metrics) != set(RATES):
            raise ValueError('Missing or unexpected test_oracle rates')
        for rate in RATES:
            value = metrics[rate].get('weighted_f1')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Nonfinite or missing weighted_f1')
            if not 0 <= value <= 1:
                raise ValueError('weighted_f1 outside [0, 1]')
            if rate not in best or value > best[rate]['weighted_f1']:
                best[rate] = dict(epoch=row['epoch'], weighted_f1=value)
    return best


def mask_hashes(metrics):
    hashes = metrics.get('mask_sha256', {})
    if set(hashes) != set(RATES) or any(
            not isinstance(value, str) or len(value) != 64 or
            any(char not in '0123456789abcdef' for char in value)
            for value in hashes.values()):
        raise ValueError('Missing or malformed test mask hashes')
    return hashes


def collect(root=ROOT, full=FULL):
    root, full = Path(root), Path(full)
    report = dict(status='pending', rows=[], pending=[], invalid=[], aggregates=[], overall=[])
    loaded = {}
    for seed in SEEDS:
        for variant in ('full',) + VARIANTS:
            path = (full if variant == 'full' else root / 'mosi' / variant) / f'seed_{seed}'
            task = dict(variant=variant, seed=seed, path=str(path))
            required = ['config.json', 'history.json', 'metrics.json']
            if variant != 'full':
                required.append('PROVENANCE.json')
            if any(not (path / name).exists() for name in required):
                report['pending'].append(dict(task, reason='Required artifacts not yet present'))
                continue
            try:
                if variant != 'full':
                    provenance = json.loads((path / 'PROVENANCE.json').read_text())
                    if provenance.get('status') == 'failed':
                        raise ValueError(f"Run failed: {provenance.get('error', 'unspecified')}")
                    if provenance.get('status') != 'complete':
                        report['pending'].append(dict(task, reason='Run has not marked completion'))
                        continue
                cfg = json.loads((path / 'config.json').read_text())
                if variant == 'full':
                    training_settings(cfg, VARIANTS[0], seed)
                else:
                    reference = loaded.get(('full', seed))
                    if reference is None:
                        report['pending'].append(dict(task, reason='Full reference unavailable'))
                        continue
                    expected = training_settings(reference['config'], variant, seed)
                    # New TrainConfig fields may be materialized with their defaults.
                    defaults = {'completion_path': 'none', 'b2_base_checkpoint': None,
                                'b2_pretrain_checkpoint': None}
                    if dict(defaults, **cfg) != dict(defaults, **expected):
                        raise ValueError('Configuration differs beyond osram_emotion_ablation')
                best = extract_best(json.loads((path / 'history.json').read_text()))
                hashes = mask_hashes(json.loads((path / 'metrics.json').read_text()))
                if variant != 'full' and hashes != reference['hashes']:
                    raise ValueError('Test mask hashes differ from inherited Full')
                loaded[(variant, seed)] = dict(config=cfg, hashes=hashes, best=best)
                report['rows'].extend(dict(variant=variant, seed=seed, rate=rate,
                                           mask_sha256=hashes[rate], **best[rate]) for rate in RATES)
            except (ValueError, KeyError, TypeError) as error:
                report['invalid'].append(dict(task, reason=str(error)))
    if report['invalid']:
        report['status'] = 'invalid'
    elif not report['pending']:
        report['status'] = 'complete'
        for variant in ('full',) + VARIANTS:
            for rate in RATES:
                values = [loaded[(variant, seed)]['best'][rate]['weighted_f1'] for seed in SEEDS]
                baseline = [loaded[('full', seed)]['best'][rate]['weighted_f1'] for seed in SEEDS]
                report['aggregates'].append(dict(variant=variant, rate=rate, n=len(values),
                                                  mean=statistics.mean(values),
                                                  std=statistics.stdev(values),
                                                  delta_full=statistics.mean(a-b for a, b in zip(values, baseline))))
            for label, rates in (('all8', RATES), ('high', ('0.5', '0.6', '0.7'))):
                values = [statistics.mean(loaded[(variant, seed)]['best'][rate]['weighted_f1']
                                          for rate in rates) for seed in SEEDS]
                baseline = [statistics.mean(loaded[('full', seed)]['best'][rate]['weighted_f1']
                                            for rate in rates) for seed in SEEDS]
                deltas = [a - b for a, b in zip(values, baseline)]
                report['overall'].append(dict(variant=variant, rates=label, n=len(values),
                                             mean=statistics.mean(values), std=statistics.stdev(values),
                                             delta_full=statistics.mean(deltas),
                                             positive_seed_count=sum(delta > 0 for delta in deltas)))
    return report


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', report)
    with (output / 'per_seed_rate.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variant', 'seed', 'rate', 'epoch', 'weighted_f1', 'mask_sha256'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(report['rows'])
    lines = ['# MOSI causal readout ablation', '',
             '**INTERNAL DIAGNOSTIC ONLY / NOT A FORMAL PAPER RESULT**', '',
             f"Status: **{report['status']}**.", '',
             'Protocol: independently maximize test-oracle weighted F1 across all 100 epochs for each seed/rate; earliest epoch wins ties.',
             'Full is inherited, not rerun. Checkpoint best.pt uses an 8-rate mean and is not used for this table.',
             'Scores and paired deltas are in percentage points; std is sample std across five seeds.', '']
    if report['aggregates']:
        lines += ['| Variant | Missing rate | n | F1 mean ± std | Δ Full |',
                  '| --- | --- | --- | --- | --- |']
        for row in report['aggregates']:
            lines.append(f"| {row['variant']} | {row['rate']} | {row['n']} | {100*row['mean']:.3f} ± {100*row['std']:.3f} | {100*row['delta_full']:+.3f} |")
        lines += ['', '## Overall (seed-first averages)', '',
                  'all8 averages rates 0.0–0.7; high averages rates 0.5, 0.6, 0.7 within each seed first. Positive seeds have a strictly positive paired average delta versus Full.', '',
                  '| Variant | Rates | n | F1 mean ± std | Δ Full | Positive seeds |',
                  '| --- | --- | --- | --- | --- | --- |']
        for row in report['overall']:
            lines.append(f"| {row['variant']} | {row['rates']} | {row['n']} | {100*row['mean']:.3f} ± {100*row['std']:.3f} | {100*row['delta_full']:+.3f} | {row['positive_seed_count']}/5 |")
    else:
        lines += ['Aggregate means and deltas withheld until all 20 seed/variant records are valid and complete.', '']
    for kind in ('pending', 'invalid'):
        if report[kind]:
            lines += [f'## {kind.title()}', '']
            lines += [f"- {row['variant']} seed {row['seed']}: {row['reason']}" for row in report[kind]]
            lines.append('')
    lines += ['## Selected epochs (completed, validated records only)', '',
              '| Variant | Seed | Rate | Epoch | F1 |', '| --- | --- | --- | --- | --- |']
    lines += [f"| {row['variant']} | {row['seed']} | {row['rate']} | {row['epoch']} | {100*row['weighted_f1']:.3f} |" for row in report['rows']]
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--full-root', type=Path, default=FULL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = collect(args.root, args.full_root)
    write_report(report, args.output or args.root / 'summary')
    print(f"status={report['status']} rows={len(report['rows'])} pending={len(report['pending'])} invalid={len(report['invalid'])}")
    if report['invalid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
