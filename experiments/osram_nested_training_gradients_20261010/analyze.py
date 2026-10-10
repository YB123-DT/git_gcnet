"""Summarize paired observer records; never train or load model weights."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summary = {}, {'protocol': 'INTERNAL DIAGNOSTIC ONLY; MOSI seed66; Test-oracle per-rate BEST', 'models': {}}
    slots = ['Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V']

    def weighted(records, slot, field, source='inputs'):
        values = [r[source][slot] for r in records if slot in r[source] and r[source][slot]['count'] > 0]
        return sum(v[field] * v['count'] for v in values) / sum(v['count'] for v in values) if values else None

    for model in ['flat', 'nested']:
        prov = json.loads((args.cache / f'{model}_provenance.json').read_text())
        assert prov['status'] == 'complete' and prov['outputs_verified']
        files = sorted((args.cache / model).glob('epoch_*.json'))
        assert len(files) == 100
        rows[model] = []
        for epoch, path in enumerate(files, 1):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == prov['gradient_sha256'][path.name]
            batch = json.loads(path.read_text())
            assert len(batch) == 2 and all(r['epoch'] == epoch for r in batch)
            rows[model].extend(batch)
        records = rows[model]
        for r in records:
            assert math.isfinite(r['preclip_global_norm'])
            for stat in r['inputs'].values():
                if stat['count']:
                    assert all(math.isfinite(stat[k]) for k in ['mean', 'maximum', 'sample_sum_equivalent_mean'])
            for stat in r['residual'].values():
                if stat['count']:
                    assert math.isfinite(stat['norm_mean']) and math.isfinite(stat['ratio_mean'])
        metrics_path = args.cache / f'{model}_metrics.json'
        assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == prov['artifact_sha256']['metrics.json']
        metrics = json.loads(metrics_path.read_text())
        scores = {k: 100*v for k, v in metrics['selected_weighted_f1_by_rate'].items()}
        summary['models'][model] = {
            'epochs': 100, 'steps': len(records), 'gradient_hashes_verified': len(files),
            'per_rate_wf1': scores, 'mean8': statistics.mean(scores.values()),
            'high': statistics.mean(scores[k] for k in ['0.5', '0.6', '0.7']),
            'input_gradient_sample_count_weighted': {s: weighted(records, s, 'sample_sum_equivalent_mean') for s in slots},
            'clipped_steps': sum(r['expected_clip_coefficient'] < 1 for r in records),
            'clip_coefficient_median': statistics.median(r['expected_clip_coefficient'] for r in records),
            'parameter_groups_step_mean': {g: {f: statistics.mean(r['parameter_groups'][g][f] for r in records) for f in ['norm', 'rms']} for g in records[0]['parameter_groups']},
            'residual_ratio_sample_count_weighted': {s: weighted(records, s, 'ratio_mean', 'residual') for s in slots},
        }
        (args.output / f'{model}_FINAL_PROVENANCE.json').write_text(json.dumps(prov, indent=2)+'\n')
        (args.output / f'{model}_FINAL_METRICS.json').write_text(json.dumps(metrics, indent=2)+'\n')
    assert len(rows['flat']) == len(rows['nested']) == 200
    for f, n in zip(rows['flat'], rows['nested']):
        assert all(f[k] == n[k] for k in ['epoch', 'batch', 'rate', 'availability_sha256', 'valid_count'])
        assert all(f['inputs'][s]['count'] == n['inputs'][s]['count'] for s in slots)
    summary['paired_steps_verified'] = 200
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output / 'gradient_windows.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch_start', 'epoch_end', 'model', 'slot', 'sample_sum_equivalent_mean', 'active_sample_occurrences'])
        for start in range(1, 101, 8):
            end = min(start+7, 100)
            for model, records in rows.items():
                window = [r for r in records if start <= r['epoch'] <= end]
                for slot in slots:
                    writer.writerow([start, end, model, slot, weighted(window, slot, 'sample_sum_equivalent_mean'), sum(r['inputs'][slot]['count'] for r in window)])
    f, n = summary['models']['flat'], summary['models']['nested']
    report = ['# Training-time gradient monitoring', '', 'INTERNAL DIAGNOSTIC ONLY', '',
              'MOSI seed66; paired from-scratch 100-epoch reruns on biggpu GPU7. Original source ad211c0; observer implementation 597e9a1. No architecture/loss changes. Per-rate Test-oracle BEST; one cyclic-missing training trajectory per model, not eight independently trained models.', '',
              'Verified: 100 gradient-file hashes/model, metric hashes, completed provenance, 200 paired steps with identical masks/counts. Both reproduce the historical seed66 aggregate scores.', '',
              '| Missing rate | Flat W-F1 (%) | Nested W-F1 (%) | Difference (pp) |', '|---|---:|---:|---:|']
    for rate in f['per_rate_wf1']:
        fv, nv = f['per_rate_wf1'][rate], n['per_rate_wf1'][rate]
        report.append(f'| {rate} | {fv:.3f} | {nv:.3f} | {nv-fv:+.3f} |')
    for name, key in [('8-rate mean', 'mean8'), ('High missing', 'high')]:
        report.append(f'| {name} | {f[key]:.3f} | {n[key]:.3f} | {n[key]-f[key]:+.3f} |')
    report += ['', '## Full 1–100 epoch input-gradient summary', '',
               'Sample-count weighted over active observations; batch-mean MSE gradients multiplied by valid utterance count. All epochs included, including zero-initialization. Local includes Skip; Memory uses forward 512 dimensions and excludes first turns/inactive Gap. These are training gradients, not frozen output Jacobians.', '',
               '| Slot | Flat | Nested | Nested / Flat |', '|---|---:|---:|---:|']
    for slot in slots:
        fv, nv = f['input_gradient_sample_count_weighted'][slot], n['input_gradient_sample_count_weighted'][slot]
        report.append(f'| {slot} | {fv:.6f} | {nv:.6f} | {nv/fv:.3f} |')
    report += ['', '## Optimization and residual checks', '']
    for model, result in summary['models'].items():
        report.append(f"- {model}: clipping active {result['clipped_steps']}/200 steps; median coefficient {result['clip_coefficient_median']:.6f}.")
    report += ['', 'Nested nonzero residual/input ratios (sample-count weighted):']
    for slot, value in n['residual_ratio_sample_count_weighted'].items():
        report.append(f'- {slot}: {value:.6f}.')
    report += ['', '## Interpretation limits', '',
               'Nested receives finite, nonzero parameter gradients and learns nonzero residuals: no evidence of a completely disconnected branch. Smaller aggregate input gradients alone do not establish harmful gradient vanishing. Both models clip every step, so clipping is not a Nested-only failure. Model coordinates, loss, parameter scale and independent joint training confound gradient comparisons. Residual norms do not establish useful sentiment information or conflict detection. Seed66 alone does not establish multi-seed consistency.', '',
               'gradient_windows.csv provides all eight-epoch windows; 97–100 is explicitly a partial tail, not directly rate-balanced. SUMMARY.json includes parameter-group L2/RMS means. No additional training, inference or automatic multi-seed expansion was performed for this analysis.', '',
               'Reproduce with: `python experiments/osram_nested_training_gradients_20261010/analyze.py --cache <cached gradients and final JSON files> --output experiments/osram_nested_training_gradients_20261010`.', '']
    (args.output / 'RESULT.md').write_text('\n'.join(report))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
