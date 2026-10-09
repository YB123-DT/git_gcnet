"""Read-only analysis of frozen-memory artifacts, including downloaded shards.

No model loading, fitting, inference, or validation-set selection occurs here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


LABEL = 'TEST-ORACLE INTERNAL DIAGNOSTIC'
RATES = tuple(round(i / 10, 1) for i in range(8))
HIGH = (.5, .6, .7)
METRICS = ('mse', 'weighted_f1_nonzero', 'acc_nonzero')
TASKS = ('current', 'previous', 'preceding3_mean', 'current_minus_previous')


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Mismatched or nonfinite predictions/targets')
    use = y != 0
    truth, pred = y[use] > 0, p[use] > 0
    f1 = 0.
    for value in (False, True):
        tp = np.sum((truth == value) & (pred == value))
        denominator = 2 * tp + np.sum((truth != value) & (pred == value)) + np.sum((truth == value) & (pred != value))
        f1 += (2 * tp / denominator if denominator else 0.) * np.sum(truth == value)
    return dict(n=len(y), nonneutral_n=int(use.sum()), mse=mean((y-p)**2),
                weighted_f1_nonzero=float(f1/len(truth)) if len(truth) else None,
                acc_nonzero=float(np.mean(truth == pred)) if len(truth) else None)


def transitions(y, before, after):
    y, before, after = map(np.asarray, (y, before, after))
    valid = y != 0
    a, b = before > 0, after > 0
    return dict(n=len(y), nonneutral_n=int(valid.sum()),
                flips=int(np.sum(valid & (a != b))),
                corrections=int(np.sum(valid & (a != (y > 0)) & (b == (y > 0)))),
                harms=int(np.sum(valid & (a == (y > 0)) & (b != (y > 0)))))


def prediction_index(data):
    ids = list(zip(data['test_conversation'].astype(str), data['test_utterance'].astype(int)))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction identity')
    return {key: i for i, key in enumerate(ids)}


def compare_predictions(before, after):
    a, b = prediction_index(before), prediction_index(after)
    ids = sorted(a.keys() & b.keys())
    ai, bi = [a[k] for k in ids], [b[k] for k in ids]
    y, z = before['test_target'][ai], after['test_target'][bi]
    if not np.array_equal(y, z) or not np.array_equal(before['test_row'][ai], after['test_row'][bi]):
        raise ValueError('Matched identities have inconsistent targets or original rows')
    p, q = before['test_prediction'][ai], after['test_prediction'][bi]
    result = transitions(y, p, q)
    result.update(loss_before_minus_after=mean((y-p)**2-(y-q)**2),
                  before_unmatched=len(a)-len(ids), after_unmatched=len(b)-len(ids))
    return result


def aggregate_probes(rows, identity='probe', value_keys=METRICS):
    """Average rates within each probe seed first; seed SD is sample SD."""
    aggregated = []
    for scope, expected in [('mean8', RATES), ('high', HIGH)]:
        groups = defaultdict(list)
        for row in rows:
            if row['rate'] in expected:
                groups[row['seed'], row['task'], row[identity]].append(row)
        for (seed, task, probe), values in sorted(groups.items()):
            rates = sorted({v['rate'] for v in values})
            if len(rates) != len(values):
                raise ValueError('Duplicate rate/seed/task/probe observation')
            aggregated.append(dict(scope=scope, seed=seed, task=task, **{identity: probe},
                                   rates_n=len(rates), expected_rates_n=len(expected), rates=rates,
                                   complete=tuple(rates) == expected,
                                   **{k: mean(v[k] for v in values) for k in value_keys}))
    return aggregated


def seed_statistics(rows, keys, value_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for identity, values in sorted(groups.items(), key=lambda x: str(x[0])):
        entry = dict(zip(keys, identity))
        seeds = sorted(v['seed'] for v in values)
        entry.update(seeds=seeds, seeds_n=len(seeds), three_probe_seeds_complete=seeds == [66, 67, 68])
        if 'rates' in values[0]:
            entry['comparable_rate_coverage'] = all(v['rates'] == values[0]['rates'] for v in values)
            entry['rate_coverage_by_seed'] = {str(v['seed']): v['rates'] for v in values}
        for key in value_keys:
            array = [v[key] for v in values if v.get(key) is not None]
            entry[key+'_mean'] = mean(array)
            entry[key+'_sd'] = float(np.std(array, ddof=1)) if len(array) > 1 else None
        result.append(entry)
    return result


def cluster_interval(rows, resamples=500):
    """Resample conversations within one rate, retaining all their target rows."""
    if not rows or resamples <= 0:
        return None
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row['conversation_id'])].append(row['delta_mse_delete_minus_control'])
    totals = np.array([np.sum(v) for v in grouped.values()])
    counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(66)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(len(totals), size=len(totals))
        values.append(float(totals[chosen].sum()/counts[chosen].sum()))
    return np.quantile(values, [.025, .975]).tolist()


def intervention_summary(rows, rate, coverage, bootstrap):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row['family'], str(row['conversation_id']), int(row['utterance_index']))
        if identity in seen:
            raise ValueError('Duplicate intervention target/family at a rate')
        seen.add(identity)
        if float(row['rate']) != rate or row['split'] != 'test':
            raise ValueError('Intervention rate/split mismatch')
        groups[row['family'], 'all', 'all'].append(row)
        groups[row['family'], str(row['pattern']), 'all'].append(row)
        if row['family'].startswith('remove_T_vs_') and row['lag_mismatch'] == 0:
            groups[row['family'], 'all', 'exact_lag'].append(row)
            groups[row['family'], str(row['pattern']), 'exact_lag'].append(row)
    result = []
    for (family, pattern, subset), values in sorted(groups.items()):
        y = np.array([v['y'] for v in values])
        predictions = {arm: np.array([v['pred_'+arm] for v in values]) for arm in ('real', 'delete', 'control')}
        stats = dict(rate=rate, family=family, pattern=pattern, subset=subset, n=len(values),
                     conversations_n=len({v['conversation_id'] for v in values}),
                     mean_lag_difference=mean(v['lag_mismatch'] for v in values),
                     mean_lag_delete=mean(v['lag_delete'] for v in values),
                     mean_lag_control=mean(v['lag_control'] for v in values),
                     local_max_error=max(v['current_local_max_error'] for v in values))
        for arm, p in predictions.items():
            name = 'original' if arm == 'real' else arm
            stats.update({name+'_'+k: v for k, v in metrics(y, p).items()})
            if arm != 'real':
                stats.update({arm+'_'+k: v for k, v in transitions(y, predictions['real'], p).items()})
                stats['delta_mse_'+arm] = mean((y-p)**2 - (y-predictions['real'])**2)
                for block in ('base', 'gap', 'gap_all'):
                    for norm in ('abs', 'rel'):
                        stats[f'{block}_{arm}_{norm}_mean'] = mean(v.get(f'{block}_drift_{arm}', {}).get(norm) for v in values)
        delta = (y-predictions['delete'])**2 - (y-predictions['control'])**2
        stats['delta_mse_delete_minus_control'] = mean(delta)
        ci_rows = [dict(conversation_id=v['conversation_id'], delta_mse_delete_minus_control=float(d)) for v, d in zip(values, delta)]
        stats['delta_mse_delete_minus_control_ci95'] = cluster_interval(ci_rows, bootstrap)
        family_coverage = [c.get('families', {}).get(family, {}) for c in coverage]
        eligible = sum(c.get('eligible', 0) for c in family_coverage)
        skipped = sum(c.get('skipped', 0) for c in family_coverage)
        stats.update(family_eligible=eligible, family_skipped=skipped,
                     coverage_fraction=len(values)/(eligible+skipped) if eligible+skipped else None,
                     coverage_denominator='all targets for family; pattern-specific denominator unavailable')
        result.append(stats)
    families = {name for c in coverage for name in c.get('families', {})}
    present = {r['family'] for r in result}
    for family in sorted(families-present):
        reasons = defaultdict(int)
        for batch in coverage:
            for reason, count in batch.get('families', {}).get(family, {}).get('reasons', {}).items():
                reasons[reason] += count
        result.append(dict(rate=rate, family=family, pattern='all', subset='all', n=0,
                           status='unavailable' if reasons.get('speaker_unavailable') else 'skipped_no_eligible_pairs',
                           skipped_reasons=dict(reasons)))
    return result


def csv_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def analyze(roots, output, bootstrap=500):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(label=LABEL, backbone_seed=66, probe_seeds=[66, 67, 68],
                  interpretation='Three probe seeds, not three independently trained backbones. Test-selected probe epochs are internal diagnostics, not independent generalization.',
                  probe_capacity=dict(A=33409, B=33089, C=33089,
                                      caveat='Approximately capacity-matched heads; frozen random 512-to-64 per-slot projection is a bottleneck. Null probe gains do not establish absence of information in memory.'),
                  score_targets=list(TASKS), sign_caveat='Sign metrics for historical and difference score targets are not current sentiment classification or evidence of relational reasoning.',
                  macro_definition=dict(mean8=list(RATES), high=list(HIGH), weighting='equal rate weight, then probe seed mean/sample SD'),
                  bootstrap=dict(resamples=bootstrap, unit='conversation within each rate; no pooled-rate confidence intervals', seed=66),
                  completed_rates=[], incomplete=[], skipped=[], verification=[], roots=[],
                  probes=[], probe_comparisons=[], interventions=[], intervention_coverage=[], original_test=[])
    seen = set()
    for root in map(Path, roots):
        status = read(root/'STATUS.json')
        report['roots'].append(dict(root=str(root.resolve()), status=status.get('status'), check_only=status.get('check_only', False)))
        recorded = {float(r['rate']): r for r in status.get('records', [])}
        for rate in sorted(set(map(float, status.get('rates', recorded.keys())))):
            directory = root / ('rate_'+f'{rate:.1f}'.replace('.', 'p'))
            required = [directory/'RESULT.json', directory/'probes/summary.json', directory/'interventions.json']
            if rate not in recorded or not all(p.exists() for p in required):
                report['incomplete'].append(dict(root=str(root), rate=rate, reason='missing completion record or required artifacts'))
                continue
            result, probes, interventions = map(read, required)
            if float(result['rate']) != rate:
                raise ValueError('RESULT rate mismatch')
            missing = []
            for run in probes['runs']:
                if run.get('status', '').startswith('skipped'):
                    continue
                path = directory/'probes'/f"seed{run['seed']}"/run['task']/run['probe']/'predictions.npz'
                if not path.exists():
                    missing.append(str(path))
            if missing:
                report['incomplete'].append(dict(root=str(root), rate=rate, reason='missing local prediction artifacts; entire rate excluded', paths=missing))
                continue
            if rate in seen:
                raise ValueError(f'Duplicate completed rate {rate}; do not pool duplicate shards')
            seen.add(rate)
            provenance = read(root/'PROVENANCE.json') if (root/'PROVENANCE.json').exists() else {}
            frozen = result.get('frozen_unchanged') is True and provenance.get('model_frozen', result.get('model_frozen')) is True
            original_f1 = result.get('original_test', {}).get('weighted_f1_nonzero')
            reference_f1 = result.get('baseline_reference_weighted_f1')
            parity = (result.get('baseline_parity_verified') is True and original_f1 is not None
                      and reference_f1 is not None and abs(original_f1-reference_f1) <= 1e-10)
            report['verification'].append(dict(rate=rate, frozen_verified=frozen, baseline_parity_verified=parity,
                                                checkpoint=result.get('checkpoint'), checkpoint_sha256=result.get('checkpoint_sha256')))
            report['original_test'].append(dict(rate=rate, **result.get('original_test', {})))
            cache = {}
            for run in probes['runs']:
                if run.get('status', '').startswith('skipped'):
                    report['skipped'].append(dict(rate=rate, **run))
                    continue
                seed, task, probe = run['seed'], run['task'], run['probe']
                if run.get('selection_split') != 'test':
                    raise ValueError('Unexpected probe selection split')
                path = directory/'probes'/f'seed{seed}'/task/probe/'predictions.npz'
                with np.load(path, allow_pickle=False) as loaded:
                    data = dict(loaded)
                prediction_index(data)
                cache[seed, task, probe] = data
                report['probes'].append(dict(rate=rate, seed=seed, task=task, probe=probe,
                                              weights=run.get('weights'), predictions=str(path.resolve()),
                                              **metrics(data['test_target'], data['test_prediction'])))
            expected = {(s, t, p) for s in probes.get('protocol', {}).get('seeds', [66, 67, 68]) for t in TASKS for p in ('A', 'B', 'C')}
            skipped_keys = {(r['seed'], r['task'], p) for r in probes['runs'] if r.get('status', '').startswith('skipped') for p in ('A', 'B', 'C')}
            for seed, task, probe in sorted(expected-cache.keys()-skipped_keys):
                report['incomplete'].append(dict(rate=rate, seed=seed, task=task, probe=probe, reason='absent completed probe'))
            for seed, task, probe in sorted(cache):
                if probe not in ('A', 'C') or (seed, task, 'B') not in cache:
                    continue
                report['probe_comparisons'].append(dict(rate=rate, seed=seed, task=task, comparison=probe+'->B',
                    **compare_predictions(cache[seed, task, probe], cache[seed, task, 'B'])))
            report['interventions'].extend(intervention_summary(interventions['rows'], rate, interventions['coverage'], bootstrap))
            report['intervention_coverage'].append(dict(rate=rate, coverage=interventions['coverage']))
            report['completed_rates'].append(rate)
    report['completed_rates'].sort()
    report['probe_macro'] = aggregate_probes(report['probes'])
    report['probe_seed_statistics'] = seed_statistics(report['probes'], ['rate', 'task', 'probe'], METRICS)
    report['probe_macro_seed_statistics'] = seed_statistics(report['probe_macro'], ['scope', 'task', 'probe'], METRICS)
    report['comparison_seed_statistics'] = seed_statistics(report['probe_comparisons'], ['rate', 'task', 'comparison'],
                                                          ['loss_before_minus_after', 'corrections', 'harms', 'flips'])
    report['comparison_macro'] = aggregate_probes(report['probe_comparisons'], 'comparison',
                                                 ('loss_before_minus_after', 'corrections', 'harms', 'flips'))
    report['comparison_macro_seed_statistics'] = seed_statistics(report['comparison_macro'], ['scope', 'task', 'comparison'],
                                                                 ['loss_before_minus_after', 'corrections', 'harms', 'flips'])
    report['partial'] = (tuple(report['completed_rates']) != RATES or bool(report['incomplete']) or bool(report['skipped'])
                         or any(r['check_only'] or r['status'] != 'complete' for r in report['roots'])
                         or any(not r['three_probe_seeds_complete'] for r in report['probe_seed_statistics']))
    report['verification_passed'] = bool(report['verification']) and all(r['frozen_verified'] and r['baseline_parity_verified'] for r in report['verification'])
    (output/'SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for table in ('probes', 'probe_macro', 'probe_seed_statistics', 'probe_macro_seed_statistics', 'probe_comparisons', 'comparison_seed_statistics', 'comparison_macro', 'comparison_macro_seed_statistics', 'original_test', 'interventions', 'verification', 'incomplete', 'skipped'):
        csv_table(output/(table+'.csv'), report[table])
    lines = [f'# {LABEL}', '', f"Partial: {report['partial']}. Completed rates: {report['completed_rates']}. Freeze and baseline parity verified: {report['verification_passed']}.", '',
             report['interpretation'], '', 'Frozen original backbone seed 66. Probe gradients use train; best probe epochs use minimum Test MSE. No validation selection.', '',
             'Probe heads are approximately capacity matched: A 33,409 parameters; B/C 33,089. The fixed random 512→64 per-slot projection limits what probes can recover; null gains do not establish absent memory information.', '',
             report['sign_caveat'], '', 'Metrics use raw score MSE and nonzero-target W-F1/ACC (fractions). Flips, corrections and harms exclude zero targets. A/C→B compare matched conversation/utterance identities and original rows.', '',
             'mean8 = rates 0.0–0.7; high = 0.5–0.7. Available rates receive equal weight; coverage and seed SD are reported explicitly. Incomplete aggregates are descriptive only.', '',
             f'Intervention intervals use {bootstrap} conversation-cluster bootstrap resamples separately per rate. Exact-lag modality comparisons are separate from all eligible pairs. Coverage and skipped reasons are preserved in SUMMARY.json.', '',
             '## Current-score probe results', '', '| Scope | Probe | MSE mean ± SD | W-F1 mean ± SD | ACC mean ± SD | Seeds |', '|---|---|---|---|---|---|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.6f}'
    for row in report['probe_macro_seed_statistics']:
        if row['task'] == 'current':
            values = [fmt(row[k+'_mean'])+' ± '+fmt(row[k+'_sd']) for k in METRICS]
            lines.append('| '+' | '.join([row['scope'], row['probe'], *values, str(row['seeds_n'])])+' |')
    lines.extend(['', 'Per-rate scores, all four score targets, matched correction/harm counts, intervention families/patterns and integrity checks are in the CSV tables and SUMMARY.json. These diagnostics alone do not establish performance improvement or memory reasoning.'])
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=500, help='conversation resamples per rate; 0 disables')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    result = analyze(args.roots, args.output, args.bootstrap)
    print(json.dumps(dict(completed_rates=result['completed_rates'], partial=result['partial'], verification_passed=result['verification_passed'])))
    if result['completed_rates'] and not result['verification_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
