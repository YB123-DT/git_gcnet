"""Produce every MOSI/MOSEI 4x3 source-removal table from completed JSONs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.osram_12_source_evidence_audit_20261019.report import (
    _csv_text, _format, _strata, _write,
)
from .analysis import ROWS, SOURCES


RANDOM_RATES = tuple(f'{i / 10:.1f}' for i in range(8))
PERSISTENT_PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
METRICS = ('ACC', 'WF1', 'MSE')


def _load_complete(run_root):
    root = run_root.resolve()
    summary_path = root / 'SUMMARY.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete' or not summary.get('no_training') or not summary.get('backbone_unchanged'):
        raise ValueError('Only a complete unchanged-backbone frozen audit can be reported')
    if summary.get('dataset') not in ('CMUMOSI', 'CMUMOSEI', 'MOSI', 'MOSEI'):
        raise ValueError('Expected scalar MOSI/MOSEI dataset')
    expected = [('random', rate) for rate in RANDOM_RATES]
    expected += [('persistent', pattern) for pattern in PERSISTENT_PATTERNS]
    if set(summary['conditions']) != {g + '_' + n.replace('.', 'p') for g, n in expected}:
        raise ValueError('Expected exactly all 15 conditions')
    conditions, hashes = [], {'SUMMARY.json': hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    for group, name in expected:
        ref = summary['conditions'][group + '_' + name.replace('.', 'p')]
        path = (root / ref['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Condition JSON must be inside run directory')
        content = path.read_bytes()
        condition = json.loads(content)
        if (condition['source_commit'] != summary['source_commit']
                or condition['anchor_sha256'] != summary['anchor']['checkpoint_sha256']
                or condition['group'] != group or condition['condition'] != name
                or ref['group'] != group or ref['condition'] != name
                or condition['checks'] != ref['checks']
                or condition['statistics']['full_metrics'] != ref['full_metrics']):
            raise ValueError('Condition identity/metrics/checks differ from run SUMMARY')
        metrics = condition['statistics']['full_metrics']
        for metric in METRICS:
            value = metrics[metric]
            if (value is None and (metric == 'MSE' or metrics['N_polarity'] > 0)) or (
                    value is not None and not math.isfinite(value)):
                raise ValueError('Invalid completed condition metrics')
        hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        conditions.append(condition)
    return summary, conditions, hashes


def _slot_records(statistics, metadata):
    records = []
    for row in ROWS:
        for source in SOURCES:
            slot = statistics['slots'][row][source]
            record = dict(metadata, row=row, source=source, N=slot['N'], N_polarity=slot['N_polarity'])
            record.update({f'{branch}_{metric}': slot[branch + '_metrics'][metric]
                           for branch in ('full', 'removed') for metric in METRICS})
            record.update({key: slot[key] for key in ('delta_ACC', 'delta_WF1', 'delta_MSE',
                           'corrections', 'harms', 'both_correct', 'both_wrong')})
            record.update({'read_norm_' + k: v for k, v in slot['read_norm'].items() if k != 'N'})
            records.append(record)
    return records


def _metric_table(stat, metric):
    label = 'W-F1' if metric == 'WF1' else metric
    lines = [f'{label}: removal score (Δ = removed − original, percentage points for ACC/W-F1).', '',
             '| Read destination | Active N | Polarity N | Original, same row cohort | Remove history A | Remove history T | Remove history V |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in ROWS:
        reference = stat['slots'][row]['A']
        values = []
        for source in SOURCES:
            item = stat['slots'][row][source]
            value, delta = item['removed_metrics'][metric], item['delta_' + metric]
            values.append('N/A' if value is None else f'{value:.4f} ({delta:+.4f})')
        lines.append(f"| {row} | {reference['N']} | {reference['N_polarity']} | "
                     + _format(reference['full_metrics'][metric]) + ' | ' + ' | '.join(values) + ' |')
    return lines


def _read_table(stat):
    lines = ['Mean source-read L2 norm; active cohorts only.', '',
             '| Read destination | History A | History T | History V |', '|---|---:|---:|---:|']
    for row in ROWS:
        lines.append('| ' + row + ' | ' + ' | '.join(
            _format(stat['slots'][row][source]['read_norm']['mean']) for source in SOURCES) + ' |')
    return lines


def summarize_run(run_root, output_root=None):
    """Write full15 conditions, per-rate ACC/WF1 tables and separate Persistent.

    A fixed original miss=.3 checkpoint is shared across every condition. This
    is not the baseline's eight independently Test-best-selected checkpoints.
    """
    run_root = Path(run_root)
    summary, conditions, hashes = _load_complete(run_root)
    output_root = Path(output_root) if output_root is not None else run_root
    output_root.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=summary['dataset'], seed=summary['anchor'].get('seed', 66),
                fold=summary['anchor'].get('fold'), source_commit=summary['source_commit'],
                anchor_sha256=summary['anchor']['checkpoint_sha256'])
    records, slots, grouped, shapley, errors = [], [], [], [], {}
    for condition in conditions:
        stat = condition['statistics']
        info = dict(meta, group=condition['group'], condition=condition['condition'])
        records.append(dict(info, **stat['full_metrics'], mask_sha256=condition['checks'].get('availability_sha256')))
        slots.extend(_slot_records(stat, info))
        for family, name, source, cohort in _strata(stat):
            grouped.extend(_slot_records(cohort, dict(info, stratum_family=family,
                           stratum=name, stratum_source=source, cohort_N=cohort['N'])))
        for source in SOURCES:
            shapley.append(dict(info, source=source, N=stat['N'],
                               mean_negative_MSE_credit=stat['shapley']['mean'][source],
                               full_minus_zero_gain=stat['shapley']['full_minus_zero_gain'],
                               efficiency_max_abs_error=stat['shapley']['efficiency_max_abs_error']))
        for checks in (condition['checks'], *condition['checks'].get('diagnostics', [])):
            for name, value in checks.items():
                if 'error' in name and isinstance(value, (int, float)):
                    errors[name] = max(errors.get(name, 0.), float(value))

    def macro(group, names):
        selected = [r for r in records if r['group'] == group and r['condition'] in names]
        return dict(conditions=len(selected), aggregation='unweighted mean of full-condition metrics, not subgroup W-F1',
                    **{metric: (sum(r[metric] for r in selected) / len(selected)
                                if selected and all(r[metric] is not None for r in selected) else None)
                       for metric in METRICS})

    aggregates = dict(random8=macro('random', RANDOM_RATES),
                      high_missing=macro('random', ('0.5', '0.6', '0.7')),
                      persistent6=macro('persistent', PERSISTENT_PATTERNS[:-1]))
    explanation = [
        'INTERNAL DIAGNOSTIC ONLY', '',
        f"Dataset {meta['dataset']}; seed {meta['seed']}; fold {meta['fold']}; source `{meta['source_commit']}`.", '',
        f"Fixed original miss=0.3 checkpoint: `{summary['anchor'].get('checkpoint', 'see SUMMARY')}`.", '',
        f"Checkpoint SHA256: `{meta['anchor_sha256']}`.", '',
        'No training; original Flat, Local Skip, Memory, Query and task head are unchanged. '
        'One fixed checkpoint is used for every rate/pattern, not the eight per-rate selected baseline models. '
        'The anchor was previously selected by Test-oracle; these are internal frozen interventions, not independent generalization results.', '',
        'Rows are read destinations Base/Gap-A/Gap-T/Gap-V; columns are historical A/T/V Value-write sources. '
        'Each cell reports the final score after removing ONLY that one source contribution at the final Flat input. '
        'Local and all other evidence remain unchanged. Sources are algebraic joint-write Value-slot contributions, not pure modality semantics.', '',
        'ACC/W-F1 use abs(gold)>0 and prediction>0; prediction=0 is nonpositive. MSE includes all valid labels. '
        'Each Gap row uses only utterances where that target modality is currently missing; its original score is recomputed on the identical cohort. '
        'Inactive/empty rows are N/A. Do not compare rows with different cohorts as if they were full-test scores.', '',
        'Positive ACC/W-F1 Δ means removing the item improves that score; positive MSE Δ means removing it increases loss. '
        'Corrections/harms compare removed vs full on nonzero-label utterances only. '
        'No subgroup W-F1 values are averaged to reconstruct full-test W-F1.', '',
    ]
    lines = ['# OSRAM scalar 12-Source Evidence Audit', ''] + explanation
    lines += ['## Full-model metrics by condition', '', '| Condition | N | Polarity N | ACC | W-F1 | MSE |',
              '|---|---:|---:|---:|---:|---:|']
    for record in records:
        lines.append(f"| {record['group']} {record['condition']} | {record['N']} | {record['N_polarity']} | "
                     + ' | '.join(_format(record[m]) for m in METRICS) + ' |')
    for name, aggregate in aggregates.items():
        lines.append('| ' + name + ' | — | — | ' + ' | '.join(_format(aggregate[m]) for m in METRICS) + ' |')
    lines += ['', 'Aggregates are equal-weight full-condition means; Persistent ATV is separate from persistent6.', '',
              '## All 15 conditions: 4×3 source removal tables', '']
    for condition in conditions:
        lines += [f"### {condition['group']} {condition['condition']}", '']
        for metric in ('ACC', 'WF1', 'MSE'):
            lines += _metric_table(condition['statistics'], metric) + ['']
        lines += _read_table(condition['statistics']) + ['']
    lines += ['## Integrity and limits', '', 'Source-sum reconstruction maxima:', '',
              '| Check | Maximum |', '|---|---:|']
    lines += [f'| {name} | {value:.9g} |' for name, value in sorted(errors.items())]
    lines += ['', 'SLOTS.csv includes paired corrections/harms, active N, polarity N, norms and all metric deltas. '
              'SLOTS_GROUPED.csv additionally stratifies current availability, strictly-past ever-seen history and source coverage. '
              'SHAPLEY.csv uses exact three-source grouped Shapley with payoff=-per-utterance MSE (not ACC/W-F1). '
              'Independent slot effects are nonlinear and not additive; source-group Shapley credits satisfy efficiency for the fixed -MSE game. '
              'Source subtraction may be out-of-distribution. Norm/cancellation are not sentiment conflict or reliability. '
              'This single-seed diagnostic does not establish significant gains or a new method.', '']
    paths = {key: str(output_root / name) for key, name in (
        ('result', 'RESULT.md'), ('json', 'REPORT.json'), ('conditions', 'CONDITIONS.csv'),
        ('slots', 'SLOTS.csv'), ('grouped_slots', 'SLOTS_GROUPED.csv'), ('shapley', 'SHAPLEY.csv'),
        ('acc_per_rate', 'ACC_PER_RATE.md'), ('wf1_per_rate', 'WF1_PER_RATE.md'), ('persistent', 'PERSISTENT.md'))}
    machine = dict(protocol='INTERNAL DIAGNOSTIC ONLY', **meta, no_training=True,
                   input_json_sha256=hashes, aggregates=aggregates, conditions=records,
                   reconstruction_error_maxima=errors,
                   metric_definition='abs(gold)>0 and prediction>0 for ACC/WF1; all valid labels for MSE',
                   delta_definition='removed minus full on same active-row cohort',
                   anchor_protocol='single fixed previously Test-oracle-selected original Flat miss=.3 checkpoint',
                   resource=dict(seconds=summary.get('seconds'), peak_memory_mib=summary.get('peak_memory_mib')))
    _write(Path(paths['result']), '\n'.join(lines))
    _write(Path(paths['json']), json.dumps(machine, indent=2, allow_nan=False) + '\n')
    for key, values in (('conditions', records), ('slots', slots), ('grouped_slots', grouped), ('shapley', shapley)):
        _write(Path(paths[key]), _csv_text(values))
    for metric, key in (('ACC', 'acc_per_rate'), ('WF1', 'wf1_per_rate')):
        text = [f'# Random eight-rate {metric} 4×3 source-removal tables', ''] + explanation
        for condition in conditions:
            if condition['group'] == 'random':
                text += [f"## random {condition['condition']}", ''] + _metric_table(condition['statistics'], metric) + ['']
        _write(Path(paths[key]), '\n'.join(text))
    text = ['# Persistent whole-conversation source-removal tables', ''] + explanation
    for condition in conditions:
        if condition['group'] == 'persistent':
            text += [f"## persistent {condition['condition']}", '']
            for metric in ('ACC', 'WF1'):
                text += _metric_table(condition['statistics'], metric) + ['']
    _write(Path(paths['persistent']), '\n'.join(text))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_root, args.output_root), indent=2))


if __name__ == '__main__':
    main()
