"""Summarize completed frozen audit JSONs; no checkpoints or NPZs are loaded."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .analysis import ROWS, SOURCES


RANDOM_RATES = tuple(f'{index / 10:.1f}' for index in range(8))
PERSISTENT_PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
METRICS = ('ACC', 'UA', 'WF1', 'CE')


def _write(path, text):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def _csv_text(rows):
    stream = io.StringIO(newline='')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _format(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _slot_records(statistics, metadata):
    records = []
    for row in ROWS:
        for source in SOURCES:
            slot = statistics['slots'][row][source]
            record = dict(metadata, row=row, source=source, N=slot['N'])
            record.update({'read_norm_'+key: value for key, value in slot['read_norm'].items() if key != 'N'})
            record.update({key: slot[key] for key in (
                'delta_CE', 'delta_ACC', 'delta_UA', 'delta_WF1',
                'corrections', 'harms', 'both_correct', 'both_wrong')})
            for branch in ('full', 'removed'):
                record.update({branch+'_'+metric: slot[branch+'_metrics'][metric] for metric in METRICS})
            records.append(record)
    return records


def _strata(statistics):
    strata = statistics['strata']
    for family in ('availability', 'history_ever'):
        for name, cohort in strata[family].items():
            yield family, name, '', cohort
    for source, source_groups in strata['source_history'].items():
        for family, groups in source_groups.items():
            for name, cohort in groups.items():
                yield 'source_'+family, name, source, cohort


def _load_complete(run_root):
    summary_path = run_root/'SUMMARY.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete':
        raise ValueError('Only a complete audit run can be reported')
    if not summary.get('no_training') or not summary.get('backbone_unchanged'):
        raise ValueError('Frozen audit integrity checks were not completed')
    expected = [('random', rate) for rate in RANDOM_RATES]
    expected += [('persistent', pattern) for pattern in PERSISTENT_PATTERNS]
    expected_keys = {group+'_'+name.replace('.', 'p') for group, name in expected}
    if set(summary['conditions']) != expected_keys:
        raise ValueError('Expected exactly 15 completed evaluation conditions')
    conditions, input_hashes = [], {'SUMMARY.json': hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    for group, name in expected:
        key = group+'_'+name.replace('.', 'p')
        reference = summary['conditions'][key]
        path = (run_root/reference['path']).resolve()
        if not path.is_relative_to(run_root.resolve()):
            raise ValueError('Condition JSON must be inside its run directory')
        content = path.read_bytes()
        condition = json.loads(content)
        if (condition['source_commit'] != summary['source_commit']
                or condition['anchor_sha256'] != summary['anchor']['checkpoint_sha256']
                or condition['group'] != group or condition['condition'] != name
                or reference['group'] != group or reference['condition'] != name):
            raise ValueError('Condition identity differs from complete run')
        if condition['statistics']['full_metrics'] != reference['full_metrics']:
            raise ValueError('Condition metrics differ from SUMMARY')
        if condition['checks'] != reference['checks']:
            raise ValueError('Condition checks differ from SUMMARY')
        if any(condition['statistics']['full_metrics'][metric] is None
               or not math.isfinite(condition['statistics']['full_metrics'][metric]) for metric in METRICS):
            raise ValueError('Completed evaluation condition must have finite metrics')
        input_hashes[str(path.relative_to(run_root.resolve()))] = hashlib.sha256(content).hexdigest()
        conditions.append(condition)
    return summary, conditions, input_hashes


def summarize_run(run_root, output_root=None):
    """Create tables from 15 completed condition JSONs, failing closed otherwise.

    Return paths keyed result/json/conditions/slots/shapley/grouped_slots. Random
    and Persistent aggregates average conditions equally, not pooled samples.
    The CSVs retain active-cohort N and empty N/A slots as blank numeric fields.
    """
    run_root = Path(run_root)
    summary, conditions, input_hashes = _load_complete(run_root)
    output_root = Path(output_root) if output_root is not None else run_root
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset=summary['dataset'], seed=summary['anchor'].get('seed', 66),
                    fold=summary['anchor'].get('fold'), source_commit=summary['source_commit'],
                    anchor_sha256=summary['anchor']['checkpoint_sha256'])
    condition_records, slots, grouped, shapley_records = [], [], [], []
    error_maxima = {}
    for condition in conditions:
        stat = condition['statistics']
        meta = dict(metadata, group=condition['group'], condition=condition['condition'])
        condition_records.append(dict(meta, **stat['full_metrics'],
            mask_sha256=condition['checks'].get('availability_sha256'),
            seconds=condition['checks'].get('seconds'), scans=condition['checks'].get('scans')))
        slots.extend(_slot_records(stat, meta))
        for family, name, source, cohort in _strata(stat):
            grouped.extend(_slot_records(cohort, dict(meta, stratum_family=family,
                stratum=name, stratum_source=source, cohort_N=cohort['N'])))
        for source in SOURCES:
            shapley_records.append(dict(meta, source=source, N=stat['N'],
                mean_negative_CE_credit=stat['shapley']['mean'][source],
                full_minus_zero_gain=stat['shapley']['full_minus_zero_gain'],
                efficiency_max_abs_error=stat['shapley']['efficiency_max_abs_error'],
                **{f'credit_{key}': value for key, value in stat['shapley']['summary'][source].items() if key != 'N'}))
        for checks in (condition['checks'], *condition['checks'].get('diagnostics', [])):
            for name, value in checks.items():
                if 'error' in name and isinstance(value, (int, float)):
                    error_maxima[name] = max(error_maxima.get(name, 0.), float(value))

    def macro(group, names):
        selected = [row for row in condition_records if row['group'] == group and row['condition'] in names]
        if len(selected) != len(names):
            raise ValueError('Incomplete aggregate conditions')
        return dict(conditions=len(names), aggregation='unweighted mean of condition metrics',
                    **{metric: sum(row[metric] for row in selected)/len(selected) for metric in METRICS})

    aggregates = dict(random8=macro('random', RANDOM_RATES),
                      high_missing=macro('random', ('0.5', '0.6', '0.7')),
                      persistent6=macro('persistent', PERSISTENT_PATTERNS[:-1]))
    machine = dict(protocol='INTERNAL DIAGNOSTIC ONLY', **metadata, no_training=True,
        input_json_sha256=input_hashes, run_root=str(run_root.resolve()),
        aggregates=aggregates, conditions=condition_records, reconstruction_error_maxima=error_maxima,
        resource=dict(seconds=summary.get('seconds'), peak_memory_mib=summary.get('peak_memory_mib')),
        statistical_scope='Single seed66; IEMOCAPSix fold5 only; no confidence intervals or significance claims',
        source_definition='Joint-write Value-slot contribution, not pure modality memory or semantic causal attribution',
        intervention_definition='Readout source subtraction/zeroing with fixed Local, original Memory, Query and classifier; potentially OOD',
        correction_definition='Original full wrong -> removed-source correct; harms are full correct -> removed-source wrong',
        shapley_definition='Exact three-player source-group Shapley, payoff=-CE, source bits A1/T2/V4')
    paths = {key: str(output_root/name) for key, name in (
        ('result', 'RESULT.md'), ('json', 'REPORT.json'), ('conditions', 'CONDITIONS.csv'),
        ('slots', 'SLOTS.csv'), ('shapley', 'SHAPLEY.csv'), ('grouped_slots', 'SLOTS_GROUPED.csv'))}
    lines = ['# OSRAM 12项历史来源贡献分析（12-Source Evidence Audit）', '',
        'INTERNAL DIAGNOSTIC ONLY', '',
        f"Dataset: {metadata['dataset']}; seed66; fold {metadata['fold']}.", '',
        f"Source Git SHA: `{metadata['source_commit']}`.", '',
        f"Fixed original miss=0.3 checkpoint: `{summary['anchor'].get('checkpoint', 'see SUMMARY anchor')}`.", '',
        f"Checkpoint SHA256: `{metadata['anchor_sha256']}`.", '',
        'No training, no TASR Adapter, no parameter updates. One fixed original Flat checkpoint is used across all conditions. '
        'Random conditions replay saved original GCNet masks; Persistent conditions use fixed whole-conversation visible patterns.', '',
        'Only seed66 is analyzed; IEMOCAP-6 is fold5 only. No confidence intervals or significance claims. '
        'The original checkpoint was selected by historical Test-oracle, not independent test generalization.', '',
        '## 1. Full-model condition metrics', '',
        'ACC, seven/six-class explicit Macro Recall (UA), and W-F1 are percentages; CE is in nats. '
        'Aggregates are unweighted means across conditions, never pooled sample scores. Persistent ATV is separate from the six-pattern mean.', '',
        '| Condition | N | ACC | UA | W-F1 | CE |',
        '|---|---:|---:|---:|---:|---:|']
    for record in condition_records:
        label = ('Random ' if record['group'] == 'random' else 'Persistent ')+record['condition']
        lines.append('| '+label+' | '+str(record['N'])+' | '+' | '.join(_format(record[key]) for key in METRICS)+' |')
    for name, aggregate in aggregates.items():
        lines.append('| '+name+' | — | '+' | '.join(_format(aggregate[key]) for key in METRICS)+' |')
    lines += ['', '## 2. Reconstruction and integrity', '',
        'The maxima below are collected across every batch and condition, including absolute and relative source-sum errors when recorded. '
        'Exact original readout, fixed current input/query, unchanged weights and RNG checks are saved in SUMMARY/per-condition JSON.', '',
        '| Check | Maximum error |', '|---|---:|']
    lines += [f'| {name} | {value:.9g} |' for name, value in sorted(error_maxima.items())]
    lines += ['', '## 3. Saved Value-source state strength', '',
        'These are source-state Frobenius norms of the decayed, strictly prewrite history at each target. '
        'They quantify algebraic contribution magnitude, not emotional utility; source norms do not add to the joint norm.', '',
        '| Condition | Source A norm | Source T norm | Source V norm | Joint norm |',
        '|---|---:|---:|---:|---:|']
    for condition in conditions:
        stat = condition['statistics']
        values = [stat['source_states'][source]['mean'] for source in SOURCES]+[stat['joint_state_norm']['mean']]
        lines.append('| '+condition['group']+' '+condition['condition']+' | '+' | '.join(_format(value) for value in values)+' |')
    lines += ['', '## 4. Read strength and frozen decision contribution', '',
        'Every condition has two 4×3 tables. Rows are read destinations; columns are historical Value-write sources. '
        'Numbers are not pure modality semantics. Active Base includes zero-history first turns. '
        'A currently visible modality has an inactive Gap: N/A, not zero importance.', '',
        'Read norms and removal effects use exactly the same active-row cohort. '
        'ΔCE = CE_removed − CE_full: positive indicates beneficial presence under this intervention. '
        'ACC/UA/W-F1 deltas and corrections/harms appear in SLOTS.csv. '
        'A removal correction is full-wrong → removed-correct; a removal harm is full-correct → removed-wrong.', '']
    for condition in conditions:
        stat = condition['statistics']
        lines += [f"### {condition['group']} {condition['condition']}", '',
                  'Mean source-read L2 norm (active cohort N in parentheses).', '',
                  '| Read destination | History A | History T | History V |', '|---|---:|---:|---:|']
        for row in ROWS:
            values = []
            for source in SOURCES:
                item = stat['slots'][row][source]
                values.append(f"{_format(item['read_norm']['mean'])} ({item['N']})" if item['N'] else 'N/A (0)')
            lines.append('| '+row+' | '+' | '.join(values)+' |')
        lines += ['', 'Removal ΔCE (positive = beneficial presence in the fixed classifier).', '',
                  '| Read destination | History A | History T | History V |', '|---|---:|---:|---:|']
        for row in ROWS:
            lines.append('| '+row+' | '+' | '.join(_format(stat['slots'][row][source]['delta_CE'], 6) for source in SOURCES)+' |')
        lines += ['', 'Cancellation: 1 − norm(sum sources)/sum(source norms); only active rows with nonzero total source norm. '
                  'This is vector cancellation, not an emotion-conflict probability.', '',
                  '| Read destination | N | Mean cancellation |', '|---|---:|---:|']
        for row in ROWS:
            item = stat['cancellation'][row]
            lines.append(f"| {row} | {item['N']} | {_format(item['mean'])} |")
        lines += ['', 'Exact A/T/V grouped Shapley uses per-sample negative CE; source groups act jointly across Base and all active Gap rows.', '',
                  '| Source A | Source T | Source V | Full-minus-zero gain | Efficiency max abs error |',
                  '|---:|---:|---:|---:|---:|',
                  '| '+' | '.join(_format(stat['shapley']['mean'][source], 6) for source in SOURCES)
                  +' | '+_format(stat['shapley']['full_minus_zero_gain'], 6)
                  +' | '+_format(stat['shapley']['efficiency_max_abs_error'], 12)+' |', '']
    lines += ['## 5. Full outputs and interpretation limits', '',
        '- `CONDITIONS.csv`: all 15 full-model conditions, masks and resource checks.',
        '- `SLOTS.csv`: 12 independent removals per condition, active N, read norms, ΔCE/ACC/UA/W-F1, corrections and harms.',
        '- `SHAPLEY.csv`: all three source credits per condition, exact efficiency and distribution summaries.',
        '- `SLOTS_GROUPED.csv`: full slot statistics stratified by current availability, strictly-past ever-seen pattern, per-source ever-seen and coverage. Empty/inactive slots remain blank numeric N/A.',
        '- Per-condition JSON: all eight coalition scores, state norms, cancellation and strata; REPORT.json records aggregate metrics and source JSON hashes.', '',
        'These measurements separate storage magnitude, original-query reads, and fixed-classifier marginal responses, '
        'but do not identify pure modality information, causal semantic relevance, or an optimal new module. '
        'Nonlinear independent slot effects are not additive. The three grouped Shapley credits sum to the negative-CE gain '
        'of full history reads over all-zero reads in this fixed game. '
        'Both full-zero history and partial subtraction can be out-of-distribution (OOD). '
        'Large norms, cancellation, or a single-seed metric gain alone do not demonstrate a model defect or a generally useful intervention.', '',
        f"Frozen inference seconds: {_format(summary.get('seconds'), 3)}; peak allocated GPU MiB: {_format(summary.get('peak_memory_mib'), 3)}.", '']
    _write(Path(paths['result']), '\n'.join(lines))
    _write(Path(paths['json']), json.dumps(machine, indent=2, allow_nan=False)+'\n')
    for key, records in (('conditions', condition_records), ('slots', slots),
                         ('shapley', shapley_records), ('grouped_slots', grouped)):
        _write(Path(paths[key]), _csv_text(records))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_root, args.output_root), indent=2))


if __name__ == '__main__':
    main()
