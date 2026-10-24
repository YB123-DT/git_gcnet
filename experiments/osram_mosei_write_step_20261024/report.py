"""Validate all completed write-step arms against the same-source Flat0.6."""
import copy
import csv
import json
from pathlib import Path

from experiments.osram_mosei_capacity_cmerc_20261024.report import CONTROL, validate_pair
from experiments.osram_nested_cmerc_tuning_20261023.run import validate_summary
from experiments.osram_mosei_nested_local_20261024.report import validate_selected_metrics
from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
from experiments.osram_text_path_router_3seed_20261019.resource_gate import write


def normalize_config(candidate, baseline):
    for key in set(candidate) | set(baseline):
        if key != 'osram_write_step' and candidate.get(key) != baseline.get(key):
            raise ValueError('Only write-step may differ: ' + key)
    return dict(candidate, osram_write_step=baseline['osram_write_step'])


def write_report(output, state):
    if len(state['jobs']) != 4 or any(j['status'] != 'complete' for j in state['jobs']):
        raise ValueError('Require all four completed jobs; do not rank partial scores')
    baseline = json.loads((CONTROL/'SUMMARY.json').read_text())
    base_history = json.loads((CONTROL/'history.json').read_text())
    runs, rows = [], []
    arms = [(dict(step=.6), CONTROL, baseline)] + [
        (j, Path(j['output']), validate_summary(j)) for j in state['jobs']]
    for job, path, summary in arms:
        history = json.loads((path/'history.json').read_text())
        validate_selected(path, summary['best_by_rate'])
        validate_selected_metrics(summary, history)
        normalized = copy.deepcopy(summary)
        normalized['config'] = normalize_config(summary['config'], baseline['config'])
        validate_pair(normalized, history, baseline, base_history)
        if summary['config']['osram_write_step'] != job['step']:
            raise ValueError('Recorded arm does not match effective write strength')
        if summary['initial_backbone_sha256'] != baseline['initial_backbone_sha256']:
            raise ValueError('Initial parameters differ from paired Flat')
        aggregate = summary['aggregate']['router']
        runs.append(dict(step=job['step'], path=str(path), aggregate=aggregate,
                         seconds=summary['seconds'], peak_memory_mib=summary['peak_memory_mib']))
        rows.extend(dict(step=job['step'], rate=r, epoch=b['epoch'], **b['metrics']['router'])
                    for r, b in summary['best_by_rate'].items())
    runs.sort(key=lambda r:r['step'])
    output = Path(output)
    write(output/'RESULT.json', dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN', runs=runs))
    with (output/'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step','rate','epoch','ACC','WF1','MSE'])
        writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSEI Flat write-step screen', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             '| Write step | ACC | W-F1 | High W-F1 |', '|---|---:|---:|---:|']
    for run in runs:
        a = run['aggregate']
        lines.append(f"| {run['step']} | {a['mean']['ACC']:.3f} | {a['mean']['WF1']:.3f} | {a['high']['WF1']:.3f} |")
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
