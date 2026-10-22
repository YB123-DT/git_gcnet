"""Completed-only direct screen with first100-epoch historical comparisons."""
import csv
import hashlib
import json
from pathlib import Path

from .worker import MODE
from experiments.osram_mosi_v3_capacity_20261022.worker import REFERENCE, RATES

CAPACITY = Path('/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1/runs')
REFERENCES = {'original_large_flat': REFERENCE,
              'flat512': CAPACITY/'f512_plain', 'flat512_v3_residual': CAPACITY/MODE}


def select_history(history, horizon=100):
    if len(history) < horizon or [r['epoch'] for r in history[:horizon]] != list(range(1, horizon+1)):
        raise ValueError('Contiguous first100-epoch history required')
    result = {}
    for row in history[:horizon]:
        for rate in RATES:
            metric = row['test'][rate]['router']
            if rate not in result or metric['WF1'] > result[rate]['metrics']['WF1']:
                result[rate] = dict(epoch=row['epoch'], metrics=metric)
    return result


def summarize(output):
    output = Path(output)
    p = output/'runs'/MODE/'SUMMARY.json'
    s = json.loads(p.read_text()) if p.exists() else {'status': 'not_started'}
    state = {k: s.get(k) for k in ('status', 'epoch', 'error', 'seconds', 'parameters')}
    records, selected, common, provenance = {}, {}, None, {}
    for name, folder in REFERENCES.items():
        record = json.loads((folder/'SUMMARY.json').read_text())
        if record.get('status') != 'complete' or record.get('epoch') != 150:
            raise ValueError('Complete protected reference required: '+name)
        history_path = folder/'history.json'
        selected[name] = select_history(json.loads(history_path.read_text()))
        records[name] = record
        provenance[name] = dict(path=str(folder), history_sha256=hashlib.sha256(
            history_path.read_bytes()).hexdigest(), selected_budget=100,
            archived_training_budget=150, new_inference=False)
    if s.get('status') == 'complete':
        if s.get('epoch') != 100 or set(s.get('evaluation', {})) != set(RATES):
            raise ValueError('Direct result is incomplete')
        cfg = s['identity']['config']
        if (cfg['osram_source_role_v3_mode'] != 'direct' or cfg['epochs'] != 100
                or cfg['seed'] != 66 or s['parameters']['total'] != 7864639):
            raise ValueError('Direct mode, budget or parameter count changed')
        selected['flat512_v3_direct'] = {r: dict(epoch=s['best_by_rate'][r]['epoch'],
            metrics=s['evaluation'][r]['router']) for r in RATES}
        if selected['flat512_v3_direct'] != select_history(
                json.loads((p.parent/'history.json').read_text())):
            raise ValueError('Stored predictions and history selection differ')
        records['flat512_v3_direct'] = s
    for name, record in records.items():
        ident = record['identity']
        masks = ident['mask_hashes'].get('test', ident['mask_hashes'])
        comparable = {key: ident[key] for key in ('split', 'seed_protocol', 'data_files')}
        comparable['mask_hashes'] = masks
        if common is not None and common != comparable:
            raise ValueError('Paired data or exact test masks differ: '+name)
        common = comparable
    rows, aggregate = [], []
    for name, values in selected.items():
        for rate in RATES:
            rows.append(dict(model=name, rate=rate, epoch=values[rate]['epoch'],
                             **{k: values[rate]['metrics'][k] for k in ('ACC', 'WF1', 'MSE')}))
        aggregate.append(dict(model=name, **{scope: {key: sum(values[r]['metrics'][key]
            for r in rates)/len(rates) for key in ('ACC', 'WF1', 'MSE')}
            for scope, rates in (('mean', RATES), ('high', RATES[5:]))}))
    report = dict(label='INTERNAL TEST-ORACLE SCREEN', state=state, aggregate=aggregate,
                  per_rate=rows, historical_provenance=provenance, comparison_budget=100,
                  initialization_confounded=True, exact_masks_and_data_matched=True)
    (output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'PER_RATE.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=('model', 'rate', 'epoch', 'ACC', 'WF1', 'MSE'))
        w.writeheader(); w.writerows(rows)
    lines = ['# MOSI direct V3 read inputs', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Seed66,100epochs,3e-4,original GCNet masks. Only completed runs enter scores.',
        'Historical150-epoch controls are reselected from their FIRST100 epoch logs.', '',
        '| Model | ACC | W-F1 | High ACC | High W-F1 |', '|---|---:|---:|---:|---:|']
    for row in aggregate:
        a, h = row['mean'], row['high']
        lines.append(f"| {row['model']} | {a['ACC']:.3f} | {a['WF1']:.3f} | {h['ACC']:.3f} | {h['WF1']:.3f} |")
    lines += ['', 'State: '+str(state), '', 'Direct uses random decoders; residual uses zero decoders.',
        'This is not an addition-only causal comparison. Internal V3 residuals remain.',
        'No CHM/LocalSkip changes. No additional reference training/inference.', '',
        '| Model | Rate | Epoch | ACC | W-F1 | MSE |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['rate']} | {row['epoch']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return report
