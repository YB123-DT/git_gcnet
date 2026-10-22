"""Descriptive test-slot pooling; never treat rates/folds as independent trials."""
import csv
import json
from pathlib import Path

import numpy as np

from .run import DATASETS, RATES, query_sensitivity, write
from .stats import summarize


def concat(paths):
    pieces = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            for k in f.files:
                pieces.setdefault(k, []).append(f[k])
    return {k: np.concatenate(v) for k, v in pieces.items()}


def summarize_all(output):
    output = Path(output)
    report, focus_csv, heads_csv, role_csv = {}, [], [], []
    lines = ['# V3 source attention audit', '', 'INTERNAL TEST-ORACLE CHECKPOINT DIAGNOSTIC', '',
        'Original standard V3 seed66 only: width64,LargeFlat1600,residual mode.',
        'MOSI/MOSEI8rates; IEMOCAP4/6 five folds x8rates. No training or model updates.',
        'Every replay uses its own saved BEST checkpoint and exact original mask.',
        'MOSI/MOSEI use original s0; IEMOCAP original cmerc-repro. Saved predictions',
        'must match within1e-6 and predicted classes exactly, before any summary.', '',
        '## Text currently missing, all historical A/T/V seen', '',
        'Rows below pool TEST SLOTS across rates/folds descriptively. The same utterance',
        'can appear at multiple rates, from different BEST weights; these are NOT',
        'independent samples, seeds, or a single deployed model. Per-rate tables follow.', '',
        '| Dataset | Role | N | A | T | V | Normalized entropy(avg heads) | Mean head entropy |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        folder = output/dataset
        state = json.loads((folder/'STATUS.json').read_text()) if (folder/'STATUS.json').exists() else {'status': 'missing'}
        expected = 8 if dataset.startswith('CMU') else 40
        if state.get('status') != 'complete' or len(state.get('completed', [])) != expected:
            report[dataset] = dict(status=state.get('status'), error=state.get('error'))
            continue
        paths = [folder/row['arrays'] for row in state['completed']]
        arrays = concat(paths)
        pooled = summarize(arrays, dataset, 'pooled-test-slots', 'all_folds')
        pooled['query_sensitivity'] = query_sensitivity(arrays)
        pooled['max_prediction_error'] = max(row['max_error'] for row in state['completed'])
        report[dataset] = dict(status='complete', pooled=pooled, per_rate={})
        for row in pooled['focus']['rows']:
            if not row['N']:
                lines.append(f"| {dataset} | {row['role']} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            a, t, v = row['weight_mean']
            lines.append(f"| {dataset} | {row['role']} | {row['N']} | {a:.4f} | {t:.4f} | {v:.4f} | {row['normalized_entropy_head_average_mean']:.4f} | {row['normalized_entropy_per_head_mean']:.4f} |")
        del arrays
        for rate in RATES:
            arrays = concat([folder/row['arrays'] for row in state['completed'] if str(row['rate']) == rate])
            result = summarize(arrays, dataset, rate, 'folds_pooled')
            result['query_sensitivity'] = query_sensitivity(arrays)
            report[dataset]['per_rate'][rate] = result
            for row in result['focus']['rows']:
                base = dict(dataset=dataset, rate=rate, role=row['role'], N=row['N'])
                focus_csv.append(dict(base, **{s: row['weight_mean'][i] if row['N'] else None
                    for i, s in enumerate(('A', 'T', 'V'))},
                    entropy=row['normalized_entropy_head_average_mean'],
                    head_entropy=row['normalized_entropy_per_head_mean']))
                for h in range(4):
                    heads_csv.append(dict(base, head=h, **{s: row['per_head_weight_mean'][h][i]
                        if row['N'] else None for i, s in enumerate(('A', 'T', 'V'))}))
            for row in result['rows']:
                if row['group'] not in ('all', 'source_count=1', 'source_count=2', 'source_count=3', 'correct', 'error'):
                    continue
                role_csv.append(dict(dataset=dataset, rate=rate, role=row['role'], group=row['group'],
                    N=row['N'], **{s: row['weight_mean'][i] if row['N'] else None
                                   for i, s in enumerate(('A', 'T', 'V'))},
                    **{s+'_std': row['weight_std'][i] if row['N'] else None
                       for i, s in enumerate(('A', 'T', 'V'))},
                    entropy=row['normalized_entropy_head_average_mean'],
                    head_entropy=row['normalized_entropy_per_head_mean']))
            del arrays
    lines += ['', '## Paired role difference and Local-query sensitivity', '',
        '| Dataset | Base-v-GapT L1 | Same argmax fraction | Base query-swap L1 | GapT query-swap L1 |',
        '|---|---:|---:|---:|---:|']
    for dataset, data in report.items():
        if data['status'] != 'complete':
            continue
        p = data['pooled']; f = p['focus']; q = p['query_sensitivity']
        fmt = lambda x: 'N/A' if x is None else f'{x:.6f}'
        lines.append('| '+dataset+' | '+' | '.join(fmt(x) for x in
            (f['base_gap_T_weight_L1_mean'], f['base_gap_T_argmax_agreement'],
             q['Base']['head_average_weight_L1'], q['Gap-T']['head_average_weight_L1']))+' |')
    lines += ['', 'Query swap uses another conversation within the same test batch, matched on',
        'role,current availability and seen-source set, retaining identical original',
        'keys/values. Only extra attention is computed; predictions keep original outputs.',
        'This measures query dependence, NOT whether selected evidence improves the task.', '',
        '## Interpretation limits', '',
        '- These are Local-to-role output CrossAttention weights, after Layer2 source mixing.',
        '- High weights are NOT original-source causal usefulness, reliability, or sentiment shift.',
        '- No attention is defined for V3-disabled ATV/no-history/padding/inactive Gap rows.',
        '  Base remains available to original Flat even if V3 Base correction is disabled.',
        '- Source counts0/1 are separate; automatic weight1 is not learned confidence.',
        '- Normalized entropy1 means uniform across valid sources;0 means concentrated.',
        '- Per-sample std is descriptive; no bootstrap/significance or three-seed claim.',
        '- Correct/error labels only stratify offline; they never form Query or select weights.',
        '- All96 per-fold/per-rate reports and detailed arrays remain remote; no weights/data',
        '  or large activation caches enter Git.', '', '## Per-rate Text-missing focus', '',
        '| Dataset | Rate | Role | N | A | T | V | Entropy |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in focus_csv:
        fmt = lambda v: 'N/A' if v is None else f'{v:.4f}'
        lines.append(f"| {row['dataset']} | {row['rate']} | {row['role']} | {row['N']} | "+
                     ' | '.join(fmt(row[k]) for k in ('A', 'T', 'V', 'entropy'))+' |')
    for name, rows in (('FOCUS.csv', focus_csv), ('HEADS.csv', heads_csv), ('ROLES.csv', role_csv)):
        if rows:
            with (output/name).open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(output/'REPORT.json', report)
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return report
