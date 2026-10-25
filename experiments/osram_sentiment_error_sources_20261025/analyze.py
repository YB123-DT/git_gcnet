"""Saved-prediction error counts by rate, observed set and strict-past exposure."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


RATES = tuple(f'{i / 10:.1f}' for i in range(8))
MODALITIES = 'ATV'
PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1),
            'AT': (1, 1, 0), 'AV': (1, 0, 1), 'TV': (0, 1, 1), 'ATV': (1, 1, 1)}
SOURCES = {
    'MOSI': Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2'),
    'MOSEI': Path('/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_history(ids, indices, availability):
    """Include all earlier valid turns, even neutral ones; never current/future."""
    seen = np.zeros_like(availability, dtype=bool)
    for conversation in np.unique(ids):
        positions = np.flatnonzero(ids == conversation)
        ordered = positions[np.argsort(indices[positions], kind='stable')]
        if not np.array_equal(indices[ordered], np.arange(len(ordered))):
            raise ValueError('History must contain consecutive valid indices starting at0')
        state = np.zeros(3, dtype=bool)
        for position in ordered:
            seen[position] = state
            state |= availability[position].astype(bool)
    return seen


def tally(y, prediction, mask, indices):
    valid = mask & (y != 0)
    wrong = (prediction > 0) != (y > 0)
    n, errors = int(valid.sum()), int((valid & wrong).sum())
    return dict(N=n, errors=errors, correct=n - errors,
                error_pct=100 * errors / n if n else None,
                positive_N=int((valid & (y > 0)).sum()),
                negative_N=int((valid & (y < 0)).sum()),
                positive_to_negative=int((valid & (y > 0) & wrong).sum()),
                negative_to_positive=int((valid & (y < 0) & wrong).sum()),
                first_N=int((valid & (indices == 0)).sum()),
                first_errors=int((valid & (indices == 0) & wrong).sum()))


def split_rate(data):
    y, p, av, indices, ids = (data[k] for k in (
        'labels', 'prediction', 'availability', 'utterance_indices', 'conversation_ids'))
    seen = strict_history(ids, indices, av)
    total = tally(y, p, np.ones(len(y), dtype=bool), indices)
    patterns, joint, marginal = {}, [], {}
    for name, pattern in PATTERNS.items():
        current = (av == pattern).all(-1)
        patterns[name] = tally(y, p, current, indices)
        missing = [i for i, present in enumerate(pattern) if not present]
        for values in itertools.product((False, True), repeat=len(missing)):
            group = current.copy()
            descriptions = []
            for m, value in zip(missing, values):
                group &= seen[:, m] == value
                descriptions.append(MODALITIES[m] + ':' + ('seen' if value else 'never'))
            joint.append(dict(observed=name, missing=''.join(MODALITIES[m] for m in missing) or 'none',
                              history=','.join(descriptions) or 'N/A', **tally(y, p, group, indices)))
        selected = [row for row in joint if row['observed'] == name]
        for key in ('N', 'errors'):
            if sum(row[key] for row in selected) != patterns[name][key]:
                raise ValueError('Joint history states do not partition the current pattern')
    for m, modality in enumerate(MODALITIES):
        current_missing = av[:, m] == 0
        marginal[modality] = {name: tally(y, p, current_missing & mask, indices)
            for name, mask in (('never', ~seen[:, m]), ('seen', seen[:, m]),
                               ('never_with_history', ~seen[:, m] & (indices > 0)))}
        missing_total = tally(y, p, current_missing, indices)
        for key in ('N', 'errors'):
            if marginal[modality]['never'][key] + marginal[modality]['seen'][key] != missing_total[key]:
                raise ValueError('Seen/never counts do not partition missing targets')
    for key in ('N', 'errors'):
        if sum(row[key] for row in patterns.values()) != total[key]:
            raise ValueError('Current patterns do not partition the total')
    return dict(total=total, patterns=patterns, joint=joint, missing_modalities=marginal)


def cell(row):
    return '--' if not row['N'] else f"{row['errors']}/{row['N']} ({row['error_pct']:.1f}%)"


def write_outputs(output, summary):
    lines = ['# MOSI/MOSEI Flat error sources', '', 'INTERNAL DIAGNOSTIC ONLY',
        'Original LargeFlat seed66, gcnet-random-mask-v1, cmerc-repro for both datasets.',
        'Saved predictions only: no new model inference/training or changed masks.',
        'Error means wrong binary polarity, excluding gold0; prediction0 is negative.',
        'Seen means observed strictly earlier in the same conversation, not necessarily retained/retrieved.',
        'Neutral-label turns remain in the availability history. First turns count as never-seen.',
        'Cells are errors / nonneutral sample count (error%). -- means no samples, not zero error.',
        'Tables are per rate. Do not add overlapping modality marginals or treat repeated rates as independent.', '']
    pattern_csv, joint_csv, marginal_csv = [], [], []
    for dataset, item in summary['datasets'].items():
        lines += [f'## {dataset}', '',
            f"Original saved mean8 ACC={item['aggregate']['mean']['ACC']:.6f}, W-F1={item['aggregate']['mean']['WF1']:.6f}.",
            f"Source: `{item['source']}`; LR={item['learning_rate']}, epochs={item['epochs']}.", '',
            '### Current observed-set split', '',
            '| Rate | ALL | A (missing TV) | T (missing AV) | V (missing AT) | AT (missing V) | AV (missing T) | TV (missing A) | ATV |',
            '|---|---|---|---|---|---|---|---|---|']
        for rate in RATES:
            row = item['rates'][rate]
            lines.append('| ' + rate + ' | ' + cell(row['total']) + ' | ' + ' | '.join(
                cell(row['patterns'][name]) for name in PATTERNS) + ' |')
            pattern_csv.append(dict(dataset=dataset, rate=rate, observed='ALL', missing='N/A', **row['total']))
            for name, values in row['patterns'].items():
                pattern_csv.append(dict(dataset=dataset, rate=rate, observed=name,
                    missing=''.join(m for m in MODALITIES if m not in name) or 'none', **values))
            joint_csv.extend(dict(dataset=dataset, rate=rate, **v) for v in row['joint'])
            for modality, values in row['missing_modalities'].items():
                marginal_csv.extend(dict(dataset=dataset, rate=rate, missing_modality=modality,
                                         history=name, **v) for name, v in values.items())
        lines += ['', '### Per-missing-modality history split (overlapping marginals)', '',
            'For example, an A-only utterance enters both missing-T and missing-V columns. Do not sum them.', '',
            '| Rate | Missing A, never | Missing A, seen | Missing T, never | Missing T, seen | Missing V, never | Missing V, seen |',
            '|---|---|---|---|---|---|---|']
        for rate in RATES:
            rows = item['rates'][rate]['missing_modalities']
            lines.append('| ' + rate + ' | ' + ' | '.join(cell(rows[m][state])
                         for m in MODALITIES for state in ('never', 'seen')) + ' |')
        lines += ['', '### Exact joint history states inside each current pattern', '',
            'These rows are mutually exclusive within a current pattern; counts reconcile to the table above.',
            'For current A, T:never,V:seen means missing T has never appeared, while missing V has appeared.',
            'ATV has no missing target and its history-state entry is N/A.', '']
        for rate in RATES:
            lines += [f'#### Rate {rate}', '', '| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |',
                      '|---|---|---|---|---|']
            for row in item['rates'][rate]['joint']:
                if row['N']:
                    lines.append(f"|{row['observed']}|{row['missing']}|{row['history']}|{cell(row)}|{row['first_errors']}/{row['first_N']}|")
    lines += ['', '## Limits', '',
        'Descriptive counts from one seed and per-rate Test-best checkpoints, not independent generalization.',
        'Never-seen contains first turns; CSV additionally reports never_with_history and first-turn counts.',
        'Seen/never groups may differ in conversation position, labels and content; these are not causal effects.',
        'Availability indicates a real observation allowed by the saved mask, not a guarantee of useful content.',
        'No inference that Memory is empty, retained the modality, or should be disabled follows from this split.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    for name, rows in (('CURRENT_PATTERNS.csv', pattern_csv), ('JOINT_HISTORY.csv', joint_csv),
                       ('MISSING_MODALITY_HISTORY.csv', marginal_csv)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite an earlier analysis')
    from sklearn.metrics import accuracy_score, f1_score
    summary = dict(status='running', label='INTERNAL DIAGNOSTIC ONLY', source_commit=args.source_commit,
                   analysis_sha256=sha(Path(__file__)), model_forwards=0, training_updates=0,
                   datasets={})
    for dataset, root in SOURCES.items():
        source = json.loads((root / 'SUMMARY.json').read_text())
        cfg = source.get('config', source.get('identity', {}).get('config', {}))
        if source['status'] != 'complete' or cfg['seed'] != 66 or cfg['missing_mask_protocol'] != 'gcnet-random-mask-v1':
            raise ValueError('Wrong source protocol')
        if cfg.get('osram_meaningful_block', 'none') != 'none' or cfg.get('osram_adapter_hidden_dim', 0) not in (0, 1600):
            raise ValueError('Require original LargeFlat, not a module/width variant')
        if source['environment']['python'] != '/data2/yb/conda_envs/cmerc-repro/bin/python':
            raise ValueError('Both source models must use the accepted cmerc environment')
        item = dict(source=str(root), summary_sha256=sha(root / 'SUMMARY.json'),
                    environment=source['environment'], learning_rate=cfg['learning_rate'],
                    epochs=cfg['epochs'], source_commit=source.get('source_commit'),
                    aggregate=source['aggregate']['router'], rates={}, predictions={})
        for rate in RATES:
            selected = source['best_by_rate'][rate]
            path = root / selected['predictions']
            if sha(path) != selected['prediction_sha256']:
                raise ValueError('Protected prediction hash changed')
            with np.load(path, allow_pickle=False) as saved:
                data = {k: saved[k].copy() for k in ('labels', 'prediction', 'availability',
                        'conversation_ids', 'utterance_indices')}
            data['labels'], data['prediction'] = data['labels'].reshape(-1), data['prediction'].reshape(-1)
            av = data['availability']
            if av.shape != (len(data['labels']), 3) or not np.isin(av, [0, 1]).all() or not (av.sum(-1) > 0).all():
                raise ValueError('Invalid observed sets')
            if not np.isfinite(data['labels']).all() or not np.isfinite(data['prediction']).all():
                raise ValueError('Nonfinite labels/predictions')
            y, p = data['labels'], data['prediction']
            valid = y != 0
            measured = dict(ACC=100 * accuracy_score(y[valid] > 0, p[valid] > 0),
                            WF1=100 * f1_score(y[valid] > 0, p[valid] > 0, average='weighted'))
            for metric, value in measured.items():
                np.testing.assert_allclose(value, selected['metrics']['router'][metric], atol=1e-9, rtol=0)
            item['rates'][rate] = dict(epoch=selected['epoch'], **split_rate(data))
            item['predictions'][rate] = dict(path=str(path), sha256=sha(path),
                checkpoint_sha256=selected['checkpoint_sha256'], mask_sha256=selected['metrics']['mask_sha256'])
        summary['datasets'][dataset] = item
    summary['status'] = 'complete'
    args.output.mkdir(parents=True)
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_outputs(args.output, summary)
    print(json.dumps({dataset: {rate: dict(total=row['total'], patterns=row['patterns'],
            history=row['missing_modalities']) for rate, row in item['rates'].items()}
            for dataset, item in summary['datasets'].items()}, indent=2))


if __name__ == '__main__':
    main()
