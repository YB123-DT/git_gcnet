"""Summarize the finite matched one-stage experiment without test selection."""
import argparse
import csv
import json
from pathlib import Path


def write_report(output, summary):
    output = Path(output)
    lines = [f"# {summary['dataset']} / {summary['mode']} one-stage run", '',
             'INTERNAL SINGLE-SEED EXPLORATION', '',
             f"Source: `{summary['source_commit']}`. One validation-selected checkpoint; "
             f"selected epoch: {summary['selected_epoch']}.",
             'Fresh initialization, original GCNet random missing; no persistent augmentation.',
             'For Joint, full denotes its own jointly trained full candidate, NOT a separate Flat control.', '',
             '| Rate | Full ACC | Final ACC | Full UA | Final UA | Final W-F1 | Final CE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    rows = []
    for rate, entry in summary['evaluation'].items():
        full, final = entry['full'], entry['router']
        values = [full['ACC'], final['ACC'], full['UA'], final['UA'], final['WF1'], final['CE']]
        lines.append('| ' + rate + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        for tag in ('full', 'router'):
            rows.append(dict(rate=rate, readout=tag, **entry[tag]))
    lines += ['', 'Compare the separate matched Flat job before interpreting any benefit.',
              'No multi-seed, multi-fold or independent generalization claim.']
    (output / 'RESULT.md').write_text('\n'.join(lines).rstrip() + '\n')
    with (output / 'METRICS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root, output):
    import numpy as np

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collected, rows, checks = {}, [], {}
    lines = ['# One-stage historical Text-path probability Router', '',
             'INTERNAL SINGLE-SEED EXPLORATION', '',
             'MELD seed66; IEMOCAP-6 seed66/fold5. From-scratch matched Flat and Joint jobs.',
             'One checkpoint selected by validation mean eight-rate ACC (ties lower CE), then replayed',
             'on the same historical exact GCNet test masks. Historical test-guided research and mask',
             'provenance preclude an independent-test paper claim. Not per-rate Test-best scores.', '',
             '| Dataset | Model | Selected epoch | ACC | UA | W-F1 | High ACC | High UA | High W-F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in ('MELD', 'IEMOCAPSix'):
        pair = {}
        for mode in ('flat', 'joint'):
            path = root / dataset / mode / 'SUMMARY.json'
            summary = json.loads(path.read_text())
            if summary['status'] != 'complete' or summary['phase'] != 'train':
                raise ValueError(f'Incomplete training: {path}')
            pair[mode] = summary
            result = summary['aggregate']['router']
            values = [result[scope][metric] for scope in ('mean', 'high') for metric in ('ACC', 'UA', 'WF1')]
            lines.append(f"| {dataset} | {mode} | {summary['selected_epoch']} | " +
                         ' | '.join(f'{v:.4f}' for v in values) + ' |')
            for rate, entry in summary['evaluation'].items():
                rows.append(dict(dataset=dataset, mode=mode, rate=rate, **entry['router']))
        if pair['flat']['initial_backbone_sha256'] != pair['joint']['initial_backbone_sha256']:
            raise ValueError('Original backbone initializations differ: ' + dataset)
        if pair['flat']['validation_mask_hashes'] != pair['joint']['validation_mask_hashes']:
            raise ValueError('Validation masks differ: ' + dataset)
        histories = {mode: json.loads((root / dataset / mode / 'history.json').read_text())
                     for mode in ('flat', 'joint')}
        if len(histories['flat']) != len(histories['joint']):
            raise ValueError('Training budgets differ: ' + dataset)
        training_masks_equal = all(a['mask_trace_sha256'] == b['mask_trace_sha256']
                                   for a, b in zip(histories['flat'], histories['joint']))
        if not training_masks_equal:
            raise ValueError('Actual original training masks differ: ' + dataset)
        # Never claim paired masks merely because runs used the same seed.
        rate_checks = {}
        for rate in pair['flat']['evaluation']:
            name = 'random_' + rate.replace('.', 'p') + '.npz'
            with np.load(root / dataset / 'flat' / 'evaluation' / name, allow_pickle=False) as a, \
                    np.load(root / dataset / 'joint' / 'evaluation' / name, allow_pickle=False) as b:
                for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability'):
                    np.testing.assert_array_equal(a[key], b[key])
            rate_checks[rate] = True
        checks[dataset] = dict(test_cohort_and_masks_exact=rate_checks,
                               original_initialization_exact=True,
                               validation_masks_exact=True, training_mask_traces_exact=True)
        collected[dataset] = pair
    lines += ['', '## Interpretation safeguards', '',
              '- Joint vs separate Flat measures the full training-method change.',
              '- Within Joint, mixture vs its own full candidate measures the learned readout contribution.',
              '- Uniform Router initialization is not a Flat identity on eligible samples.',
              '- Eligibility is current Text missing AND strictly-past Text seen; no labels enter the Router.',
              '- Source terms are joint Value-slot contributions, not pure semantic modality attribution.',
              '- No automatic expansion to other seeds/folds or hyperparameter search.']
    (output / 'RESULT.md').write_text('\n'.join(lines).rstrip() + '\n')
    with (output / 'METRICS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'SUMMARY.json').write_text(json.dumps(dict(runs=collected, checks=checks),
                                                  indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.runs_root, args.output)
