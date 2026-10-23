"""Export small verified result records without copying raw tensors to Git."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, metrics, covariance_means = {}, [], {}
    for name, dataset, expected_n in [('mosi', 'CMUMOSI', 686), ('mosei', 'CMUMOSEI', 4659)]:
        folder = args.execution / name
        state_path = folder / 'STATUS.json'
        state = json.loads(state_path.read_text())
        if (state['status'] != 'complete' or state['smoke'] or
                state['new_training_steps'] != 0 or not state['weights_unchanged_exact']):
            raise ValueError('Not a completed, unchanged frozen diagnostic: ' + name)
        if set(state['rates']) != {'0.3', '0.7'}:
            raise ValueError('Wrong diagnostic rate coverage')
        for r in state['rates'].values():
            if (not r['full_test_coverage'] or r['utterances'] != expected_n or
                    r['max_clean_prediction_parity_error'] > 1e-6):
                raise ValueError('Incomplete or nonmatching clean replay: ' + name)
        identity = state['identity']
        compact = {k: v for k, v in state.items() if k != 'identity'}
        compact['identity'] = {k: v for k, v in identity.items() if k != 'data_files'}
        compact['complete_remote_status_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        compact['data_files_count'] = len(identity['data_files'])
        records[dataset] = compact
        covariance_means[dataset] = {}
        for rate in ('0.3', '0.7'):
            count, matrix = 0, np.zeros((4, 4))
            for path in sorted((folder / 'covariance').glob(f'rate_{rate}_batch_*.npz')):
                with np.load(path, allow_pickle=False) as archive:
                    g = archive['G']
                if g.shape[1:] != (8, 4, 4) or not np.isfinite(g).all():
                    raise ValueError('Bad covariance artifact')
                count += len(g)
                matrix += g.sum(axis=(0, 1))
            if count != expected_n:
                raise ValueError('Incomplete full covariance artifact coverage')
            covariance_means[dataset][rate] = dict(N=count,
                roles=['Base', 'Gap-A', 'Gap-T', 'Gap-V'],
                mean_unit_covariance_including_inactive_zeros=(matrix / (count * 8)).tolist())
        result = json.loads((folder / 'AGGREGATE.json').read_text())
        for r in result['groups']:
            if r['group'] != 'overall':
                continue
            metrics.append(dict(dataset=dataset, rate=r['rate'], sigma=r['sigma'], N=r['N'],
                clean_ACC=r['clean']['ACC'], noisy_ACC=r['noisy']['ACC'],
                clean_W_F1=r['clean']['W_F1'], noisy_W_F1=r['noisy']['W_F1'],
                clean_MSE=r['clean']['MSE'], delta_MSE=r['delta_MSE_mean'],
                mean_abs_prediction_shift=r['prediction_abs_shift_mean'],
                flip_percent=r['flip_rate_percent'], corrections_mean=r['corrections_mean'],
                harms_mean=r['harms_mean'], covariance_relative_error=state['rates'][r['rate']]['covariance_relative_frobenius_error']))
    if len({r['source_commit'] for r in records.values()}) != 1:
        raise ValueError('Do not combine source versions')
    if args.output.exists():
        raise FileExistsError('Use a fresh summary destination')
    args.output.mkdir(parents=True)
    for name in ('mosi', 'mosei'):
        target = args.output / name
        target.mkdir()
        for filename in ('AGGREGATE.json', 'RESULT.md'):
            shutil.copy2(args.execution / name / filename, target / filename)
    (args.output / 'PROVENANCE.json').write_text(json.dumps(records, indent=2) + '\n')
    (args.output / 'COVARIANCE_MEANS.json').write_text(json.dumps(covariance_means, indent=2) + '\n')
    with (args.output / 'OVERALL.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
