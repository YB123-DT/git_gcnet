"""CPU-only exact Test-best recovery; never train, infer or change weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def verify_identity(selected, canonical):
    def keys(value):
        result = [(str(c), int(t)) for c, t in zip(value['conversation_ids'], value['utterance_indices'])]
        if len(result) != len(set(result)):
            raise ValueError('Duplicate conversation/utterance identity')
        return result
    actual, expected = keys(selected), keys(canonical)
    if set(actual) != set(expected):
        raise ValueError('Different test utterances, not merely a different loader order')
    indices = {key: i for i, key in enumerate(expected)}
    order = [indices[key] for key in actual]
    for field in ('labels', 'speakers'):
        np.testing.assert_array_equal(selected[field], canonical[field][order])


def recover_run(path, canonical, output):
    import torch
    from .worker import aggregate, class_metrics
    from gcnet_missing_m3.training_resume import _sha
    raw = json.loads((path / 'SUMMARY.json').read_text())
    history = json.loads((path / 'history.json').read_text())
    if len(history) != 50:
        if raw['status'] == 'failed':
            raise ValueError(f'Training did not reach 50 epochs: {path}')
        return None
    if raw['status'] == 'running':
        return None
    rates = {}
    for i in range(8):
        checkpoint_path = path / f'best_miss_0p{i}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        ref = checkpoint['selection_artifact']
        selected_path = path / ref['path']
        if _sha(selected_path) != ref['sha256']:
            raise ValueError('Selected-epoch artifact hash mismatch')
        with np.load(selected_path) as selected:
            verify_identity(selected, canonical)
            availability = selected['availability']
            if not np.isin(availability, [0, 1]).all() or (availability.sum(-1) < 1).any():
                raise ValueError('Invalid selected-epoch availability')
            if (i == 0 and not (availability == 1).all()) or (i == 7 and not (availability.sum(-1) == 1).all()):
                raise ValueError('GCNet endpoint missing-mask contract violated')
            metrics = class_metrics(selected['labels'], selected['logits'])
        if abs(metrics['ACC'] / 100 - checkpoint['selection_accuracy']) > 1e-8:
            raise ValueError('Selected ACC differs from the original winning checkpoint')
        rates[str(i/10)] = dict(metrics, epoch=checkpoint['epoch'],
                               checkpoint_sha256=_sha(checkpoint_path),
                               selected_file_sha256=ref['sha256'],
                               mask_sha256=ref['availability_sha256'])
    result = dict(raw, status='complete', error=None, testbest=aggregate(rates),
        completion_scope='Exact Test-best recovery ONLY; no new training or inference',
        training_completed_epochs=50,
        original_worker_summary=str(path / 'SUMMARY.json'),
        original_worker_status=raw['status'], original_worker_error=raw.get('error'),
        paired={'status': 'not_completed', 'reason': 'Cross-seed positional replay order mismatch; not required for requested Test-best replication'},
        recovery_code_sha=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        identity_check='Same complete utterance key set, label and speaker per key; order may differ')
    from .scheduler import write_json
    write_json(output / path.name / 'SUMMARY.json', result)
    return result


def main():
    from .replicate import make_jobs, summarize
    from .scheduler import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'MANIFEST.json').read_text())
    output = args.root / 'testbest_recovery_v1'
    output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(manifest['configurations'], args.root)
    recovered = {}
    deadline = time.monotonic() + 7200
    while True:
        records = []
        for job in jobs:
            identifier = job['id']
            if identifier not in recovered:
                config_id = identifier.rsplit('_seed', 1)[0]
                source = next(item for item in manifest['configurations'] if item['id'] == config_id)
                source_dir = Path(source['source_summary']).parent
                with np.load(source_dir / 'paired_miss_0p0.npz') as canonical:
                    result = recover_run(Path(job['output']), canonical, output / 'runs')
                if result is not None:
                    recovered[identifier] = result
                    print(json.dumps({'job': identifier, 'testbest': result['testbest']['mean']}), flush=True)
            records.append(dict(id=identifier, status='complete' if identifier in recovered else 'pending',
                                summary=recovered.get(identifier)))
        report = summarize(manifest['configurations'], records, output)
        write_json(output / 'RECOVERY.json', dict(training_source='ab3d7e6',
            status=report['status'], recovered_jobs=list(recovered),
            original_outputs_unchanged=True, new_training=False, new_inference=False,
            root_cause='All protocol loaders, including test, use a seed-derived EpochSeededSubsetSampler; positional seed66 replay is not cross-seed safe'))
        if len(recovered) == len(jobs) or not args.wait:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Remaining training did not finish within the bounded recovery wait')
        time.sleep(15)


if __name__ == '__main__':
    main()
