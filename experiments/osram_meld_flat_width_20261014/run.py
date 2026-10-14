"""Change only MELD Flat adapter hidden width under the retained GCNet mask."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from experiments.osram_meld_write_retention_20261014.reproduce import (
    HISTORICAL_COMMIT, now, normalized, sha, write,
)


def settings_for_width(source, width):
    if width not in (512, 1024, 2048):
        raise ValueError('width must be one of the three planned settings')
    required = dict(dataset='MELD', seed=66, epochs=50, batch_size=32,
                    learning_rate=.001, missing_mask_protocol='gcnet-random-mask-v1',
                    train_rate_mode='cyclic', backbone_type='osram',
                    training_objective='emotion-only', osram_bidirectional=False,
                    osram_readout_fusion='flat', osram_output_dim=1600,
                    osram_adapter_hidden_dim=0, osram_meaningful_block='none',
                    checkpoint_selection='test-oracle-per-rate')
    for key, expected in required.items():
        if source.get(key) != expected:
            raise ValueError(f'{key} must equal {expected!r}')
    if tuple(source['train_missing_rates']) != tuple(i / 10 for i in range(8)):
        raise ValueError('train_missing_rates must preserve the original eight rates')
    if source.get('osram_disable_local_skip', False):
        raise ValueError('Original Local skip must be active')
    result = dict(source)
    result['osram_adapter_hidden_dim'] = width
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, choices=(512, 1024, 2048), required=True)
    for name in ('original-code', 'source-config', 'baseline-reference', 'meld-pickle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.source_config, args.meld_pickle, args.baseline_reference / 'metrics.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = json.loads(args.source_config.read_text())
    settings = settings_for_width(source, args.width)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the requested UUID')
    gpu_rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits',
    ], text=True)
    matched = [line.split(',') for line in gpu_rows.splitlines()
               if line.split(',')[-1].strip() == args.gpu_uuid]
    if len(matched) != 1 or matched[0][0].strip() == '4':
        raise ValueError('Unknown UUID or forbidden host GPU4')
    root = Path(__file__).resolve().parents[2]
    for name in ('train_gcnet.py', 'model.py', 'osram.py'):
        relative = 'gcnet_missing_m3/' + name
        original = subprocess.check_output(['git', 'show', f'{HISTORICAL_COMMIT}:{relative}'], cwd=root)
        if hashlib.sha256(original).hexdigest() != sha(args.original_code / relative):
            raise ValueError(f'Historical code mismatch: {relative}')
    os.environ.update(GCNET_DATASET_ROOT=str(args.original_code / 'dataset'),
                      GCNET_MELD_PICKLE=str(args.meld_pickle),
                      GCNET_CACHE_ROOT=str(args.output.parent / 'cache'))
    sys.path.insert(0, str(args.original_code.resolve()))
    import gcnet_missing_m3.train_gcnet as training
    if not Path(training.__file__).resolve().is_relative_to(args.original_code.resolve()):
        raise RuntimeError('Training imported from the wrong code snapshot')
    config = training.TrainConfig(**settings)
    effective = normalized(asdict(config))
    changed = {key for key in source if effective[key] != normalized(source[key])}
    if changed != {'osram_adapter_hidden_dim'}:
        raise ValueError(f'Unexpected effective-config changes: {changed}')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    provenance = dict(status='running', started_utc=now(), pid=os.getpid(),
                      host=socket.gethostname(), gpu_uuid=args.gpu_uuid,
                      host_gpu_index=int(matched[0][0]), seed=66,
                      training_commit=HISTORICAL_COMMIT,
                      runner_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      baseline_reference=str(args.baseline_reference),
                      source_config_sha256=sha(args.source_config),
                      meld_pickle_sha256=sha(args.meld_pickle), config=effective,
                      changed_config_fields=sorted(changed),
                      protocol='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle ACC selection',
                      checkpoint_policy='retain all eight best checkpoints',
                      architecture=f'4352 -> {args.width} -> 1600; original Local skip and classifier',
                      threads=1)
    write(args.output / 'PROVENANCE.json', provenance)
    try:
        metrics = training.run_experiment(config, 'MELD-pickle', 'MELD-pickle', 'MELD-pickle', args.output)
        reference = json.loads((args.baseline_reference / 'metrics.json').read_text())
        scores = {}
        for field in ('selected_accuracy_by_rate', 'selected_weighted_f1_by_rate'):
            values = metrics[field]
            scores[field] = dict(by_rate=values, mean=sum(values.values()) / 8,
                                 high_mean=sum(values[str(i / 10)] for i in (5, 6, 7)) / 3,
                                 baseline_mean=sum(reference[field].values()) / 8)
        for index in range(8):
            if not (args.output / f'best_miss_0p{index}.pt').is_file():
                raise FileNotFoundError('Missing retained per-rate best checkpoint')
        first = torch.load(args.output / 'best_miss_0p0.pt', map_location='cpu', weights_only=False)
        adapter_count = sum(value.numel() for name, value in first['model'].items()
                            if name.startswith('osram.emotion_adapter.'))
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        saved_scores = {'width': {}, 'baseline': {}}
        for index in range(8):
            filename = f'predictions_miss_0p{index}.npz'
            with np.load(args.output / filename) as current, np.load(args.baseline_reference / filename) as base:
                np.testing.assert_array_equal(current['labels'], base['labels'])
                np.testing.assert_array_equal(current['availability'], base['availability'])
                for name, values in (('width', current), ('baseline', base)):
                    labels, predictions = values['labels'], values['predictions']
                    saved_scores[name][str(index / 10)] = {
                        'ACC': float(accuracy_score(labels, predictions)),
                        'WF1': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
                        'UA': float(recall_score(labels, predictions, labels=list(range(7)),
                                                average='macro', zero_division=0)),
                    }
        write(args.output / 'WIDTH_SUMMARY.json', dict(label='INTERNAL DIAGNOSTIC ONLY',
                                                     scores=scores, adapter_parameters=adapter_count,
                                                     saved_mask_scores=saved_scores,
                                                     final_labels_and_masks_exact=True,
                                                     UA_note='UA uses final saved-mask predictions; not unavailable winning-epoch masks'))
        provenance.update(status='complete', finished_utc=now(), adapter_parameters=adapter_count)
    except BaseException as error:
        provenance.update(status='failed', finished_utc=now(), error=repr(error))
        raise
    finally:
        write(args.output / 'PROVENANCE.json', provenance)


if __name__ == '__main__':
    main()
