"""Recover exact GCNet selection masks by replaying the historical NumPy stream.

Only mask draws are replayed. No model training, labels for selection, or
missing-algorithm change. Scores are accepted only after final mask parity and
selection ACC/W-F1 parity both pass.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np

from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
from gcnet_modality_jepa.protocol import SeedBundle


def recover_masks(config, train, test, selected_epochs, *, rates=tuple(i / 10 for i in range(8))):
    dataset = train.dataset
    lengths = [len(dataset.videoLabelsNew[vid]) for vid in dataset.vids]
    test_batches = list(test.batch_sampler)
    # This literal component is the historical trainer's actual initialization
    # seed, even for MELD fold1. Changing it to fold1 would change the masks.
    rng = np.random.RandomState(SeedBundle(config.seed).derive('missing_m3_model_init:fold:5'))

    def sample(batch, rate, collect=False):
        sizes = [lengths[i] for i in batch]
        length, batch_size = max(sizes), len(batch)
        host = random_mask(3, length * batch_size, rate, rng).reshape(batch_size, length, 3)
        guest = random_mask(3, length * batch_size, rate, rng).reshape(batch_size, length, 3)
        if collect:
            return np.concatenate([
                np.where(np.asarray(dataset.videoSpeakersNew[dataset.vids[index]])[:, None] == 0,
                         host[b, :sizes[b]], guest[b, :sizes[b]])
                for b, index in enumerate(batch)
            ])

    selected = {}
    for epoch in range(config.epochs):
        train.sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train.batch_sampler):
            rate = config.train_missing_rates[(epoch + batch_index) % len(config.train_missing_rates)]
            sample(batch, rate)
        for rate in rates:
            key = f'{rate:.1f}'
            collect = selected_epochs.get(key) == epoch + 1
            parts = [sample(batch, rate, collect) for batch in test_batches]
            if collect:
                selected[key] = np.concatenate(parts)
    if set(selected) != set(selected_epochs):
        raise ValueError('Selected epoch outside the historical training budget')
    final = {f'{rate:.1f}': np.concatenate([sample(batch, rate, True) for batch in test_batches])
             for rate in rates}
    return selected, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--meld-pickle', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal --gpu-uuid')
    if args.gpu_uuid == 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264':
        raise ValueError('Host GPU4 is forbidden')
    os.environ['GCNET_MELD_PICKLE'] = str(args.meld_pickle)
    import torch
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from gcnet_missing_m3.train_gcnet import TrainConfig, _move_batch, _collect_predictions
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model, _sha256, _write
    from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
        scatter_saved_availability, replay_view_from_availability,
    )
    torch.set_num_threads(2)
    device = torch.device('cuda')
    args.output.mkdir(parents=True)
    summary = dict(status='running', label='INTERNAL DIAGNOSTIC ONLY',
                   metric='UA at original per-rate ACC-selected checkpoint and exact selection masks',
                   inference_only=True, sources={})
    _write(args.output / 'SUMMARY.json', summary)
    try:
        for source_index, source in enumerate(args.sources):
            config = TrainConfig(**json.loads((source / 'config.json').read_text()))
            if not (config.dataset == 'MELD' and config.missing_mask_protocol == 'gcnet-random-mask-v1'
                    and config.train_rate_mode == 'cyclic' and config.training_objective == 'emotion-only'
                    and tuple(config.train_missing_rates) == tuple(i / 10 for i in range(8))
                    and config.osram_readout_fusion == 'flat' and not config.osram_bidirectional
                    and config.checkpoint_selection == 'test-oracle-per-rate'):
                raise ValueError('Unsupported historical draw sequence')
            checkpoints = {}
            hashes = {}
            for i in range(8):
                path = source / f'best_miss_0p{i}.pt'
                hashes[str(i / 10)] = _sha256(path)
                checkpoints[str(i / 10)] = torch.load(path, map_location='cpu', weights_only=False)
            selected_epochs = {rate: checkpoint['epoch'] for rate, checkpoint in checkpoints.items()}
            train, _, test, ad, td, vd = get_loaders('MELD-pickle', 'MELD-pickle', 'MELD-pickle', 1,
                                                  'MELD', config.batch_size, 0, config.seed,
                                                  validation_fraction=config.validation_fraction,
                                                  evaluation_protocol=config.evaluation_protocol)
            selected, final = recover_masks(config, train[0], test[0], selected_epochs)
            for i in range(8):
                with np.load(source / f'predictions_miss_0p{i}.npz') as saved:
                    np.testing.assert_array_equal(final[str(i / 10)], saved['availability'])
            source_output = args.output / f'source_{source_index}'
            source_output.mkdir()
            model = _build_model(replace(config, device='cuda'), (ad, td, vd), device)
            rates = {}
            for i in range(8):
                key = str(i / 10)
                checkpoint = checkpoints[key]
                model.load_state_dict(checkpoint['model'], strict=True)
                cursor = 0
                predictions, labels = [], []
                with torch.no_grad():
                    for raw in test[0]:
                        data = _move_batch(raw, device)
                        availability, cursor = scatter_saved_availability(selected[key], cursor, data[7])
                        view = replay_view_from_availability(data, availability, (ad, td, vd))
                        logits, _, _, _ = model([view['incomplete']], view['availability'], view['qmask'],
                                               view['umask'], view['lengths'], predict_missing=False)
                        pred, y, _ = _collect_predictions('MELD', logits, view['labels'], view['umask'],
                                                         config.mosi_task_mode)
                        predictions.append(pred)
                        labels.append(y)
                pred, y = np.concatenate(predictions), np.concatenate(labels)
                acc = float(accuracy_score(y, pred))
                wf1 = float(f1_score(y, pred, average='weighted', zero_division=0))
                if abs(acc - checkpoint['selection_accuracy']) > 1e-10:
                    raise ValueError(f'Best ACC not reproduced: {source}, {key}, {acc}')
                if abs(wf1 - checkpoint['selection_weighted_f1']) > 1e-10:
                    raise ValueError(f'Best W-F1 not reproduced: {source}, {key}, {wf1}')
                assert cursor == len(selected[key])
                np.savez_compressed(source_output / f'predictions_miss_0p{i}.npz',
                                    predictions=pred, labels=y, availability=selected[key])
                rates[key] = dict(epoch=checkpoint['epoch'], ACC=100 * acc, WF1=100 * wf1,
                                  UA=100 * float(recall_score(y, pred, labels=list(range(7)),
                                                            average='macro', zero_division=0)),
                                  checkpoint_sha256=hashes[key], best_ACC_WF1_exact=True)
            summary['sources'][str(source)] = dict(seed=config.seed, rates=rates,
                original_final_masks_exact=True, mean={m: sum(r[m] for r in rates.values()) / 8
                                                       for m in ('ACC', 'WF1', 'UA')},
                high_mean={m: sum(rates[str(i / 10)][m] for i in (5, 6, 7)) / 3
                           for m in ('ACC', 'WF1', 'UA')})
            _write(args.output / 'SUMMARY.json', summary)
            print(json.dumps(summary['sources'][str(source)]), flush=True)
            del model, checkpoints
        summary['status'] = 'complete'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        _write(args.output / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
