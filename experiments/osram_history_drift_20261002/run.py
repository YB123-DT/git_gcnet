"""Frozen original-Flat, same-current/different-history diagnostic (no fitting)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_readout_paths_20260930.run import (
    DEFAULT_DATA, DEFAULT_REFERENCE, RATES, read, write, sha, state_hash,
    split_ids, verify_gpu, configuration_dict,
)

LABEL = 'Frozen history intervention; inherited per-rate Test-oracle checkpoints; no training or selection'
MODES = ('A', 'T', 'V', 'mixed')


def scalar(value):
    return value.item() if hasattr(value, 'item') else value


def evaluate(model, loader, schedule, dimensions, device, args, split, rate, output):
    import torch
    from sklearn.metrics import f1_score
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view
    from experiments.osram_history_drift_20261002.diagnostic import (
        nested_view, capture, anchor_metrics,
    )

    tag = f'{split}_miss_{rate:.1f}'.replace('.', 'p')
    anchors = []
    predictions, labels, seen = [], [], set()
    counts = {mode: dict(anchors=0, observed_bits=0, deleted_bits=0) for mode in MODES}
    loader.sampler.set_epoch(0)
    batch_count = 0
    max_local = 0.
    max_local_relative = 0.
    artifacts = {}
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            view = _prepare_view(_move_batch(raw, device), schedule, 0, dimensions)
            availability = view['availability']
            valid = view['umask'].T.bool()
            call = lambda v: model([v['incomplete']], v['availability'], v['qmask'],
                                   v['umask'], v['lengths'], predict_missing=False)
            baseline = capture(model, lambda: call(view))
            if batch_index == 0:
                repeat = capture(model, lambda: call(view))
                for key in baseline:
                    if not torch.equal(baseline[key], repeat[key]):
                        raise AssertionError(f'identity repeat mismatch: {key}')
            if not torch.isfinite(baseline['prediction'][valid]).all():
                raise AssertionError('nonfinite baseline prediction')
            predictions.extend(baseline['prediction'][valid].reshape(-1).cpu().tolist())
            labels.extend(view['labels'].T[valid].cpu().tolist())
            seen.update(str(x) for x in view['conversation_ids'])
            masks = {'availability1': availability.cpu().numpy(),
                     'umask': view['umask'].cpu().numpy(),
                     'conversation_ids': np.asarray(view['conversation_ids'], dtype=str)}
            baseline_rows = []
            for t, b in valid.nonzero(as_tuple=False).cpu().tolist():
                baseline_rows.append(dict(conversation_id=str(view['conversation_ids'][b]),
                    utterance_index=t, label=float(view['labels'][b, t]),
                    pred1=float(baseline['prediction'][t, b].reshape(-1)[0])))
            baseline_file = output / f'{tag}_batch_{batch_index}_baseline.json'
            write(baseline_file, baseline_rows)
            artifacts[baseline_file.name] = sha(baseline_file)
            for mode in MODES:
                # Common random draws pair the A/T/V interventions fairly.
                seed = 66000000 + int(round(rate * 10)) * 100000 + (split == 'test') * 10000 + batch_index * 100
                availability2, meta = nested_view(availability, view['umask'], mode, seed, args.drop_prob)
                expanded = torch.repeat_interleave(availability2,
                    torch.tensor(dimensions, device=device), dim=-1)
                second = dict(view, availability=availability2,
                              incomplete=torch.where(expanded.bool(), view['incomplete'], 0.))
                anchor = meta['contrast_mask']
                if (availability2 > availability).any() or (availability2[valid].sum(-1) < 1).any():
                    raise AssertionError('nested nonempty availability invariant failed')
                if not torch.equal(availability[anchor], availability2[anchor]):
                    raise AssertionError('anchor current availability changed')
                if not torch.equal(view['incomplete'][anchor], second['incomplete'][anchor]):
                    raise AssertionError('anchor raw current inputs changed')
                changed = (availability != availability2).any(-1) & valid
                strictly_prior = changed.long().cumsum(0) - changed.long()
                if not torch.equal(anchor, valid & ~changed & (strictly_prior > 0)):
                    raise AssertionError('anchor history/current definition failed')
                altered = capture(model, lambda: call(second))
                if not torch.isfinite(altered['prediction'][valid]).all():
                    raise AssertionError('nonfinite altered prediction')
                if anchor.any():
                    error = (baseline['local'][anchor] - altered['local'][anchor]).abs().max().item()
                    max_local = max(max_local, error)
                    relative = ((baseline['local'][anchor] - altered['local'][anchor]).norm(dim=-1) /
                                (baseline['local'][anchor].norm(dim=-1) + 1e-8)).max().item()
                    max_local_relative = max(max_local_relative, relative)
                    if error > 1e-5 or relative > 1e-6:
                        raise AssertionError(f'local isolation failed: absolute error {error}')
                prefix = valid & (changed.long().cumsum(0) == 0)
                for key in baseline:
                    if not torch.allclose(baseline[key][prefix], altered[key][prefix], atol=1e-5, rtol=1e-6):
                        raise AssertionError(f'unaffected causal prefix mismatch: {key}')
                metrics = anchor_metrics(baseline, altered, availability)
                counts[mode]['anchors'] += int(anchor.sum())
                counts[mode]['observed_bits'] += int(availability[valid].sum())
                counts[mode]['deleted_bits'] += int((availability - availability2)[valid].sum())
                masks[f'availability2_{mode}'] = availability2.cpu().numpy()
                masks[f'contrast_mask_{mode}'] = anchor.cpu().numpy()
                for t, b in anchor.nonzero(as_tuple=False).cpu().tolist():
                    prior_counts = meta['prior_deleted_modalities'][t, b].cpu().tolist()
                    row = dict(seed=66, rate=rate, split=split, mode=mode,
                        conversation_id=str(view['conversation_ids'][b]), utterance_index=t,
                        label=float(view['labels'][b, t]),
                        current_pattern=''.join(x for x, active in zip('ATV', availability[t,b].tolist()) if active),
                        nearest_prior_change_distance=int(meta['nearest_prior_change_distance'][t,b]),
                        prior_deleted_bit_count=int(meta['prior_deleted_bit_count'][t,b]),
                        prior_deleted_modalities=''.join(x for x, n in zip('ATV', prior_counts) if n),
                        prior_deleted_a=prior_counts[0], prior_deleted_t=prior_counts[1], prior_deleted_v=prior_counts[2],
                        pred1=float(baseline['prediction'][t,b].reshape(-1)[0]),
                        pred2=float(altered['prediction'][t,b].reshape(-1)[0]))
                    row.update({key: scalar(value[t,b]) for key, value in metrics.items()})
                    anchors.append(row)
            mask_file = output / f'{tag}_batch_{batch_index}_masks.npz'
            with mask_file.open('xb') as stream:
                np.savez_compressed(stream, **masks)
            artifacts[mask_file.name] = sha(mask_file)
            batch_count += 1
            if args.smoke:
                break
    if not args.smoke and seen != set(args.expected_ids):
        raise AssertionError('split conversation coverage mismatch')
    target = output / f'{tag}_anchors.csv'
    if anchors:
        with target.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(anchors[0]))
            writer.writeheader()
            writer.writerows(anchors)
    else:
        target.touch(exist_ok=False)
    y, p = np.asarray(labels), np.asarray(predictions)
    keep = y != 0
    score = float(f1_score(y[keep] > 0, p[keep] > 0, average='weighted'))
    reference = None
    if split == 'test' and not args.smoke:
        reference = float(read(args.reference_root / 'seed_66/metrics.json')['test'][f'{rate:.1f}']['weighted_f1'])
        if abs(reference - score) > 1e-10:
            raise AssertionError(f'original evaluation mismatch: {score} vs {reference}')
    return dict(anchors_file=target.name, anchors_sha256=sha(target), anchors=len(anchors),
                counts=counts, baseline_weighted_f1=score, reference_weighted_f1=reference,
                baseline_utterances=len(labels), nonzero_label_utterances=int(keep.sum()),
                batches=batch_count, local_absolute_max_error=max_local,
                local_relative_max_error=max_local_relative,
                identity_repeat_exact=True, prefix_allclose=True,
                artifact_sha256=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--drop-prob', type=float, default=.2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.drop_prob <= 1:
        parser.error('drop probability must be in [0,1]')
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    gpu = verify_gpu()
    args.output_root.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', label=LABEL, seed=66, started_at=datetime.now(timezone.utc).isoformat(), records=[])
    write(args.output_root / 'status.json', status)
    try:
        import torch
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        torch.set_num_threads(2)
        torch.manual_seed(66)
        source = args.reference_root / 'seed_66'
        configuration_dict(66, args.reference_root)
        raw_config = read(source / 'config.json')
        flags = ('osram_history_query_adapter', 'osram_post_grn', 'osram_history_input_gate',
                 'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate',
                 'osram_hierarchical_feature_only', 'osram_local_skip_gate',
                 'osram_memory_only_adapter', 'completion_write_to_memory', 'paired_history_views')
        if any(raw_config.get(key, False) for key in flags):
            raise ValueError('source must be original Flat')
        config = TrainConfig(**raw_config)
        if config.dataset != 'CMUMOSI' or config.mosi_task_mode != 'regression':
            raise ValueError('Requires original MOSI regression protocol')
        write(args.output_root / 'source_config.json', raw_config)
        features = [str(args.dataset_root / 'CMUMOSI/features' / name) for name in
                    ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        train, val, test, *dimensions = get_loaders(audio_root=features[0], text_root=features[1],
            video_root=features[2], num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
            num_workers=0, seed=66, validation_fraction=config.validation_fraction,
            evaluation_protocol=config.evaluation_protocol)
        loaders = dict(train=train[0], validation=val[0], test=test[0])
        ids = split_ids(loaders)
        write(args.output_root / 'split_ids.json', ids)
        device = torch.device('cuda:0')
        model = _build_model(config, tuple(dimensions)).to(device).requires_grad_(False).eval()
        files = [Path(__file__), Path(__file__).with_name('diagnostic.py'),
                 REPO / 'gcnet_missing_m3/osram.py', REPO / 'gcnet_missing_m3/model.py',
                 REPO / 'gcnet_missing_m3/train_gcnet.py']
        write(args.output_root / 'provenance.json', dict(label=LABEL, command=sys.argv,
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            hostname=platform.node(), gpu=gpu, code_sha256={str(p.relative_to(REPO)): sha(p) for p in files},
            source_config_sha256=sha(source / 'config.json'),
            reference_metrics_sha256=sha(source / 'metrics.json'),
            label_file_sha256=sha(args.dataset_root / 'CMUMOSI/CMUMOSI_features_raw_2way.pkl'),
            features=features, feature_directory_mtime_ns={p: Path(p).stat().st_mtime_ns for p in features},
            mask_epoch=0, drop_prob=args.drop_prob, model_mode='eval', training=False,
            checkpoint_selection='inherited per-rate Test-oracle', smoke=args.smoke))
        for rate in ((0.,) if args.smoke else RATES):
            checkpoint = source / f"best_miss_{rate:.1f}.pt".replace('.', 'p', 1)
            checkpoint_before = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            before = state_hash(model)
            for split in (('validation',) if args.smoke else ('validation', 'test')):
                args.expected_ids = ids[split]
                detail = evaluate(model, loaders[split], _schedules(config, split)[rate],
                                  tuple(dimensions), device, args, split, rate, args.output_root)
                after = state_hash(model)
                checkpoint_after = sha(checkpoint)
                if before != after or checkpoint_before != checkpoint_after:
                    raise AssertionError('frozen model or source checkpoint changed')
                record = dict(seed=66, rate=rate, split=split, checkpoint=str(checkpoint),
                    checkpoint_epoch=int(state['epoch']), checkpoint_sha256=checkpoint_before,
                    checkpoint_after_sha256=checkpoint_after,
                    model_state_before_sha256=before, model_state_after_sha256=after, **detail)
                status['records'].append(record)
                write(args.output_root / 'status.json', status)
                print(json.dumps(record, sort_keys=True), flush=True)
        status['status'] = 'complete'
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(args.output_root / 'status.json', status)


if __name__ == '__main__':
    main()
