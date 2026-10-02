"""Replay archived Text-history interventions on frozen paired-view A/B models."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_history_drift_20261002.run import (
    DEFAULT_DATA, RATES, read, write, sha, state_hash, split_ids, verify_gpu, scalar)


def validate_saved_pair(a, b, umask, anchors):
    valid = umask.T.astype(bool)
    assert a.shape == b.shape == (*valid.shape, 3)
    assert np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
    assert (b <= a).all() and (b.sum(-1)[valid] >= 1).all()
    assert np.array_equal(a[..., [0, 2]], b[..., [0, 2]])
    changed = np.any(a != b, axis=-1) & valid
    prior = changed.cumsum(0) - changed
    assert np.array_equal(anchors, valid & ~changed & (prior > 0))
    assert np.array_equal(a[anchors], b[anchors])


def evaluate(model, loader, schedule, dimensions, device, args, split, rate, output, source):
    import torch
    from sklearn.metrics import f1_score
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view
    from experiments.osram_history_drift_20261002.diagnostic import capture, anchor_metrics
    tag = f'{split}_miss_{rate:.1f}'.replace('.', 'p')
    archived = args.flat_results / f'{tag}_anchors.csv'
    with archived.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['mode'] == 'T']
    expected = {(r['conversation_id'], int(r['utterance_index'])): r for r in rows}
    assert len(expected) == len(rows)
    result, predictions, labels, seen, used = [], [], [], set(), set()
    artifacts = {archived.name: sha(archived)}
    maximum, relative_maximum = 0., 0.
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            view = _prepare_view(_move_batch(raw, device), schedule, 0, dimensions)
            path = args.flat_results / f'{tag}_batch_{batch_index}_masks.npz'
            artifacts[path.name] = sha(path)
            with np.load(path) as saved:
                a, b, umask, anchors = (saved[k] for k in (
                    'availability1', 'availability2_T', 'umask', 'contrast_mask_T'))
                validate_saved_pair(a, b, umask, anchors)
                assert np.array_equal(saved['conversation_ids'], np.asarray(view['conversation_ids'], dtype=str))
                assert np.array_equal(umask, view['umask'].cpu().numpy())
                assert np.array_equal(a, view['availability'].cpu().numpy())
                availability2 = torch.as_tensor(b, device=device)
                anchor = torch.as_tensor(anchors, device=device)
            expanded = torch.repeat_interleave(availability2,
                torch.tensor(dimensions, device=device), dim=-1)
            second = dict(view, availability=availability2,
                          incomplete=torch.where(expanded.bool(), view['incomplete'], 0.))
            assert torch.equal(view['incomplete'][anchor], second['incomplete'][anchor])
            call = lambda v: model([v['incomplete']], v['availability'], v['qmask'],
                v['umask'], v['lengths'], predict_missing=False)
            first = capture(model, lambda: call(view))
            if batch_index == 0:
                repeat = capture(model, lambda: call(view))
                assert all(torch.equal(first[k], repeat[k]) for k in first)
            altered = capture(model, lambda: call(second))
            valid = view['umask'].T.bool()
            assert torch.isfinite(first['prediction'][valid]).all()
            assert torch.isfinite(altered['prediction'][valid]).all()
            delta = first['local'][anchor] - altered['local'][anchor]
            if anchor.any():
                maximum = max(maximum, float(delta.abs().max()))
                relative_maximum = max(relative_maximum, float((delta.norm(dim=-1) /
                    (first['local'][anchor].norm(dim=-1) + 1e-8)).max()))
                assert maximum <= 1e-5 and relative_maximum <= 1e-6
            changed = (view['availability'] != availability2).any(-1) & valid
            prefix = valid & (changed.long().cumsum(0) == 0)
            assert all(torch.allclose(first[k][prefix], altered[k][prefix], atol=1e-5, rtol=1e-6) for k in first)
            metrics = anchor_metrics(first, altered, view['availability'])
            for t, b in anchor.nonzero(as_tuple=False).cpu().tolist():
                key = str(view['conversation_ids'][b]), t
                assert key in expected and key not in used
                used.add(key)
                row = dict(expected[key])
                assert float(row['label']) == float(view['labels'][b, t])
                row.update({name: scalar(value[t, b]) for name, value in metrics.items()})
                result.append(row)
            predictions.extend(first['prediction'][valid].reshape(-1).cpu().tolist())
            labels.extend(view['labels'].T[valid].cpu().tolist())
            seen.update(map(str, view['conversation_ids']))
    assert used == set(expected) and seen == set(args.expected_ids)
    target = output / f'{tag}_anchors.csv'
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    y, p = np.asarray(labels), np.asarray(predictions)
    keep = y != 0
    score = float(f1_score(y[keep] > 0, p[keep] > 0, average='weighted'))
    reference = float(read(source / 'metrics.json')['test'][f'{rate:.1f}']['weighted_f1']) if split == 'test' else None
    assert reference is None or abs(score - reference) < 1e-10, (score, reference)
    return dict(split=split, rate=rate, anchors_file=target.name, anchors_sha256=sha(target),
        anchors=len(result), baseline_weighted_f1=score, reference_weighted_f1=reference,
        baseline_utterances=len(labels), nonzero_label_utterances=int(keep.sum()),
        local_absolute_max_error=maximum, local_relative_max_error=relative_maximum,
        identity_repeat_exact=True, prefix_allclose=True, saved_mask_anchor_replay_exact=True,
        source_artifact_sha256=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--flat-results', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, default=Path('/data1/yb/remote_experiments/osram_paired_history_views_20261001/runs'))
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    gpu = verify_gpu()
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _attach_history_projector
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    torch.set_num_threads(2)
    torch.manual_seed(66)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for arm in ('control', 'contrastive'):
        source, output = args.runs_root / arm, args.output_root / arm
        output.mkdir()
        status = dict(status='running', arm=arm, seed=66, records=[], started_at=datetime.now(timezone.utc).isoformat())
        write(output / 'status.json', status)
        try:
            raw_config = read(source / 'config.json')
            config = TrainConfig(**raw_config)
            assert config.paired_history_views and config.seed == 66
            assert config.dataset == 'CMUMOSI' and config.mosi_task_mode == 'regression'
            write(output / 'source_config.json', raw_config)
            roots = [str(args.dataset_root / 'CMUMOSI/features' / name) for name in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
            train, val, test, *dimensions = get_loaders(audio_root=roots[0], text_root=roots[1],
                video_root=roots[2], num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
                num_workers=0, seed=66, validation_fraction=config.validation_fraction,
                evaluation_protocol=config.evaluation_protocol)
            loaders = dict(train=train[0], validation=val[0], test=test[0])
            ids = split_ids(loaders)
            assert ids == read(args.flat_results / 'split_ids.json')
            device = torch.device('cuda:0')
            model = _build_model(config, tuple(dimensions)).to(device)
            _attach_history_projector(model, config)
            model.requires_grad_(False).eval()
            projector_calls = []
            hook = model.history_contrast_projector.register_forward_pre_hook(lambda *_: projector_calls.append(1))
            write(output / 'provenance.json', dict(training=False, model_mode='eval', gpu=gpu,
                command=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
                code_sha256={str(p.relative_to(REPO)): sha(p) for p in [Path(__file__),
                    Path(__file__).with_name('diagnostic.py'), REPO / 'gcnet_missing_m3/model.py',
                    REPO / 'gcnet_missing_m3/osram.py', REPO / 'gcnet_missing_m3/train_gcnet.py']},
                source_config_sha256=sha(source / 'config.json'), source_metrics_sha256=sha(source / 'metrics.json'),
                flat_results=str(args.flat_results), flat_status_sha256=sha(args.flat_results / 'status.json'),
                checkpoint_selection='inherited per-rate Test-oracle', new_masks=False,
                inference_projector=False, seed=66))
            for rate in RATES:
                checkpoint = source / f'best_miss_{rate:.1f}.pt'.replace('.', 'p', 1)
                checkpoint_hash = sha(checkpoint)
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(state['model'], strict=True)
                before = state_hash(model)
                for split in ('validation', 'test'):
                    args.expected_ids = ids[split]
                    record = evaluate(model, loaders[split], _schedules(config, split)[rate],
                        tuple(dimensions), device, args, split, rate, output, source)
                    after = state_hash(model)
                    assert before == after and sha(checkpoint) == checkpoint_hash
                    assert not projector_calls
                    record.update(seed=66, checkpoint=str(checkpoint), checkpoint_epoch=int(state['epoch']),
                        checkpoint_sha256=checkpoint_hash, checkpoint_after_sha256=sha(checkpoint),
                        model_state_before_sha256=before, model_state_after_sha256=after, projector_calls=0)
                    status['records'].append(record)
                    write(output / 'status.json', status)
                    print(json.dumps(dict(arm=arm, **record)), flush=True)
            hook.remove()
            status['status'] = 'complete'
        except BaseException:
            status.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            status['updated_at'] = datetime.now(timezone.utc).isoformat()
            write(output / 'status.json', status)


if __name__ == '__main__':
    main()
