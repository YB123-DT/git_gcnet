"""One fixed Flat parent; only original Nested trains; validation selects one epoch."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from types import MethodType

from experiments.osram_core20_20261005.run import sha, write, now, verify_snapshot

PREFIX = 'osram.meaningful_block.'
LABEL = 'INTERNAL DIAGNOSTIC ONLY; source Test-oracle; Stage2 validation mean-8 W-F1'
RATES = tuple(i / 10 for i in range(8))


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith(PREFIX):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_parent(model, state):
    expected = {k for k in model.state_dict() if k.startswith(PREFIX)}
    if not expected or set(model.state_dict()) - set(state) != expected or set(state) - set(model.state_dict()):
        raise ValueError('parent must match exactly except newly added Nested')
    result = model.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == expected and not result.unexpected_keys


def freeze_parent(model):
    import torch
    model.requires_grad_(False)
    model.osram.meaningful_block.requires_grad_(True)
    def train_nested_only(self, mode=True):
        torch.nn.Module.train(self, False)
        self.osram.meaningful_block.train(mode)
        return self
    model.train = MethodType(train_nested_only, model)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in params} == {id(p) for p in model.osram.meaningful_block.parameters()}
    return params


def summary(metrics):
    return dict(mean_8rate=100 * sum(v['weighted_f1'] for v in metrics.values()) / 8,
                high_missing=100 * sum(metrics[f'{r:.1f}']['weighted_f1'] for r in RATES[-3:]) / 3)


def main():
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, train_epoch, evaluate_rate, _apply_epoch_learning_rate
    from gcnet_modality_jepa.train_gcnet import get_loaders, set_random_seed
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    snapshot = verify_snapshot(source)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    rows = [line.strip().split(', ') for line in devices.splitlines()]
    assert any(index != '4' and uuid == args.gpu_uuid for index, uuid in rows)
    ref = json.loads(args.reference.read_text())
    assert ref['seed'] == 66 and ref['dataset'] == 'CMUMOSI'
    assert ref['training_objective'] == 'emotion-only' and ref['epochs'] == 100
    assert ref['optimizer'] == 'adam' and ref['learning_rate'] == .001 and ref['weight_decay'] == 1e-5
    assert ref.get('osram_adapter_hidden_dim', 0) == 0
    base_config = TrainConfig(**ref)
    c = TrainConfig(**dict(asdict(base_config), osram_meaningful_block='nested_gnn_rooted_evidence',
                          checkpoint_selection='validation'))
    parent = args.reference.parent / 'best_miss_0p7.pt'
    data = json.loads(args.data_manifest.read_text())
    for path, digest in data['files'].items():
        if sha(path) != digest:
            raise ValueError('data changed: ' + path)
    os.environ['GCNET_DATASET_ROOT'] = data['dataset_root']
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    provenance = dict(status='running', label=LABEL, started_utc=now(), pid=os.getpid(), server='biggpu',
        gpu_uuid=args.gpu_uuid, source_checkpoint=str(parent), source_checkpoint_sha256=sha(parent),
        source_commit=snapshot['code_commit'], snapshot_sha256=sha(source / 'SNAPSHOT.json'),
        data_manifest_sha256=sha(args.data_manifest), reference_sha256=sha(args.reference),
        selection='mean eight validation W-F1, strict improvement, includes epoch0; test only before/after',
        resume_supported=False, torch_version=torch.__version__)
    write(out / 'PROVENANCE.json', provenance)
    write(out / 'config.json', asdict(c))
    try:
        torch.set_num_threads(2)
        set_random_seed(66)
        roots = data['feature_roots']
        train, val, test, ad, td, vd = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
            num_folder=1, dataset=c.dataset, batch_size=c.batch_size, num_workers=0, seed=c.seed,
            validation_fraction=c.validation_fraction, evaluation_protocol=c.evaluation_protocol)
        metadata = {name: loaders[0].protocol_metadata for name, loaders in [('train', train), ('validation', val), ('test', test)]}
        sets = [set(m['indices']) for m in metadata.values()]
        assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i))
        write(out / 'SPLITS.json', metadata)
        dims, device = (ad, td, vd), torch.device('cuda:0')
        state = torch.load(parent, map_location='cpu', weights_only=False)
        model = _build_model(c, dims).to(device)
        load_parent(model, state['model'])
        params = freeze_parent(model)
        anchor_hash = frozen_hash(model)
        provenance.update(source_epoch=int(state['epoch']), frozen_state_sha256=anchor_hash,
                          trainable_parameters=sum(p.numel() for p in params))
        assert provenance['trainable_parameters'] == 159235
        optimizer = torch.optim.Adam(params, lr=c.learning_rate, weight_decay=c.weight_decay)
        schedules = {split: _schedules(c, split) for split in ('train', 'validation', 'test')}
        def evaluate(net, split, collect=False):
            loader = {'validation': val, 'test': test}[split][0]
            metrics, predictions = {}, {}
            for rate in RATES:
                key = f'{rate:.1f}'
                metrics[key], predictions[key] = evaluate_rate(net, loader, schedules[split][rate], c.dataset,
                    dims, device, collect, c.mosi_task_mode, c.task_regression_loss, c.task_smooth_l1_beta)
            return metrics, predictions
        initial, initial_pred = evaluate(model, 'test', True)
        # Explicitly verify actual Flat output, not only zero-initialized parameter values.
        flat = _build_model(base_config, dims).to(device)
        flat.load_state_dict(state['model'], strict=True)
        flat_metrics, flat_pred = evaluate(flat, 'test', True)
        parity_errors = {}
        for rate in initial:
            for key in initial_pred[rate]:
                if key == 'predictions':
                    a, b = initial_pred[rate][key], flat_pred[rate][key]
                    parity_errors[rate] = float(np.max(np.abs(a - b)))
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
                    assert np.array_equal(a > 0, b > 0), 'zero-init changed polarity'
                else:
                    assert np.array_equal(initial_pred[rate][key], flat_pred[rate][key]), (rate, key)
            assert initial[rate]['weighted_f1'] == flat_metrics[rate]['weighted_f1']
            assert initial[rate]['mask_sha256'] == flat_metrics[rate]['mask_sha256']
            np.savez_compressed(out / f'flat_miss_{rate}.npz', **flat_pred[rate])
        reference_metrics = json.loads((args.reference.parent / 'metrics.json').read_text())
        assert abs(flat_metrics['0.7']['weighted_f1'] - reference_metrics['test']['0.7']['weighted_f1']) < 1e-10
        initial, initial_pred = flat_metrics, flat_pred
        del flat, state
        best, _ = evaluate(model, 'validation')
        best_score, best_epoch = summary(best)['mean_8rate'], 0
        initial_nested = {k: v.detach().cpu().clone() for k, v in model.osram.meaningful_block.state_dict().items()}
        history = []
        def save(name, epoch):
            payload = dict(nested=model.osram.meaningful_block.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, best_epoch=best_epoch, best_validation=best, config=asdict(c),
                source_checkpoint=str(parent), source_sha256=provenance['source_checkpoint_sha256'],
                frozen_sha256=anchor_hash, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), resume_supported=False)
            tmp = out / (name + '.tmp')
            torch.save(payload, tmp)
            tmp.replace(out / name)
        save('best.pt', 0)
        write(out / 'INITIAL.json', dict(test=initial, test_summary=summary(initial), validation=best,
            zero_init_polarity_and_wf1_exact=True, zero_init_max_abs_error=parity_errors,
            zero_init_tolerance=dict(rtol=1e-6, atol=1e-6)))
        write(out / 'PROVENANCE.json', provenance)
        print('INITIAL ' + json.dumps(summary(initial)) + ' zero-init numerical/polarity parity PASS', flush=True)
        # Builders/parity evaluation do not determine training RNG.
        set_random_seed(66)
        for epoch in range(c.epochs):
            _apply_epoch_learning_rate(optimizer, c, epoch)
            train[0].sampler.set_epoch(epoch)
            training = train_epoch(model, train[0], optimizer, c, schedules['train'], epoch, dims, device)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            assert frozen_hash(model) == anchor_hash, 'frozen state drift'
            if epoch == 0:
                assert any(not torch.equal(v.cpu(), initial_nested[k]) for k, v in model.osram.meaningful_block.state_dict().items())
            metrics, _ = evaluate(model, 'validation')
            score = summary(metrics)['mean_8rate']
            assert np.isfinite(score)
            if score > best_score:
                best, best_score, best_epoch = metrics, score, epoch + 1
                save('best.pt', epoch + 1)
            history.append(dict(epoch=epoch + 1, train=training, validation=metrics, validation_mean=score,
                                frozen_unchanged=True, best_epoch=best_epoch))
            write(out / 'history.json', history)
            save('last.pt', epoch + 1)
            print(f'epoch={epoch+1}/100 validation={score:.4f} best={best_score:.4f}@{best_epoch}', flush=True)
        selected = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        model.osram.meaningful_block.load_state_dict(selected['nested'], strict=True)
        assert frozen_hash(model) == anchor_hash
        final, final_pred = evaluate(model, 'test', True)
        paired = {}
        for rate in initial:
            before, after = initial_pred[rate], final_pred[rate]
            for key in ('labels', 'availability'):
                assert np.array_equal(before[key], after[key])
            assert initial[rate]['mask_sha256'] == final[rate]['mask_sha256']
            labels = before['labels'].reshape(-1)
            valid = labels != 0
            ok0 = (before['predictions'].reshape(-1) > 0) == (labels > 0)
            ok1 = (after['predictions'].reshape(-1) > 0) == (labels > 0)
            paired[rate] = dict(N=int(valid.sum()), corrections=int((valid & ~ok0 & ok1).sum()),
                                harms=int((valid & ok0 & ~ok1).sum()))
            np.savez_compressed(out / f'nested_miss_{rate}.npz', **after)
        write(out / 'metrics.json', dict(label=LABEL, source_epoch=provenance['source_epoch'], best_epoch=best_epoch,
            baseline=initial, nested=final, baseline_summary=summary(initial), nested_summary=summary(final), paired=paired,
            best_validation=best, selection_includes_epoch0=True))
        provenance.update(status='complete', completed_utc=now(), frozen_state_unchanged=True, outputs_verified=True)
        provenance['artifact_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'PROVENANCE.json'}
        write(out / 'PROVENANCE.json', provenance)
    except BaseException as error:
        write(out / 'PROVENANCE.json', dict(provenance, status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
