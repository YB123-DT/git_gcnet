"""Shared native-model factory, exact-mask replay, and first-stage training.

No dependency on an historical experiment runner is required. Numerical model,
data-loader, loss, and GCNet mask implementations remain the repository's own.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from . import protocol as p

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def load_strict(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError('Expected a full original model state under key model')
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint


def build_model(cfg, dimensions, device):
    """The exact original constructor, including its fold-specific init seed."""
    from gcnet_missing_m3.model import MissingM3GraphModel
    from gcnet_missing_m3.train_gcnet import _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import set_random_seed

    shape = _dataset_shape(cfg.dataset)
    set_random_seed(SeedBundle(cfg.seed).derive(f'missing_m3_model_init:fold:{cfg.fold}'))
    keywords = {name: getattr(cfg, name) for name in inspect.signature(MissingM3GraphModel).parameters
                if hasattr(cfg, name)}
    for name in ('base_model', 'adim', 'tdim', 'vdim', 'D_e', 'graph_hidden_size'):
        keywords.pop(name, None)
    keywords.update(n_speakers=int(shape['num_speakers']), n_classes=int(shape['num_classes']),
        time_attn=cfg.time_attention, no_cuda=device.type != 'cuda',
        complete_state_jepa=cfg.training_objective == 'complete-state',
        write_state_completion=cfg.training_objective == 'write-state',
        future_state_jepa=cfg.training_objective == 'future-state')
    model = MissingM3GraphModel(cfg.base_model, *dimensions, cfg.hidden, cfg.hidden // 2,
                               **keywords).to(device)
    return model.requires_grad_(True).train()


def load_data_manifest():
    if sha(p.INPUTS) != p.INPUTS_SHA256:
        raise ValueError('Accepted MOSEI data manifest changed')
    manifest = json.loads(p.INPUTS.read_text())['data']['CMUMOSEI']
    for path, digest in manifest['files'].items():
        if sha(path) != digest:
            raise ValueError('Accepted MOSEI feature/label changed: ' + path)
    return manifest, dict(path=str(p.INPUTS), sha256=p.INPUTS_SHA256,
                          verified_files=len(manifest['files']), all_file_hashes_exact=True)


def bind_data_environment(manifest, cache_root):
    """Must be called BEFORE importing any trainer that imports config.py."""
    if 'config' in sys.modules:
        expected = Path(manifest['dataset_root']).resolve()
        actual = Path(os.environ.get('GCNET_DATASET_ROOT', '')).resolve()
        if actual != expected:
            raise RuntimeError('config.py was imported before the accepted dataset was bound')
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(cache_root)
    os.environ.pop('GCNET_MELD_PICKLE', None)


def require_environment(gpu_uuid):
    import sklearn

    if gpu_uuid == p.FORBIDDEN_GPU or not gpu_uuid.startswith('GPU-'):
        raise ValueError('Bind one healthy physical GPU UUID; GPU4 is forbidden')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the requested real GPU UUID')
    environment = dict(python=sys.executable, python_version=sys.version, torch=torch.__version__,
                       cuda=torch.version.cuda, numpy=np.__version__, sklearn=sklearn.__version__)
    if (Path(sys.executable).resolve() != p.PYTHON.resolve() or
            (environment['torch'], environment['cuda'], environment['numpy'], environment['sklearn']) !=
            ('1.13.1+cu117', '11.7', '1.26.4', '1.3.2')):
        raise ValueError('Require unchanged cmerc-repro environment: ' + str(environment))
    # Torch1.13's UUID parser may report zero devices while CUDA itself sees one.
    # Repair BOTH imported aliases, exactly as in the accepted repaired parent.
    runtime = torch._C._cuda_getDeviceCount()
    original, random_count = torch.cuda.device_count(), torch.cuda.random.device_count()
    if runtime != 1:
        raise RuntimeError('Require exactly one UUID-isolated CUDA device')
    if original != runtime:
        torch.cuda.device_count = lambda: runtime
    torch.cuda.random.device_count = torch.cuda.device_count
    repair = dict(runtime_count=runtime, original_python_count=original,
                  original_random_count=random_count, random_alias_repaired=random_count != runtime)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    matches = [row.split(',') for row in rows.splitlines() if row.split(',')[1].strip() == gpu_uuid]
    if len(matches) != 1 or int(matches[0][0]) == 4:
        raise RuntimeError('GPU UUID is not a verified healthy physical device')
    index, uuid, free, total = [v.strip() for v in matches[0]]
    return environment, dict(physical_index=int(index), uuid=uuid, free_mib=int(free),
                             total_mib=int(total)), repair


def source_hashes():
    paths = [ROOT / 'config.py']
    for directory in ('gcnet_missing_m3', 'gcnet_modality_jepa', 'gcnet_jepa_replacement',
                      'gcnet_plci_jepa', 'experiments/cam_unified_20261030'):
        paths.extend((ROOT / directory).rglob('*.py'))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}


def setup(variant, output, gpu_uuid):
    environment, gpu, repair = require_environment(gpu_uuid)
    manifest, proof = load_data_manifest()
    bind_data_environment(manifest, Path(output).parent / 'feature_cache' / 'CMUMOSEI')
    if sha(p.PARENT / 'config.json') != p.PARENT_CONFIG_SHA256:
        raise ValueError('Protected Flat config SHA256 changed')
    from gcnet_missing_m3 import train_gcnet as trainer

    raw = json.loads((p.PARENT / 'config.json').read_text())
    cfg = trainer.TrainConfig(**p.variant_config(raw, variant))
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset=cfg.dataset, num_folder=1, batch_size=cfg.batch_size, num_workers=0,
        seed=66, validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    model = build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    initial = state_digest(model)
    if sum(v.numel() for v in model.parameters()) != 13509793 or initial != p.INITIAL_SHA256:
        raise RuntimeError('Original Flat dimensions or protected same-seed initialization changed')
    masks = torch.load(p.MASKS, map_location='cpu', weights_only=False)
    masks = masks.get('test', masks)
    if set(masks) != set(p.RATES):
        raise ValueError('Protected mask bank must contain eight test rates')
    mask_hashes = {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
                                              for row in masks[rate])).hexdigest() for rate in p.RATES}
    parent = json.loads((p.PARENT / 'SUMMARY.json').read_text())
    if parent['status'] != 'complete' or parent['initial_backbone_sha256'] != initial:
        raise ValueError('Protected parent is incomplete or its initialization differs')
    reference_hashes = parent['identity']['mask_reference']['hashes']
    if reference_hashes != mask_hashes:
        raise ValueError('Protected parent mask hashes differ')
    split = {name: loader.protocol_metadata for name, loader in
             (('train', train[0]), ('validation', validation[0]), ('test', test[0]))}
    if split != parent['identity']['split']:
        raise ValueError('Official dataset split/order differs from the accepted parent')
    identity = dict(version=1, label=p.LABEL, variant=variant, config=asdict(cfg), dimensions=dimensions,
        data_manifest=proof, parent_config=dict(path=str(p.PARENT / 'config.json'), sha256=p.PARENT_CONFIG_SHA256),
        initial_backbone_sha256=initial, original_checkpoint_loaded_for_training=False,
        mask_reference=dict(path=str(p.MASKS), sha256=sha(p.MASKS), hashes=mask_hashes),
        split=split, environment=environment, source_hashes=source_hashes(),
        selection='per-rate TEST W-F1; strict improvement; earliest ties',
        task_path_ablation='Full architecture retained; blocked task slots are zero before Flat input LN')
    return dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), model=model,
                train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
                test_masks=masks, identity=identity, environment=environment, gpu=gpu, rng_repair=repair,
                initial_backbone_sha256=initial)


def forward(model, view):
    if hasattr(model, 'forward_view'):
        return model.forward_view(view)
    prediction, _, _, completion = model([view['incomplete']], view['availability'], view['qmask'],
        view['umask'], view['lengths'], predict_missing=False)
    if completion is not None:
        raise RuntimeError('Completion must remain disabled')
    return prediction


def metadata(data):
    valid = data[7].bool().cpu().numpy()
    ids = np.asarray([str(v) for v in data[-1]])
    return dict(labels=data[8].detach().cpu().numpy()[valid],
        speakers=data[6].detach().cpu().numpy()[valid],
        conversation_ids=np.broadcast_to(ids[:, None], valid.shape)[valid],
        utterance_indices=np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid])


def saved_view(bundle, data, record):
    for key, value in metadata(data).items():
        np.testing.assert_array_equal(value, record[key], err_msg='Protected test order differs: ' + key)
    availability = record['availability'].to(data[7].device)
    valid = data[7].T.bool()
    if availability.shape != (*valid.shape, 3) or not availability[valid].bool().any(-1).all():
        raise ValueError('Protected availability shape/valid utterance visibility is invalid')
    if availability[~valid].bool().any():
        raise ValueError('Protected availability must zero padding')
    return bundle['trainer']._prepare_view_from_primary_masks(data, availability, availability,
                                                             bundle['dimensions'])


def evaluate(bundle, model, rate, max_batches=None):
    loader, trainer = bundle['test_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    model.eval()
    pieces = {}
    with isolated_rng(66 + 99193), torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            record = bundle['test_masks'][rate][index]
            view = saved_view(bundle, data, record)
            pred = forward(model, view)
            valid = data[7].bool()
            values = dict(metadata(data), prediction=pred.transpose(0, 1)[valid].cpu().numpy(),
                          availability=view['availability'].transpose(0, 1)[valid].cpu().numpy())
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    labels, prediction = arrays['labels'].reshape(-1), arrays['prediction'].reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite regression prediction')
    metrics = trainer._metrics(bundle['cfg'].dataset, labels, prediction, 'regression')
    return dict(ACC=100 * metrics['accuracy'], WF1=100 * metrics['weighted_f1'],
                MSE=float(np.mean((labels - prediction) ** 2)), N=int(labels.size),
                mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest()), arrays


def train_epoch(bundle, model, optimizer, epoch, max_batches=None):
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule

    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    model.train()
    digest, count, total_loss, steps, grad_norm = hashlib.sha256(), 0, 0., 0, 0.
    selected_rates = []
    gradient_groups = {}
    tick = time.monotonic()
    for index, raw in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        data = trainer._move_batch(raw, next(model.parameters()).device)
        rate = cycle.rate_for(epoch, index)
        view = trainer._prepare_view(data, schedules[rate], epoch, bundle['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, view)
        loss = trainer._task_loss(cfg.dataset, prediction, view['labels'], view['umask'],
            cfg.mosi_task_mode, cfg.task_regression_loss, cfg.task_smooth_l1_beta)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite original emotion MSE')
        loss.backward()
        for name, value in model.named_parameters():
            if value.grad is not None and not torch.isfinite(value.grad).all():
                raise RuntimeError('Nonfinite gradient: ' + name)
        if max_batches is not None:
            # Smoke-only detail; a zero gradient through the original zero-init
            # adapter on step one is recorded honestly, not called disconnection.
            for name, value in model.named_parameters():
                group = next((prefix for prefix in ('observed_set.projectors', 'osram.key_projectors',
                    'osram.value_projectors', 'osram.query_projector', 'osram.local_path',
                    'osram.local_skip', 'osram.emotion_adapter', 'smax_fc') if name.startswith(prefix)), 'other')
                row = gradient_groups.setdefault(group, dict(connected=0, nonzero=0, finite=True))
                if value.grad is not None:
                    row['connected'] += 1
                    row['nonzero'] += int(bool(value.grad.ne(0).any()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite global gradient norm')
        optimizer.step()
        n = int(view['umask'].sum())
        count += n
        total_loss += float(loss.detach()) * n
        grad_norm += float(norm)
        steps += 1
        selected_rates.append(float(rate))
        digest.update(json.dumps([str(v) for v in data[-1]], separators=(',', ':')).encode())
        digest.update(view['availability'].detach().cpu().numpy().tobytes())
    if not steps:
        raise ValueError('No training batches were processed')
    return dict(epoch=epoch + 1, task_MSE=total_loss / count, N=count, optimizer_steps=steps,
        mask_trace_sha256=digest.hexdigest(), rate_per_batch=selected_rates,
        learning_rates={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
        gradient_norm=grad_norm / steps, gradients=gradient_groups,
        seconds=time.monotonic() - tick)
