"""Frozen-adapter training, exact-mask replay and recovery utilities for TASR."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def freeze_for_adapter(model):
    model.eval().requires_grad_(False)
    model.osram.tasr.requires_grad_(True)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names or any(not name.startswith('osram.tasr.') for name in names):
        raise RuntimeError('Only the added TASR adapter may train')
    return names


def check_adapter_gradients(model):
    nonzero = connected = 0
    for name, value in model.named_parameters():
        if not name.startswith('osram.tasr.'):
            if value.requires_grad or value.grad is not None:
                raise RuntimeError('Frozen original parameter received training gradient: '+name)
        elif value.grad is not None:
            connected += 1
            if not torch.isfinite(value.grad).all(): raise RuntimeError('Nonfinite TASR gradient: '+name)
            nonzero += bool(value.grad.ne(0).any())
    if not connected: raise RuntimeError('No adapter gradient connection')
    return dict(connected_parameters=connected, nonzero_parameters=nonzero)


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('osram.tasr.'): continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(model, optimizer, *, epoch, identity, history, best_key):
    return dict(version=1, epoch=epoch, identity=identity, history=history, best_key=best_key,
                adapter={k: v.detach().cpu().clone() for k,v in model.osram.tasr.state_dict().items()},
                optimizer=optimizer.state_dict(), rng=rng_state(),
                recovery='Strict-load immutable protected anchor, then adapter/optimizer/RNG; no copied backbone weights')


def atomic_torch_save(path, state):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state, temporary); temporary.replace(path)


def restore_checkpoint(model, optimizer, checkpoint, *, identity):
    if checkpoint['identity'] != identity: raise ValueError('Resume identity differs from immutable run')
    model.osram.tasr.load_state_dict(checkpoint['adapter'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'])


def resolve_selection(checkpoint, output, expected_sha):
    root = Path(output).resolve()
    artifact = checkpoint['selection_artifact']
    path = (root/artifact['path']).resolve()
    if not path.is_relative_to(root): raise ValueError('Selection artifact escapes protected run')
    if artifact['sha256'] != expected_sha or sha(path) != expected_sha:
        raise ValueError('Selected artifact hash differs from protected registry')
    return path


def read_selected_artifacts(case):
    """Resolve all eight original mask files using real checkpoint metadata."""
    selected, references = {}, {}
    for rate, ref in case['evaluation_references'].items():
        if sha(ref['checkpoint']) != ref['checkpoint_sha256']:
            raise ValueError('Protected checkpoint hash changed: '+rate)
        checkpoint = torch.load(ref['checkpoint'], map_location='cpu', weights_only=False)
        if checkpoint['config'] != case['expected_config'] or checkpoint['epoch'] != ref['epoch']:
            raise ValueError('Protected effective config/epoch changed: '+rate)
        path = resolve_selection(checkpoint, case['output'], ref['selection_sha256'])
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != ref['mask_sha256']:
            raise ValueError('Exact selected availability hash changed: '+rate)
        np.testing.assert_array_equal(arrays['predictions'], arrays['logits'].argmax(1))
        selected[rate] = arrays
        references[rate] = dict(ref, actual_selection=str(path))
    return selected, references


def metadata(data):
    valid = data[7].bool().cpu().numpy()
    ids = np.asarray([str(v) for v in data[-1]])
    return dict(labels=data[8].detach().cpu().numpy()[valid].astype(np.int64),
                speakers=data[6].detach().cpu().numpy()[valid],
                conversation_ids=np.broadcast_to(ids[:,None], valid.shape)[valid],
                utterance_indices=np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid])


def forward(model, view):
    logits, _, _, completion = model([view['incomplete']], view['availability'], view['qmask'],
        view['umask'], view['lengths'], predict_missing=False)
    if completion is not None: raise RuntimeError('Completion must remain disabled')
    return logits


def collect(model, loader, trainer, dimensions, classes, *, saved=None, pattern=None, batches=None):
    """One frozen trajectory per condition; all cohort/mask comparisons are exact."""
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics, fixed_availability
    pieces = {k: [] for k in ('labels','speakers','conversation_ids','utterance_indices','availability','logits')}
    diagnostics, cursor = [], 0
    model.eval()
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for index, raw in enumerate(loader):
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta = metadata(data); n = len(meta['labels'])
            if saved is not None:
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, saved[key][cursor:cursor+n])
            if pattern is not None:
                availability = fixed_availability(data[7], pattern)
            elif batches is not None:
                record = batches[index]
                for key, value in meta.items(): np.testing.assert_array_equal(value, record[key])
                availability = record['availability'].to(data[7].device)
            elif saved is not None:
                batch = torch.zeros((*data[7].shape,3), device=data[7].device)
                batch[data[7].bool()] = torch.as_tensor(saved['availability'][cursor:cursor+n], device=batch.device, dtype=batch.dtype)
                availability = batch.transpose(0,1)
            else:
                raise ValueError('Explicit common masks or fixed pattern required')
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, dimensions)
            current = forward(model, view).transpose(0,1)[data[7].bool()].cpu().numpy()
            if not np.isfinite(current).all(): raise RuntimeError('Nonfinite task logits')
            for key in meta: pieces[key].append(meta[key])
            pieces['availability'].append(availability.transpose(0,1)[data[7].bool()].cpu().numpy())
            pieces['logits'].append(current)
            diagnostic = dict(getattr(model.osram, 'last_diagnostics', {}))
            diagnostics.append(dict(valid=n, source_tracking=diagnostic.get('source_tracking', {}),
                                    tasr=diagnostic.get('tasr', {})))
            cursor += n
    if saved is not None and cursor != len(saved['labels']): raise ValueError('Incomplete common cohort')
    arrays = {key: np.concatenate(parts) for key,parts in pieces.items()}
    arrays['predictions'] = arrays['logits'].argmax(1)
    metrics = classification_metrics(arrays['labels'],arrays['logits'],classes)
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    return metrics, arrays, diagnostics


def paired_counts(original, candidate):
    for key in ('labels','conversation_ids','utterance_indices','speakers','availability'):
        if key in original or key in candidate:
            np.testing.assert_array_equal(original[key],candidate[key])
    old = original['predictions'] == original['labels']
    new = candidate['predictions'] == candidate['labels']
    return dict(corrections=int((~old & new).sum()), harms=int((old & ~new).sum()),
                both_correct=int((old & new).sum()), both_wrong=int((~old & ~new).sum()), N=len(old))


def aggregate_rates(rates):
    from .protocol import RATES
    return dict(rates=rates, **{scope:{m:sum(rates[r][m] for r in chosen)/len(chosen)
        for m in ('ACC','UA','WF1','CE')} for scope, chosen in (('mean',RATES),('high',RATES[5:]))})


def attach_tasr(model, mode):
    """Attach only new parameters; preserve the backbone and global RNG exactly."""
    from gcnet_missing_m3.tasr import TASRAdapter
    if mode not in ('real','neutral') or getattr(model.osram,'tasr',None) is not None:
        raise ValueError('Attach one real or neutral adapter exactly once')
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(66+731321)
        adapter = TASRAdapter(model.osram.latent_dim,model.osram.num_heads*model.osram.value_dim, source_mode=mode)
    model.osram.tasr = adapter.to(device)
    model.osram.osram_tasr = True
    model.osram.osram_source_tracking = True
    model.osram.osram_tasr_source_mode = mode
    return freeze_for_adapter(model)


def setup_case(dataset, gpu_uuid, output):
    """Current strict model + mandatory exact historical replay, or fail closed.

    No historical package is silently replaced or partially loaded. The current
    implementation must reproduce the protected checkpoint's saved prediction
    before any adapter can train. A failed parity gate requires investigation,
    never strict=False or a substitute checkpoint.
    """
    from .protocol import build_cases, adapter_split_indices
    case = build_cases()[dataset]
    if not gpu_uuid.startswith('GPU-'): raise ValueError('A real GPU UUID is required')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    output = Path(output)
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent/'feature_cache'/dataset)
    if dataset == 'MELD':
        feature = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
        if sha(feature) != case['dataset_sha256']: raise ValueError('Accepted MELD feature hash changed')
        os.environ['GCNET_MELD_PICKLE'] = str(feature)
        os.environ['GCNET_DATASET_ROOT'] = str(Path(case['numerical_source']['source'])/'dataset')
        roots = ('MELD-pickle',)*3
    else:
        data_root = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
        os.environ['GCNET_DATASET_ROOT'] = str(data_root)
        os.environ.pop('GCNET_MELD_PICKLE',None)
        if sha(data_root/'IEMOCAP/IEMOCAP_features_raw_6way.pkl') != case['label_sha256']:
            raise ValueError('Accepted IEMOCAP labels changed')
        roots = tuple(case['feature_roots'])
        if not all(Path(path).is_dir() for path in roots): raise FileNotFoundError('IEMOCAP features unavailable')
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders, _build_protocol_loader
    from gcnet_modality_jepa.protocol import SeedBundle
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_persistent_patterns_20261018.evaluate import evaluate_condition
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    environment = require_replication_environment(torch)
    repair_uuid_rng(torch)
    gpu = guard_gpu(gpu_uuid)
    torch.set_num_threads(2)
    cfg = trainer.TrainConfig(**case['expected_config'])
    train,validation,test,*dimensions = get_loaders(*roots,dataset=dataset,
        num_folder=1 if dataset=='MELD' else 5,batch_size=32,num_workers=0,
        seed=66,validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    original_train, original_validation, test_loader = train[case['fold']-1],validation[case['fold']-1],test[case['fold']-1]
    if dataset == 'IEMOCAPSix':
        train_ids,val_ids,split = adapter_split_indices(original_train.sampler.indices,original_train.dataset.vids)
        split_hash = hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
        def subset(ids,name):
            return _build_protocol_loader(original_train.dataset,ids,dataset,5,name,32,0,
                                          SeedBundle(66),split_hash,'adapter-training-conversation-holdout')
        train_loader,validation_loader = subset(train_ids,'train'),subset(val_ids,'validation')
    else:
        train_loader,validation_loader = original_train,original_validation
        split = dict(method='official-train-dev-test',backbone_previously_saw_adapter_validation=False)
    train_ids, val_ids, test_ids = (set(loader.sampler.indices) for loader in (train_loader,validation_loader,test_loader))
    if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids:
        raise ValueError('Adapter training/validation/test conversations overlap')
    split.update(train_indices=sorted(train_ids),validation_indices=sorted(val_ids),test_indices=sorted(test_ids))
    model = _build_model(cfg,tuple(dimensions),torch.device('cuda:0'))
    if sha(case['checkpoint']) != case['checkpoint_sha256']: raise ValueError('Anchor checkpoint changed')
    checkpoint = torch.load(case['checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['config'] != case['expected_config']: raise ValueError('Anchor effective config mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval().requires_grad_(False)
    selected,references = read_selected_artifacts(case)
    test_loader.sampler.set_epoch(0)
    with isolated_rng(66+76191):
        parity, original = evaluate_condition(model,test_loader,tuple(dimensions),selected['0.3'],trainer,None,case['classes'])
    # Class equality alone is not enough to claim unchanged numerical behavior.
    if parity['checks']['selected_logit_max_abs_error'] != 0:
        raise RuntimeError('Current code is not bit-exact to protected historical anchor; bridge/review required')
    return dict(model=model,trainer=trainer,train_loader=train_loader,validation_loader=validation_loader,
                test_loader=test_loader,dimensions=tuple(dimensions),case=case,cfg=cfg,
                environment=environment,gpu=gpu,split=split,selected=selected,references=references,
                anchor_parity=parity,anchor_predictions=original,
                numeric_strategy='current-strict-load-with-bit-exact-historical-selected-mask-gate')
