"""Original IEMOCAP training with ONLY the selected B6 PCA settings added.

Standalone launcher imports an immutable numerical checkout, never the changing
working tree. INTERNAL TEST-ORACLE SCREEN; no claim of independent validation.
"""
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
import time

import numpy as np

SOURCE_SHA = '785a01242ef1a82b97d5337acbb33ba963df5537'
DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
LABEL_HASHES = {
    'IEMOCAPFour': ('IEMOCAP_features_raw_4way.pkl', '072cffa9c528dd0536792de7ac85d5a0efaf25d4a9e3ed01f7e45453547cc9fd'),
    'IEMOCAPSix': ('IEMOCAP_features_raw_6way.pkl', 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def class_metrics(labels, logits, classes):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels, logits = np.asarray(labels), np.asarray(logits)
    if (classes not in (4, 6) or labels.ndim != 1 or not len(labels)
            or logits.shape != (len(labels), classes) or not np.isfinite(logits).all()
            or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or (labels >= classes).any()):
        raise ValueError('Require aligned finite IEMOCAP four/six-class labels/logits')
    prediction = logits.argmax(1)
    recall = recall_score(labels, prediction, labels=np.arange(classes), average=None, zero_division=0)
    scores = logits - logits.max(1, keepdims=True)
    ce = np.log(np.exp(scores).sum(1)) - scores[np.arange(len(labels)), labels]
    return dict(ACC=100 * accuracy_score(labels, prediction), UA=100 * float(recall.mean()),
                WF1=100 * f1_score(labels, prediction, average='weighted', zero_division=0),
                CE=float(ce.mean()), recall=(100 * recall).tolist(),
                support=np.bincount(labels, minlength=classes).tolist(), N=len(labels))


def aggregate(rates):
    if set(rates) != {str(i / 10) for i in range(8)}:
        raise ValueError('Eight complete missing rates required')
    return dict(rates=rates,
                mean={k: float(np.mean([r[k] for r in rates.values()])) for k in ('ACC', 'UA', 'WF1', 'CE')},
                high={k: float(np.mean([rates[str(i / 10)][k] for i in (5, 6, 7)])) for k in ('ACC', 'UA', 'WF1', 'CE')})


def repair_uuid_rng(torch):
    """Torch1.13 imports device_count by value in cuda.random; patch both aliases.

    Only repair enumeration of the ONE already UUID-isolated runtime device.
    This also ensures manual_seed_all and recovery save the real GPU RNG.
    """
    runtime = torch._C._cuda_getDeviceCount()
    original = torch.cuda.device_count()
    random_count = torch.cuda.random.device_count()
    if runtime != 1:
        raise RuntimeError('Require exactly one UUID-bound CUDA runtime device')
    if original != runtime:
        torch.cuda.device_count = lambda: runtime
    torch.cuda.random.device_count = torch.cuda.device_count
    return dict(runtime_count=runtime, original_python_count=original,
                original_random_count=random_count, random_alias_repaired=random_count != runtime)


def require_replication_environment(torch):
    """Fail before training rather than silently changing the comparison runtime."""
    import sklearn
    expected_python = Path('/data2/yb/conda_envs/cmerc-repro/bin/python')
    environment = dict(executable=sys.executable, python=sys.version,
                       torch=torch.__version__, cuda=torch.version.cuda,
                       numpy=np.__version__, sklearn=sklearn.__version__)
    if (Path(sys.executable).resolve() != expected_python.resolve()
            or environment['torch'] != '1.13.1+cu117'
            or environment['cuda'] != '11.7'
            or environment['numpy'] != '1.26.4'
            or environment['sklearn'] != '1.3.2'):
        raise ValueError(f'Require the accepted cmerc-repro comparison runtime: {environment}')
    return environment


def validate(config, *, flat_only=False, flat_replication=False):
    if flat_replication and not flat_only:
        raise ValueError('Flat replication requires the explicit Flat-only path')
    expected = dict(epochs=100, batch_size=32, optimizer='adam', learning_rate=0.001,
        weight_decay=1e-5, dropout=0.5, projector_dropout=0.1, local_fusion_dropout=0.2,
        lr_schedule='constant', gradient_clip_norm=1.0, emotion_loss_mode='sample-mean',
        training_objective='emotion-only', missing_mask_protocol='gcnet-random-mask-v1',
        train_rate_mode='cyclic', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='accuracy', preserve_selection_artifacts=True,
        evaluation_protocol='official', osram_output_dim=1600, osram_adapter_hidden_dim=0,
        osram_readout_fusion='flat', osram_disable_local_skip=False,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, osram_write_step=0.6,
        osram_bidirectional=False,
        osram_meaningful_block='learned_robust_pca_evidence_decomposition',
        osram_rpca_rank=2, osram_rpca_steps=14,
        osram_rpca_threshold_init=0.33599569865951234,
        osram_rpca_lr_multiplier=2.8200380294401013, device='cuda:0')
    if flat_only:
        for name in ('osram_rpca_rank', 'osram_rpca_steps', 'osram_rpca_threshold_init'):
            expected.pop(name)
        expected.update(osram_meaningful_block='none', osram_rpca_lr_multiplier=1.0)
        allowed_seeds = (67, 68) if flat_replication else (66,)
        if config['seed'] not in allowed_seeds:
            raise ValueError(f'Flat seeds authorized for this entry: {allowed_seeds}')
    for key, value in expected.items():
        original_default = False if key == 'osram_disable_local_skip' else None
        if config.get(key, original_default) != value:
            raise ValueError(f'Unexpected {key}: {config.get(key)!r}, expected {value!r}')
    if config['dataset'] not in LABEL_HASHES or config['seed'] not in (66, 67, 68) or config['fold'] not in range(1, 6):
        raise ValueError('Only authorized IEMOCAP4/6 × seeds66–68 × folds1–5')
    if config['train_missing_rates'] != [i / 10 for i in range(8)]:
        raise ValueError('Original eight-rate cyclic schedule required')


def smoke(config, trainer, roots, torch):
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    loaders = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=5, dataset=config.dataset, batch_size=config.batch_size, num_workers=0,
        seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    train, _, _, ad, td, vd = loaders
    loader = train[config.fold - 1]
    # Cover the longest real conversations, not a conveniently short first batch.
    indices = sorted(loader.sampler.indices,
        key=lambda i: len(loader.dataset.videoLabelsNew[loader.dataset.vids[i]]), reverse=True)[:config.batch_size]
    raw = loader.collate_fn([loader.dataset[i] for i in indices])
    device = torch.device('cuda:0')
    model = _build_model(config, (ad, td, vd), device).train().requires_grad_(True)
    if model.teacher is not None:
        model.teacher.eval().requires_grad_(False)
    data = trainer._move_batch(raw, device)
    groups, group_provenance = trainer._optimizer_parameter_groups(model, config)
    optimizer = torch.optim.Adam(groups, weight_decay=config.weight_decay)
    core = model.osram.meaningful_block
    before = {} if core is None else {n: p.detach().clone() for n, p in core.named_parameters()}
    reports = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step, rate in enumerate((0.7, 0.7, 0.0)):
        schedule = GCNetLegacyRandomMaskSchedule(config.dataset, 'train', config.fold, rate, config.seed)
        optimizer.zero_grad(set_to_none=True)
        view = trainer._prepare_view(data, schedule, epoch=step, dimensions=(ad, td, vd))
        prediction = model([view['incomplete']], view['availability'], view['qmask'],
                           view['umask'], view['lengths'], predict_missing=False)[0]
        loss = trainer._task_loss(config.dataset, prediction, view['labels'], view['umask'])
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Nonfinite original task loss/gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm))
        reports.append(dict(step=step+1, rate=rate, CE=float(loss.detach()), gradient_norm=norm,
            block_nonzero_parameters=0 if core is None else sum(
                p.grad is not None and bool(p.grad.ne(0).any()) for p in core.parameters())))
        optimizer.step()
    torch.cuda.synchronize()
    changed = 0 if core is None else sum(not torch.equal(before[n], p) for n, p in core.named_parameters())
    if core is not None and (not changed or not any(r['block_nonzero_parameters'] for r in reports)):
        raise RuntimeError('B6 path did not actually learn')
    if len(torch.cuda.get_rng_state_all()) != 1:
        raise RuntimeError('Recovery would omit CUDA RNG')
    return dict(status='complete', smoke_only=True, steps=reports, block_updated_parameters=changed,
        parameters=sum(p.numel() for p in model.parameters()),
        block_parameters=0 if core is None else sum(p.numel() for p in core.parameters()),
        padded_length=int(raw[7].shape[1]), batch_conversations=len(indices),
        optimizer_parameter_groups=group_provenance,
        peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        seconds=time.monotonic() - started, cuda_rng_states=len(torch.cuda.get_rng_state_all()))


def selected_results(output, torch, classes):
    rates = {}
    for i in range(8):
        checkpoint_path = output / f'best_miss_0p{i}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        reference = checkpoint['selection_artifact']
        selected_path = (output / reference['path']).resolve()
        if not selected_path.is_relative_to(output.resolve()) or sha(selected_path) != reference['sha256']:
            raise ValueError('Selected artifact path/hash mismatch')
        with np.load(selected_path, allow_pickle=False) as selected:
            scores = class_metrics(selected['labels'], selected['logits'], classes)
            if not np.array_equal(selected['predictions'], selected['logits'].argmax(1)):
                raise ValueError('Stored classes disagree with selected logits')
            if int(selected['selected_epoch']) != checkpoint['epoch']:
                raise ValueError('Selected epoch mismatch')
            if abs(scores['ACC'] / 100 - checkpoint['selection_accuracy']) > 1e-8:
                raise ValueError('ACC and UA must come from the SAME selected evaluation')
            mask_hash = hashlib.sha256(selected['availability'].tobytes()).hexdigest()
            if mask_hash != reference['availability_sha256']:
                raise ValueError('Selected availability hash mismatch')
        rates[str(i / 10)] = dict(scores, epoch=checkpoint['epoch'],
            checkpoint_sha256=sha(checkpoint_path), selection_sha256=reference['sha256'],
            mask_sha256=mask_hash)
    return aggregate(rates)


def verify_first_epoch_retention(output, torch, classes, *, cuda_rng_states):
    """Read the ALREADY saved first-epoch artifacts; no model or sampler call."""
    if cuda_rng_states != 1:
        raise ValueError('A complete UUID-bound CUDA RNG state is required for recovery')
    result = selected_results(output, torch, classes)
    if any(row['epoch'] != 1 for row in result['rates'].values()):
        raise ValueError('First-epoch retention check must verify epoch1 artifacts')
    report = dict(status='complete', epoch=1, best_checkpoints=8, selected_artifacts=8,
                  cuda_rng_states=cuda_rng_states, testbest=result,
                  checks='weight/artifact hashes, mask hash, epoch, ACC, UA, logits and classes')
    write(Path(output) / 'FIRST_EPOCH_RETENTION_CHECK.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--flat-only', action='store_true',
                        help='Authorized seed66 original Flat restoration; no B6 module')
    parser.add_argument('--flat-replication', action='store_true',
                        help='Explicitly authorized Flat seeds67/68 in the accepted cmerc-repro runtime')
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text())
    validate(raw_config, flat_only=args.flat_only, flat_replication=args.flat_replication)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['GCNET_DATASET_ROOT'] = str(DATA_ROOT)
    os.environ['GCNET_CACHE_ROOT'] = str(args.output.parent / '_iemocap_feature_cache')
    os.environ.pop('GCNET_MELD_PICKLE', None)
    source = args.source_root.resolve()
    actual_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if actual_sha != SOURCE_SHA:
        raise ValueError('Immutable numerical source HEAD mismatch')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--',
        'gcnet_missing_m3', 'gcnet_modality_jepa'], cwd=source, text=True).strip():
        raise ValueError('Numerical source differs from committed snapshot')
    sys.path.insert(0, str(source))
    import torch
    replication_environment = require_replication_environment(torch) if args.flat_replication else None
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    gpu = guard_gpu(args.gpu_uuid)
    if gpu['host_index'] not in (1, 3):
        raise ValueError('Only authorized physical GPUs1/3; GPU5/4 excluded')
    repair = repair_uuid_rng(torch)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = trainer.TrainConfig(**raw_config)
    label_name, label_hash = LABEL_HASHES[config.dataset]
    label_path = DATA_ROOT / 'IEMOCAP' / label_name
    if sha(label_path) != label_hash:
        raise ValueError('IEMOCAP label/data version changed')
    roots = [str(DATA_ROOT / 'IEMOCAP' / 'features' / name)
             for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    if not all(Path(p).is_dir() for p in roots):
        raise FileNotFoundError('Original IEMOCAP feature root unavailable')
    hashes = {str(p.relative_to(source)): sha(p) for folder in ('gcnet_missing_m3', 'gcnet_modality_jepa')
              for p in sorted((source / folder).glob('*.py'))}
    identity = dict(source_git_sha=actual_sha, code_hashes=hashes, worker_sha256=sha(__file__),
                    label_sha256=label_hash, config_sha256=sha(args.config),
                    uuid_rng_patch='enumeration-and-cuda.random-alias-v1')
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / 'SUMMARY.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous['identity'] != identity or previous.get('smoke_only', False) != args.smoke:
            raise ValueError('Refuse mixed source/data/config/smoke in one output')
        if previous['status'] == 'complete':
            print('Already complete; not training twice', flush=True)
            return
    summary = dict(status='running', smoke_only=args.smoke, config=raw_config,
        effective_config=asdict(config), dataset=config.dataset, seed=config.seed, fold=config.fold, identity=identity,
        method='Original_Flat' if args.flat_only else 'B6_tuned_pca_only',
        protocol='INTERNAL TEST-ORACLE SCREEN; per-rate selected ACC',
        provenance=dict(hostname=socket.gethostname(), gpu=gpu, uuid_rng=repair, pid=os.getpid(),
            feature_roots=roots, environment=(replication_environment if args.flat_replication
                else dict(torch=torch.__version__, numpy=np.__version__)),
            **({'replication_environment': replication_environment} if args.flat_replication else {}),
            command=sys.argv, no_meld_training_hyperparameters=True), started_unix=time.time())
    write(summary_path, summary)
    started = time.monotonic()
    try:
        if args.smoke:
            summary.update(smoke(config, trainer, roots, torch))
        else:
            class CompactState(TrainingState):
                def commit_epoch(self, **kwargs):
                    super().commit_epoch(**kwargs)
                    if args.flat_only and kwargs['next_epoch'] == 1:
                        if getattr(self.model.osram, 'meaningful_block', None) is not None:
                            raise RuntimeError('Flat restoration must not instantiate a Block2')
                        verify_first_epoch_retention(args.output, torch,
                            4 if config.dataset == 'IEMOCAPFour' else 6,
                            cuda_rng_states=len(torch.cuda.get_rng_state_all()))
                    keep = {self._checked(ref[k]).resolve() for ref in self.references.values() for k in ('model', 'selection')}
                    for path in (self.output / 'training_versions').iterdir():
                        if path.is_file() and path.resolve() not in keep:
                            path.unlink()  # Only THIS run's superseded temporary recovery cache.
            state = CompactState(args.output, identity=identity)
            torch.cuda.reset_peak_memory_stats()
            trainer.run_experiment(config, *roots, str(args.output), training_state=state)
            summary.update(status='complete', testbest=selected_results(args.output, torch,
                4 if config.dataset == 'IEMOCAPFour' else 6),
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
        summary['finished_unix'] = time.time()
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary['total_seconds'] = time.monotonic() - started
        write(summary_path, summary)


if __name__ == '__main__':
    main()
