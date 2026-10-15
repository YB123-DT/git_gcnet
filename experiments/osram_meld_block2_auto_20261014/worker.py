"""One real MELD training/replay job. INTERNAL TEST-ORACLE SCREEN."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
DATASET_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
COMMON = Path('/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0')
# config.PATH_TO_LABEL is materialized when train_gcnet is imported. These
# paths must be fixed BEFORE that import, not merely before get_loaders().
os.environ.update(GCNET_DATASET_ROOT='/data2/yb/paper/remote_runs_20261012/data',
                  GCNET_MELD_PICKLE=str(DATASET))

from .methods import METHODS, validate_training_config
from gcnet_missing_m3.training_resume import TrainingState, _sha
from gcnet_missing_m3 import train_gcnet as trainer
from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model, _write
from experiments.osram_meld_write_retention_20261014.run import guard_gpu
from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    scatter_saved_availability, replay_view_from_availability,
)

def source_hashes():
    files = list((ROOT / 'gcnet_missing_m3').glob('*.py'))
    files += list((ROOT / 'gcnet_modality_jepa').glob('*.py'))
    files += list(Path(__file__).parent.glob('*.py'))
    return {str(p.relative_to(ROOT)): _sha(p) for p in sorted(files)}


def repair_uuid_enumeration():
    """Process-local Torch1.13 compatibility: admitted UUID is one real device.

    NVML can falsely return0devices despite CUDA runtime seeing1. Correct the
    count before seed_all/TrainingState; do not change UUID or host-card binding.
    All jobs (including Flat) share this shim and record it in provenance.
    """
    runtime_count = torch._C._cuda_getDeviceCount()
    python_count = torch.cuda.device_count()
    if runtime_count != 1:
        raise RuntimeError('Require exactly one admitted CUDA runtime device')
    if python_count != runtime_count:
        torch.cuda.device_count = lambda: runtime_count
    return dict(cuda_runtime_count=runtime_count, original_python_count=python_count,
                uuid_enumeration_repaired=python_count != runtime_count)


class CompactState(TrainingState):
    """Retain active BEST references, not an unbounded epoch-model archive."""
    def commit_epoch(self, **kwargs):
        super().commit_epoch(**kwargs)
        keep = {self._checked(ref[key]).resolve() for ref in self.references.values()
                for key in ('model', 'selection')}
        versions = self.output / 'training_versions'
        if versions.exists():
            for path in versions.iterdir():
                if path.is_file() and path.resolve() not in keep:
                    path.unlink()  # only this NEW run's unreferenced recovery cache


def class_metrics(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if len(labels) == 0 or logits.shape != (len(labels), 7) or not np.isfinite(logits).all():
        raise ValueError('Require aligned finite seven-class outputs')
    predictions = logits.argmax(1)
    recalls = recall_score(labels, predictions, labels=np.arange(7), average=None, zero_division=0)
    scores = logits - logits.max(1, keepdims=True)
    ce = np.log(np.exp(scores).sum(1)) - scores[np.arange(len(labels)), labels]
    return dict(ACC=100. * accuracy_score(labels, predictions), UA=100. * float(recalls.mean()),
                WF1=100. * f1_score(labels, predictions, average='weighted', zero_division=0),
                CE=float(ce.mean()), recall=(100. * recalls).tolist(),
                support=np.bincount(labels, minlength=7).tolist(), N=len(labels))


def aggregate(rates):
    return dict(rates=rates,
                mean={key: float(np.mean([v[key] for v in rates.values()])) for key in ('ACC', 'UA', 'WF1', 'CE')},
                high={key: float(np.mean([rates[str(i / 10)][key] for i in (5, 6, 7)])) for key in ('ACC', 'UA', 'WF1', 'CE')})


@torch.no_grad()
def original_flat_logits(model, local, base, gap, availability, umask):
    """Bypass ONLY Block2 using inputs captured before it; no second scan."""
    valid = umask.T.bool()
    local = torch.where(valid[..., None], local, 0.).contiguous()
    base = torch.where(valid[..., None], base, 0.)
    gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
    inputs = torch.cat((local, base, gap.flatten(2)), -1)
    osram = model.osram
    hidden = osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(inputs))
    hidden = torch.where(valid[..., None], hidden, 0.)
    return model.smax_fc(hidden)


def load_data(config):
    from gcnet_modality_jepa.train_gcnet import get_loaders
    train, _, test, ad, td, vd = get_loaders(audio_root='MELD-pickle', text_root='MELD-pickle',
        video_root='MELD-pickle', num_folder=1, dataset='MELD', batch_size=config.batch_size,
        num_workers=0, seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    return train[0], test[0], (ad, td, vd)


@torch.no_grad()
def paired_replay(model, loader, dimensions, saved, method):
    """One CHM forward per batch, on/off Flat heads share its exact raw reads."""
    model.eval()
    device = next(model.parameters()).device
    cursor = 0
    parts = {k: [] for k in ('on', 'off', 'labels', 'conversation_ids', 'utterance_indices', 'speakers', 'availability')}
    for raw in loader:
        data = trainer._move_batch(raw, device)
        availability, cursor = scatter_saved_availability(saved['availability'], cursor, data[7])
        view = replay_view_from_availability(data, availability, dimensions)
        captured = []
        hook = None
        if method != 'Flat':
            hook = model.osram.meaningful_block.register_forward_pre_hook(lambda module, args: captured.append(args))
        try:
            on = model([view['incomplete']], availability, view['qmask'], view['umask'],
                       view['lengths'], predict_missing=False)[0]
        finally:
            if hook is not None:
                hook.remove()
        if method == 'Flat':
            off = on
        else:
            if len(captured) != 1:
                raise RuntimeError('Block2 route must execute exactly once')
            off = original_flat_logits(model, *captured[0])
        valid = view['umask'].bool()
        labels = view['labels'][valid].cpu().numpy()
        begin = cursor - len(labels)
        np.testing.assert_array_equal(labels, saved['labels'][begin:cursor])
        ids = np.asarray([str(v) for v in view['conversation_ids']])
        for name, tensor in (('on', on), ('off', off)):
            parts[name].append(tensor.transpose(0, 1)[valid].cpu().numpy())
        parts['labels'].append(labels)
        parts['conversation_ids'].append(np.broadcast_to(ids[:, None], valid.shape)[valid.cpu().numpy()])
        parts['utterance_indices'].append(np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid.cpu().numpy()])
        parts['speakers'].append(view['qmask'][valid].cpu().numpy())
        parts['availability'].append(availability.transpose(0, 1)[valid].cpu().numpy())
    if cursor != len(saved['labels']):
        raise RuntimeError('Common-mask replay did not consume all labels')
    result = {k: np.concatenate(v) for k, v in parts.items()}
    np.testing.assert_array_equal(result['availability'], saved['availability'])
    return result


def build_smoke_model(config, dimensions, device):
    # The reused replay constructor intentionally freezes EVERY parameter.
    # Restore normal student training for smoke only, never unfreeze EMA targets.
    model = _build_model(config, dimensions, device).train().requires_grad_(True)
    if model.teacher is not None:
        model.teacher.eval().requires_grad_(False)
    return model


def smoke(config, output, method):
    train, test, dimensions = load_data(config)
    trainer.set_random_seed(config.seed)
    model = build_smoke_model(config, dimensions, torch.device('cuda:0'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
    schedule = GCNetLegacyRandomMaskSchedule('MELD', 'train', 1, .7, config.seed)
    # Worst-length REAL training conversations cover the padded33-step memory
    # and maximum active head-token workload, rather than a small first batch.
    longest = sorted(train.sampler.indices,
        key=lambda i: len(train.dataset.videoLabelsNew[train.dataset.vids[i]]), reverse=True)[:config.batch_size]
    raw = train.collate_fn([train.dataset[i] for i in longest])
    data = trainer._move_batch(raw, torch.device('cuda:0'))
    core = model.osram.meaningful_block if method != 'Flat' else None
    before = None if core is None else {name: p.detach().clone() for name, p in core.named_parameters()}
    reports = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        view = trainer._prepare_view(data, schedule, epoch=step, dimensions=dimensions)
        logits = model([view['incomplete']], view['availability'], view['qmask'], view['umask'],
                       view['lengths'], predict_missing=False)[0]
        loss = trainer._task_loss('MELD', logits, view['labels'], view['umask'])
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads) or not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke gradient/loss')
        norm = float(torch.nn.utils.clip_grad_norm_(params, config.gradient_clip_norm))
        block = [] if core is None else [p.grad for p in core.parameters() if p.grad is not None]
        reports.append(dict(step=step + 1, CE=float(loss.detach()), gradient_norm=norm,
                            block_nonzero_parameters=sum(int(g.ne(0).any()) for g in block)))
        optimizer.step()
    torch.cuda.synchronize()
    changed = 0 if core is None else sum(not torch.equal(before[name], p) for name, p in core.named_parameters())
    if core is not None and (not changed or not any(v['block_nonzero_parameters'] for v in reports)):
        raise RuntimeError('Block2 did not receive actual task gradients/update after three steps')
    return dict(status='complete', method=method, smoke_only=True, steps=reports,
                smoke_conversation_ids=[str(v) for v in raw[-1]], smoke_padded_length=int(raw[7].shape[1]),
                parameters=sum(p.numel() for p in model.parameters()),
                block_parameters=0 if core is None else sum(p.numel() for p in core.parameters()),
                block_updated_parameters=changed, seconds=time.monotonic() - started,
                peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                note='GPU smoke uses .7 to activate Gaps; this is not a trained performance result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replication-seed', type=int, choices=(67, 68), default=None)
    parser.add_argument('--testbest-only', action='store_true')
    args = parser.parse_args()
    if args.testbest_only and args.replication_seed is None:
        raise ValueError('Test-best-only is an explicit replication mode, not a screen/HPO shortcut')
    gpu = guard_gpu(args.gpu_uuid)
    repair = repair_uuid_enumeration()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    raw_config = json.loads(args.config.read_text())
    config = trainer.TrainConfig(**raw_config)
    candidates = [k for k, v in METHODS.items() if v['block'] == config.osram_meaningful_block]
    if len(candidates) != 1:
        raise ValueError('Exactly one of the seven pre-registered variants required')
    method = candidates[0]
    if _sha(DATASET) != DATASET_SHA:
        raise ValueError('MELD feature version mismatch')
    os.environ.update(GCNET_DATASET_ROOT='/data2/yb/paper/remote_runs_20261012/data',
                      GCNET_MELD_PICKLE=str(DATASET), GCNET_CACHE_ROOT=str(args.output.parent / '_feature_cache'))
    common_hashes = {str(i / 10): _sha(COMMON / f'predictions_miss_0p{i}.npz') for i in range(8)}
    inventory = json.loads((ROOT / 'experiments/osram_meld_query_decision_20261014/CANONICAL_INPUTS.json').read_text())
    if common_hashes != inventory['mask_sha256']:
        raise ValueError('Protected common masks changed')
    provenance = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_hashes=source_hashes(), dataset_sha256=DATASET_SHA, mask_protocol=config.missing_mask_protocol,
        common_mask_hashes=common_hashes, seed=config.seed, hostname=socket.gethostname(), gpu=gpu,
        environment=dict(torch=torch.__version__, numpy=np.__version__), uuid_runtime=repair,
        command=os.sys.argv, selection='INTERNAL TEST-ORACLE SCREEN; per-rate ACC',
        config_sha256=hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest())
    if config.missing_mask_protocol != 'gcnet-random-mask-v1':
        raise ValueError('Original GCNet missing protocol is mandatory')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(status='running', method=method, name=METHODS[method]['name'], config=asdict(config),
                   provenance=provenance, started_unix=time.time(), pid=os.getpid())
    previous = args.output / 'SUMMARY.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if any(old['provenance'].get(k) != provenance.get(k) for k in
               ('git_sha', 'code_hashes', 'dataset_sha256', 'config_sha256')):
            raise ValueError('Refuse to mix code/data/config versions in one run')
        if old.get('smoke_only', False) != args.smoke:
            raise ValueError('Smoke and formal runs must use separate output directories')
        if old['status'] == 'complete':
            print('Already complete; no duplicate training', flush=True)
            return
    _write(previous, summary)
    started = time.monotonic()
    try:
        if args.smoke:
            summary.update(smoke(config, args.output, method))
            return
        validate_training_config(config, args.replication_seed)
        identity = {k: provenance[k] for k in ('git_sha', 'code_hashes', 'dataset_sha256', 'config_sha256')}
        state = CompactState(args.output, identity=identity)
        torch.cuda.reset_peak_memory_stats()
        training_start = time.monotonic()
        trainer.run_experiment(config, 'MELD-pickle', 'MELD-pickle', 'MELD-pickle',
                               str(args.output), training_state=state)
        train_seconds = time.monotonic() - training_start
        _, test, dimensions = load_data(config)
        model = _build_model(config, dimensions, torch.device('cuda:0')).eval()
        best, on, off, changes = {}, {}, {}, {}
        id_digest = hashlib.sha256()
        for i in range(8):
            rate = str(i / 10)
            checkpoint_path = args.output / f'best_miss_0p{i}.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            ref = checkpoint['selection_artifact']
            selected_path = args.output / ref['path']
            if _sha(selected_path) != ref['sha256']:
                raise ValueError('Selected predictions do not match checkpoint reference')
            with np.load(selected_path) as selected:
                best[rate] = dict(class_metrics(selected['labels'], selected['logits']),
                    epoch=checkpoint['epoch'], mask_sha256=ref['availability_sha256'],
                    checkpoint_sha256=_sha(checkpoint_path), selected_file_sha256=ref['sha256'])
                if abs(best[rate]['ACC'] / 100. - checkpoint['selection_accuracy']) > 1e-8:
                    raise ValueError('Selected ACC not derived from same saved predictions')
                shutil.copyfile(selected_path, args.output / f'selected_predictions_miss_0p{i}.npz')
                selection_ids = selected['conversation_ids'].copy()
                selection_times = selected['utterance_indices'].copy()
                selection_speakers = selected['speakers'].copy()
            if args.testbest_only:
                continue
            with np.load(COMMON / f'predictions_miss_0p{i}.npz') as saved:
                replay = paired_replay(model, test, dimensions, saved, method)
            np.testing.assert_array_equal(replay['conversation_ids'], selection_ids)
            np.testing.assert_array_equal(replay['utterance_indices'], selection_times)
            np.testing.assert_array_equal(replay['speakers'], selection_speakers)
            id_digest.update(replay['conversation_ids'].tobytes())
            id_digest.update(replay['utterance_indices'].tobytes())
            id_digest.update(replay['speakers'].tobytes())
            for container, name in ((on, 'on'), (off, 'off')):
                container[rate] = dict(class_metrics(replay['labels'], replay[name]),
                    epoch=checkpoint['epoch'], mask_sha256=common_hashes[rate])
            a, b, y = replay['off'].argmax(1), replay['on'].argmax(1), replay['labels']
            changes[rate] = dict(N=len(y), corrections=int(((a != y) & (b == y)).sum()),
                harms=int(((a == y) & (b != y)).sum()), both_correct=int(((a == y) & (b == y)).sum()),
                both_wrong=int(((a != y) & (b != y)).sum()))
            np.savez_compressed(args.output / f'paired_miss_0p{i}.npz', **replay)
        summary.update(status='complete', training_seconds=train_seconds,
            totaltrainsec=train_seconds,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peakmemorybytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
            block_parameters=0 if method == 'Flat' else sum(p.numel() for p in model.osram.meaningful_block.parameters()),
            testbest=aggregate(best), paired=(dict(status='not_requested', reason='Test-best-only replication')
                if args.testbest_only else dict(on=aggregate(on), off=aggregate(off), changes=changes,
                mask_digest=hashlib.sha256(json.dumps(common_hashes, sort_keys=True).encode()).hexdigest(),
                id_digest=id_digest.hexdigest())), resume='full epoch-boundary optimizer/RNG; immutable BEST artifacts',
            finished_unix=time.time())
        _write(previous, summary)
        # Completed HPO jobs need no restart cache. Preserve every BEST weight
        # and exact artifact; release ONLY this run's temporary recovery archive.
        if 'hpo' in args.output.parts or 'optuna' in args.output.parts:
            try:
                versions = args.output / 'training_versions'
                if versions.is_dir():
                    for path in versions.iterdir():
                        if path.is_file():
                            path.unlink()
                    versions.rmdir()
                recovery = args.output / 'last_training.pt'
                if recovery.exists():
                    recovery.unlink()
                summary['completed_hpo_recovery_cache_released'] = True
            except OSError as cleanup_error:
                # A successful measured run must not become a failed trial
                # solely because releasing its temporary cache failed.
                summary['cache_release_warning'] = repr(cleanup_error)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), finished_unix=time.time())
        raise
    finally:
        summary['seconds'] = time.monotonic() - started
        _write(previous, summary)


if __name__ == '__main__':
    main()
