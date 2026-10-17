"""Original seed66 Flat training with alternate, canonically aligned MELD features.

The numerical trainer/model are imported exclusively from the immutable 737fdb1
snapshot. This launcher changes data paths only; reported metrics use the saved
selection-epoch logits and masks, without another stochastic test forward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import time

SOURCE_SHA = '737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c'
SOURCE_INVENTORY_SHA256 = '53ac268a20baa3d3a120be5dd40787776dc452c5ab2d5dc8a3baefded7321deb'
SOURCE_PACKAGES = ('gcnet_missing_m3', 'gcnet_modality_jepa',
                   'gcnet_jepa_replacement', 'gcnet_plci_jepa')
METHODS_PATH = 'experiments/osram_meld_block2_auto_20261014/methods.py'
BASE_PATH = 'experiments/osram_meld_block2_auto_20261014/BASE_CONFIG.json'
MODEL_BUILDER_PATH = 'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_source(root):
    """Pin all 131 Python/config inputs, even for snapshots without a .git dir."""
    paths = [path for package in SOURCE_PACKAGES for path in (root / package).glob('*.py')]
    paths += [root / name for name in ('config.py', METHODS_PATH, BASE_PATH, MODEL_BUILDER_PATH)]
    if len(paths) != 131 or any(not path.is_file() for path in paths):
        raise ValueError('Frozen source inventory is incomplete')
    hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}
    if json_digest(hashes) != SOURCE_INVENTORY_SHA256:
        raise ValueError('Frozen source differs from required commit ' + SOURCE_SHA)
    return hashes


def validate_config(raw, expected):
    if json_digest(raw) != json_digest(expected):
        changed = sorted(key for key in set(raw) | set(expected) if raw.get(key) != expected.get(key))
        raise ValueError('Require exact original Flat configuration; differing keys: ' + str(changed))


def configure_environment(data, output, gpu_uuid):
    """Run before importing Torch or any module that materializes config paths."""
    os.environ.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid, GCNET_MELD_PICKLE=str(data.resolve()),
        GCNET_DATASET_ROOT=str(data.resolve().parent),
        GCNET_CACHE_ROOT=str((output / '_feature_cache').resolve()),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', PYTHONHASHSEED='66',
    )


def guard_gpu(uuid):
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the explicit GPU UUID')
    listing = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits',
    ], text=True)
    matches = [line.split(',') for line in listing.splitlines() if line.split(',')[-1].strip() == uuid]
    if len(matches) != 1 or matches[0][0].strip() == '4':
        raise ValueError('GPU UUID is not unique or identifies forbidden host GPU 4')
    return dict(host_index=int(matches[0][0]), uuid=uuid, logical_device='cuda:0')


def repair_uuid_enumeration(torch):
    runtime_count, python_count = torch._C._cuda_getDeviceCount(), torch.cuda.device_count()
    if runtime_count != 1:
        raise RuntimeError('Require exactly one admitted CUDA runtime device')
    if python_count != runtime_count:
        torch.cuda.device_count = lambda: runtime_count
    return dict(cuda_runtime_count=runtime_count, original_python_count=python_count,
                uuid_enumeration_repaired=python_count != runtime_count)


def require_cuda_rng(state):
    rng = state.get('rng')
    if not isinstance(rng, dict) or not rng.get('cuda') or len(rng['cuda']) != 1:
        raise ValueError('Complete single-device CUDA RNG state is mandatory for resume')


def compact_state_class(training_state, torch):
    class CompactState(training_state):
        def restore(self):
            path = self.output / 'last_training.pt'
            if path.exists():
                require_cuda_rng(torch.load(path, map_location='cpu', weights_only=False))
            return super().restore()

        def commit_epoch(self, **kwargs):
            if not torch.cuda.is_initialized() or len(torch.cuda.get_rng_state_all()) != 1:
                raise ValueError('Require initialized single-device CUDA RNG before commit')
            super().commit_epoch(**kwargs)
            keep = {self._checked(ref[key]).resolve() for ref in self.references.values()
                    for key in ('model', 'selection')}
            versions = self.output / 'training_versions'
            if versions.exists():
                for path in versions.iterdir():
                    if path.is_file() and path.resolve() not in keep:
                        path.unlink()
    return CompactState


def class_metrics(labels, logits):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels, logits = np.asarray(labels), np.asarray(logits)
    if (labels.ndim != 1 or len(labels) == 0 or logits.shape != (len(labels), 7)
            or not np.isfinite(logits).all() or labels.dtype.kind not in 'iu'
            or (labels < 0).any() or (labels > 6).any()):
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
                mean={key: sum(v[key] for v in rates.values()) / 8 for key in ('ACC', 'UA', 'WF1', 'CE')},
                high={key: sum(rates[str(i / 10)][key] for i in (5, 6, 7)) / 3
                      for key in ('ACC', 'UA', 'WF1', 'CE')})


def selected_metrics(output, torch):
    import numpy as np
    rates = {}
    for index in range(8):
        path = output / f'best_miss_0p{index}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        ref = checkpoint['selection_artifact']
        selected = (output / ref['path']).resolve()
        if not selected.is_relative_to(output.resolve()) or sha256(selected) != ref['sha256']:
            raise ValueError('Selected prediction artifact integrity mismatch')
        with np.load(selected) as saved:
            if int(saved['selected_epoch']) != checkpoint['epoch']:
                raise ValueError('Selection epoch does not match checkpoint')
            if checkpoint['selection_rate'] != index / 10:
                raise ValueError('Selection rate does not match checkpoint filename')
            result = class_metrics(saved['labels'], saved['logits'])
            if abs(result['ACC'] / 100. - checkpoint['selection_accuracy']) > 1e-8:
                raise ValueError('Checkpoint ACC does not match same-selection saved logits')
            if hashlib.sha256(saved['availability'].tobytes()).hexdigest() != ref['availability_sha256']:
                raise ValueError('Selection mask digest mismatch')
        destination = output / f'selected_predictions_miss_0p{index}.npz'
        shutil.copyfile(selected, destination)
        rates[str(index / 10)] = dict(result, epoch=checkpoint['epoch'],
            mask_sha256=ref['availability_sha256'], checkpoint_sha256=sha256(path),
            selected_file_sha256=ref['sha256'])
    return aggregate(rates)


def assert_data_binding(config_module, data):
    if Path(config_module.PATH_TO_LABEL['MELD']).resolve() != data.resolve():
        raise ValueError('Effective MELD DATAFILE differs from requested alternate data')


def smoke(config, loader, dimensions, source_root, trainer, torch):
    """Three real forward/backward batches; no test iteration or selection."""
    build_model = runpy.run_path(str(source_root / MODEL_BUILDER_PATH))['_build_model']
    model = build_model(config, dimensions, torch.device('cuda:0')).train().requires_grad_(True)
    if model.teacher is not None:
        model.teacher.eval().requires_grad_(False)
    groups, group_info = trainer._optimizer_parameter_groups(model, config)
    optimizer = torch.optim.Adam(groups, weight_decay=config.weight_decay)
    from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
    schedule = GCNetLegacyRandomMaskSchedule('MELD', 'train', 1, .7, config.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = model.smax_fc.weight.detach().clone()
    reports = []
    for index, raw in enumerate(loader):
        if index == 3:
            break
        optimizer.zero_grad(set_to_none=True)
        data = trainer._move_batch(raw, torch.device('cuda:0'))
        view = trainer._prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        logits = model([view['incomplete']], view['availability'], view['qmask'], view['umask'],
                       view['lengths'], predict_missing=False)[0]
        loss = trainer._task_loss('MELD', logits, view['labels'], view['umask'])
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite or absent smoke loss/gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()
        reports.append(dict(step=index + 1, CE=float(loss.detach()), gradient_norm=float(norm)))
    if len(reports) != 3 or torch.equal(before, model.smax_fc.weight):
        raise ValueError('Smoke requires three actual classifier updates')
    torch.cuda.synchronize()
    return dict(status='complete', smoke_only=True, steps=reports, optimizer_parameter_groups=group_info,
                parameters=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--data-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root, data, output = args.source_root.resolve(), args.data.resolve(), args.output.resolve()
    source_hashes = validate_source(root)
    expected = runpy.run_path(str(root / METHODS_PATH))['build_config'](
        json.loads((root / BASE_PATH).read_text()), 'Flat', output)
    raw_config = json.loads(args.source_config.read_text())
    validate_config(raw_config, expected)
    if sha256(data) != args.data_sha256:
        raise ValueError('Alternate MELD data SHA256 mismatch')
    configure_environment(data, output, args.gpu_uuid)
    gpu = guard_gpu(args.gpu_uuid)
    sys.path.insert(0, str(root))
    import torch
    import numpy as np
    config_module = importlib.import_module('config')
    trainer = importlib.import_module('gcnet_missing_m3.train_gcnet')
    resume_module = importlib.import_module('gcnet_missing_m3.training_resume')
    for module in (config_module, trainer, resume_module):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Imported numerical source escapes immutable source root')
    assert_data_binding(config_module, data)
    repair = repair_uuid_enumeration(torch)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = trainer.TrainConfig(**raw_config)
    # This is data construction only; trainer resets all seeds before its loaders.
    loaders = trainer.get_loaders(audio_root='MELD-pickle', text_root='MELD-pickle',
        video_root='MELD-pickle', num_folder=1, dataset='MELD', batch_size=config.batch_size,
        num_workers=0, seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    dimensions = tuple(int(value) for value in loaders[3:])
    identity = dict(source_git_sha=SOURCE_SHA, source_hashes=source_hashes,
                    worker_sha256=sha256(Path(__file__)), dataset_sha256=args.data_sha256,
                    config_sha256=json_digest(raw_config), smoke_only=args.smoke)
    provenance = dict(identity, source_root=str(root), datafile=str(data), gpu=gpu,
        hostname=socket.gethostname(), command=sys.argv, seed=66, uuid_runtime=repair,
        dimensions=dict(audio=dimensions[0], text=dimensions[1], visual=dimensions[2]),
        environment=dict(python=sys.version, executable=sys.executable,
                         torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda),
        selection='Per-rate test-best ACC; UA/WF1 from identical saved selection-epoch logits/masks')
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'SUMMARY.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Refuse mixed source/config/data or smoke/formal resume')
        if previous['status'] == 'complete':
            print('Already complete; no duplicate training', flush=True)
            return
    elif any(output.glob('best_miss_*.pt')) or (output / 'last_training.pt').exists():
        raise ValueError('Existing training artifacts lack worker provenance; use new output')
    summary = dict(status='running', method='Flat', identity=identity, provenance=provenance,
                   config=asdict(config), smoke_only=args.smoke, pid=os.getpid(), started_unix=time.time())
    write_json(summary_path, summary)
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        if args.smoke:
            trainer.set_random_seed(config.seed)
            summary.update(smoke(config, loaders[0][0], dimensions, root, trainer, torch))
        else:
            state = compact_state_class(resume_module.TrainingState, torch)(output, identity=identity)
            assert_data_binding(config_module, data)
            training_started = time.monotonic()
            trainer.run_experiment(config, 'MELD-pickle', 'MELD-pickle', 'MELD-pickle',
                                   str(output), training_state=state)
            summary['training_seconds'] = time.monotonic() - training_started
            summary.update(status='complete', testbest=selected_metrics(output, torch),
                           parameters=sum(p.numel() for p in state.model.parameters()),
                           block_parameters=0,
                           resume='Full epoch-boundary model/optimizer/schedule/RNG with immutable BEST artifacts')
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary.update(seconds=time.monotonic() - started, finished_unix=time.time(),
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(summary_path, summary)


if __name__ == '__main__':
    main()
