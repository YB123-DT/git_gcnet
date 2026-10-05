"""One source-pinned C01–C20 screening run (no test-driven hyperparameter search)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

from experiments.osram_meaningful20_20261003.run import EXPECTED, FORBIDDEN
from gcnet_missing_m3.core20 import METHODS, CONTROLS

LABEL = 'INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle screening; NOT A FORMAL PAPER RESULT'
# Fixed BEFORE training. These are transfer coefficients, not published optimal
# recipes. Raw-feature generative ELBO sums 2560 coordinates (C11); .001 keeps
# its explicit likelihood objective from being silently averaged into a new loss.
AUX_WEIGHTS = {'C02': .1, 'C03': 1., 'C11': .001, 'C14': 1., 'C15': .1, 'C16': .1, 'C19': 1.}
CHANGED = {'C18': {'emotion_loss_mode': 'pattern-groupdro-author'},
           'C17': {'mosi_task_mode': 'binary'},
           'C17-binary-control': {'mosi_task_mode': 'binary'}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    from gcnet_missing_m3.training_resume import _json
    _json(Path(path), value)


def now():
    return datetime.now(timezone.utc).isoformat()


def candidate_config(reference, method, *, seed=66):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    if method not in METHODS + CONTROLS:
        raise ValueError('unsupported method')
    for key, value in EXPECTED.items():
        if reference.get(key) != value:
            raise ValueError(f'baseline protocol mismatch: {key}')
    if seed not in (66, 67, 68) or reference.get('seed') != seed:
        raise ValueError('use the exact same-seed Flat reference')
    if any(reference.get(key, False) for key in FORBIDDEN):
        raise ValueError('baseline already contains an intervention')
    for key in ('core20_method', 'osram_meaningful_block', 'osram_readout_candidate'):
        if reference.get(key, 'none') != 'none':
            raise ValueError(f'baseline has {key}')
    for key in ('initial_backbone_checkpoint', 'joint_pretrain_checkpoint', 'teacher_checkpoint'):
        if reference.get(key) is not None:
            raise ValueError('from-scratch baseline required')
    baseline = asdict(TrainConfig(**reference))
    delta = dict(core20_method=method, core20_aux_weight=AUX_WEIGHTS.get(method, 1.),
                 **CHANGED.get(method, {}))
    config = TrainConfig(**dict(baseline, **delta))
    return config, {key: {'baseline': baseline[key], 'candidate': value}
                    for key, value in asdict(config).items() if baseline[key] != value}


def verify_snapshot(root):
    snapshot = json.loads((root / 'SNAPSHOT.json').read_text())
    for name, digest in snapshot['source_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('immutable source changed: ' + name)
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS + CONTROLS, required=True)
    parser.add_argument('--seed', type=int, choices=(66, 67, 68), default=66)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('must bind the exact healthy host GPU UUID')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    if f'4, {args.gpu_uuid}' in devices or not any(line.strip().endswith(args.gpu_uuid) for line in devices.splitlines()):
        raise ValueError('GPU4 forbidden or UUID not found')
    snapshot = verify_snapshot(source)
    data = json.loads(args.data_manifest.read_text())
    for path, digest in data['files'].items():
        if sha(path) != digest:
            raise ValueError('data hash mismatch: ' + path)
    reference = json.loads(args.reference.read_text())
    config, delta = candidate_config(reference, args.method, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'RUN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance_path = args.output / 'PROVENANCE.json'
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text())
        if prior.get('status') == 'complete':
            print('already complete', flush=True)
            return
    provenance = dict(status='running', label=LABEL, method=args.method, seed=args.seed,
        started_utc=now(), pid=os.getpid(), server=socket.gethostname(), gpu_uuid=args.gpu_uuid,
        code_commit=snapshot['code_commit'], snapshot_sha256=sha(source / 'SNAPSHOT.json'),
        reference=str(args.reference), reference_config_sha256=sha(args.reference),
        data_manifest_sha256=sha(args.data_manifest), configuration_delta=delta,
        config=asdict(config), environment={'python': sys.version, 'platform': platform.platform()})
    write(provenance_path, provenance)
    write(args.output / 'RAW_CONFIG.json', asdict(config))
    try:
        import torch
        from gcnet_missing_m3.train_gcnet import run_experiment
        from gcnet_missing_m3.training_resume import TrainingState
        torch.set_num_threads(1)
        identity = {'snapshot': provenance['snapshot_sha256'], 'data': provenance['data_manifest_sha256'],
                    'config': hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()}
        state = TrainingState(args.output, identity=identity)
        torch.cuda.reset_peak_memory_stats()
        metrics = run_experiment(config, *data['feature_roots'], args.output, training_state=state)
        if len(json.loads((args.output / 'history.json').read_text())) != 100:
            raise ValueError('missing completed epochs')
        artifacts = ['config.json', 'metrics.json', 'history.json', 'last_training.pt']
        artifacts += [f'best_miss_0p{i}.pt' for i in range(8)]
        artifacts += [f'predictions_miss_0p{i}.npz' for i in range(8)]
        if any(not (args.output / name).is_file() for name in artifacts):
            raise ValueError('incomplete selected checkpoint/prediction artifacts')
        verify_snapshot(source)
        provenance.update(status='complete', finished_utc=now(), exit_code=0,
            outputs_verified=True, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            artifact_sha256={name: sha(args.output / name) for name in artifacts})
        write(provenance_path, provenance)
    except BaseException as error:
        provenance.update(status='failed', finished_utc=now(), error=repr(error), exit_code=1)
        write(provenance_path, provenance)
        raise


if __name__ == '__main__':
    main()
