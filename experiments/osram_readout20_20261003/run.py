"""One fixed, authorized seed66 candidate. No training protocol overrides."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_cfg84_history_query_random_20260928 import run as common

read, write, sha = common.read, common.write, common.sha
LABEL = common.LABEL + '; SINGLE-SEED 20-CANDIDATE SCREENING HAS SELECTION BIAS'
CANDIDATES = ('film', 'se', 'eca', 'cbam', 'gct', 'simam', 'sk', 'mlb', 'mfb',
              'mutan', 'block', 'mcb', 'ban', 'dcnv2', 'cin', 'autoint', 'din',
              'dlrm', 'aff', 'nonlocal')
GPU_UUIDS = {'1': 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
             '5': 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62'}
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66')
DATASET = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
EXPECTED = dict(seed=66, dataset='CMUMOSI', epochs=100, batch_size=32,
                optimizer='adam', learning_rate=.001, weight_decay=.00001,
                training_objective='emotion-only', task_regression_loss='mse',
                mosi_task_mode='regression', completion_path='none',
                train_rate_mode='cyclic', train_missing_rates=[i / 10 for i in range(8)],
                checkpoint_selection='test-oracle-per-rate', evaluation_protocol='official',
                evaluate_test=True, osram_bidirectional=False, osram_write_step=.6,
                osram_readout_fusion='flat', osram_ablation='full', osram_emotion_ablation='full',
                osram_gap_read='residual', osram_output_dim=1600, osram_num_heads=8,
                osram_key_dim=64, osram_value_dim=64, latent_dim=256,
                osram_forward_slot_reuse=False, backbone_type='osram', device='cuda',
                lr_schedule='constant', readout_type='shared')
FORBIDDEN_FLAGS = ('paired_history_views', 'osram_relation_block', 'osram_relation_dual_readout',
                   'osram_decision_correction', 'osram_gap_increment_filter', 'osram_post_grn', 'osram_history_query_adapter',
                   'classification_completion', 'osram_local_skip_gate', 'osram_memory_only_adapter',
                   'osram_history_input_gate', 'osram_local_evidence_gate',
                   'osram_hierarchical_evidence_gate', 'osram_hierarchical_feature_only', 'completion_write_to_memory',
                   'local_context_residual', 'node_interaction_residual', 'text_core')


def now():
    return datetime.now(timezone.utc).isoformat()


def require_active_screen():
    status = read(Path(__file__).with_name('SCREEN_STATUS.json'))
    if status.get('status') != 'approved' or not status.get('training_authorized_for_this_manifest'):
        raise RuntimeError('This basic-operator screen was withdrawn by the user; do not launch it')


def effective_config_dict(config):
    # TrainConfig normalizes train_missing_rates to tuple; JSON stores lists.
    return json.loads(json.dumps(asdict(config)))


def candidate_config(base, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Only the fixed twenty candidates are authorized')
    for key, value in EXPECTED.items():
        if base.get(key) != value:
            raise ValueError(f'Baseline {key}={base.get(key)!r}, expected {value!r}')
    for key in FORBIDDEN_FLAGS:
        if base.get(key, False):
            raise ValueError(f'Baseline adaptation enabled: {key}')
    for key in ('initial_backbone_checkpoint', 'b2_base_checkpoint', 'b2_pretrain_checkpoint',
                'teacher_checkpoint', 'text_subspace_checkpoint'):
        if base.get(key) is not None:
            raise ValueError(f'Not from scratch: {key}')
    if base.get('osram_readout_candidate', 'none') != 'none':
        raise ValueError('Baseline already contains a candidate')
    return dict(base, osram_readout_candidate=candidate)


def validate_manifest(manifest, commit):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Use immutable full forty-character source commit')
    if manifest.get('code_commit') != commit:
        raise ValueError('Manifest and requested source commit disagree')
    if manifest.get('candidates') != list(CANDIDATES):
        raise ValueError('Manifest must lock exactly the twenty authorized IDs in order')
    for key, value in (('seed', 66), ('epochs', 100)):
        if manifest.get(key, value) != value:
            raise ValueError(f'Unexpected manifest {key}')


def validate_gpu(gpu):
    if str(gpu) not in GPU_UUIDS:
        raise ValueError('Select one host GPU1 or GPU5; GPU4 is always forbidden')


def gpu_check(gpu='1'):
    validate_gpu(gpu)
    line = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, free = [x.strip() for x in line.split(',')]
    if uuid != GPU_UUIDS[str(gpu)]:
        raise RuntimeError(f'Host GPU{gpu} UUID mismatch')
    return int(free)


def source_hashes():
    paths = set(REPO.glob('gcnet_missing_m3/**/*.py'))
    paths.update(REPO.glob('gcnet/**/*.py'))
    paths.update(REPO.glob('experiments/**/*.py'))
    paths.update(REPO.glob('*.py'))
    paths.add(Path(common.__file__).resolve())
    paths.add(REPO / 'experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py')
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def verify_source(manifest, commit):
    validate_manifest(manifest, commit)
    locked = manifest.get('source_sha256')
    if not locked:
        raise ValueError('Runtime manifest must include locked source_sha256')
    actual = source_hashes()
    if actual != locked:
        raise ValueError('Immutable source snapshot differs from manifest')
    return actual


def completion_status(output):
    """Cheap stdlib-only audit; full mask verification is performed before provenance completion."""
    output = Path(output)
    if not (output / 'PROVENANCE.json').exists():
        return ('incomplete' if output.exists() and any(output.iterdir()) else 'missing'), 'no provenance'
    try:
        p = read(output / 'PROVENANCE.json')
        if p.get('status') == 'failed':
            return 'failed', p.get('error', 'training failure')
        if p.get('status') != 'complete':
            return 'incomplete', p.get('status', 'unknown provenance')
        history = read(output / 'history.json')
        metrics = read(output / 'metrics.json')
        if (len(history) != 100 or [h.get('epoch') for h in history] != list(range(1, 101))
                or metrics.get('selection_protocol') != 'per-rate-test-oracle'):
            return 'incomplete', 'history or selection protocol mismatch'
        if set(metrics.get('test', {})) != {f'{i / 10:.1f}' for i in range(8)}:
            return 'incomplete', 'missing rate metrics'
        files = ['history.json', 'metrics.json', 'config.json', 'PARAMETERS.json']
        files.extend(f'best_miss_0p{i}.pt' for i in range(8))
        for name in files:
            if not (output / name).is_file() or (output / name).stat().st_size == 0:
                return 'incomplete', f'missing or empty {name}'
        required = ('source_sha256', 'reference_sha256', 'code_commit', 'manifest_sha256',
                    'environment', 'pid', 'gpu', 'gpu_uuid', 'outputs_verified')
        if any(not p.get(key) for key in required):
            return 'incomplete', 'missing provenance verification'
        if (p['gpu_uuid'] != GPU_UUIDS.get(str(p['gpu']))
                or p['config']['seed'] != 66 or p['config']['epochs'] != 100):
            return 'incomplete', 'unexpected provenance protocol'
        for name in ('metrics.json', 'history.json', 'config.json', 'PARAMETERS.json'):
            if p.get('artifact_sha256', {}).get(name) != sha(output / name):
                return 'incomplete', f'{name} hash mismatch'
        return 'complete', None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return 'incomplete', repr(error)


def parameter_counts(config):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    result = {}
    for candidate in ('none', config['osram_readout_candidate']):
        with torch.random.fork_rng(devices=[]):
            model = _build_model(TrainConfig(**dict(config, osram_readout_candidate=candidate)),
                                 (512, 1024, 1024))
        result[candidate] = dict(total=sum(p.numel() for p in model.parameters()),
                                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
    result['added'] = result[config['osram_readout_candidate']]['total'] - result['none']['total']
    return result


def train(args):
    require_active_screen()
    validate_gpu(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the single recorded host GPU')
    manifest = read(args.manifest)
    sources = verify_source(manifest, args.commit)
    config = candidate_config(read(args.reference / 'config.json'), args.candidate)
    if read(args.reference / 'metrics.json').get('selection_protocol') != 'per-rate-test-oracle':
        raise ValueError('Baseline metrics protocol mismatch')
    if read(args.reference / 'PROVENANCE.json').get('status') != 'complete':
        raise ValueError('Baseline incomplete')
    if gpu_check(args.gpu) < 5000:
        raise RuntimeError('Insufficient free GPU memory; protocol will not be changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.run.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(exist_ok=False)
        record = dict(status='starting', label=LABEL, candidate=args.candidate, seed=66,
                      code_commit=args.commit, integration_baseline_commit='5eb6061',
                      original_baseline_implementation_commit='7d56881f13e50b98a65a8fc9ca19f31222b39a55',
                      source_sha256=sources, manifest_sha256=sha(args.manifest),
                      reference_sha256={n: sha(args.reference / n) for n in ('config.json', 'metrics.json')},
                      baseline=str(args.reference), config=config, server='biggpu', from_scratch=True,
                      gpu=args.gpu, gpu_uuid=GPU_UUIDS[args.gpu], pid=os.getpid(), started_at=now())
        write(args.output / 'PROVENANCE.json', record)
        try:
            import torch
            from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
            torch.set_num_threads(2)
            effective = TrainConfig(**config)
            record.update(status='training', effective_config=effective_config_dict(effective),
                          environment=dict(python=platform.python_version(), torch=torch.__version__,
                          cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                          gpu_name=torch.cuda.get_device_name(0), platform=platform.platform()))
            write(args.output / 'PROVENANCE.json', record)
            write(args.output / 'PARAMETERS.json', parameter_counts(config))
            roots = [str(args.dataset / 'CMUMOSI/features' / name) for name in
                     ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
            run_experiment(effective, *roots, output_dir=str(args.output))
            common.verify_outputs(args.output, args.reference)
            if len(read(args.output / 'history.json')) != 100:
                raise ValueError('Training did not complete exactly 100 epochs')
            if read(args.output / 'config.json') != effective_config_dict(effective):
                raise ValueError('Saved effective configuration mismatch')
            if verify_source(manifest, args.commit) != sources:
                raise ValueError('Source changed during training')
            record.update(status='complete', outputs_verified=True,
                artifact_sha256={p.name: sha(p) for p in args.output.iterdir()
                                 if p.is_file() and p.suffix in ('.pt', '.json', '.npz')
                                 and p.name != 'PROVENANCE.json'})
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            record['updated_at'] = now()
            write(args.output / 'PROVENANCE.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', choices=CANDIDATES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--gpu', default='1')
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--dataset', type=Path, default=DATASET)
    train(p.parse_args())
