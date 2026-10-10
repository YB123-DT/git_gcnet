"""Budget-only continuation of original Nested using its historical source."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--historical-source', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--wrapper-commit', required=True)
    args = parser.parse_args()
    assert args.output.resolve() != args.original.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    assert any(line.strip().endswith(args.gpu_uuid) for line in gpu_rows.splitlines())
    assert f'4, {args.gpu_uuid}' not in gpu_rows
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'EXTEND.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Never overwrite an already initialized continuation or original run.
    assert not (args.output / 'last_training.pt').exists(), 'continuation already initialized'
    prior = json.loads((args.original / 'PROVENANCE.json').read_text())
    assert prior['outputs_verified'] and prior['exit_code'] == 0
    for name, digest in prior['source_sha256'].items():
        if name.endswith('.py'):
            assert sha(args.historical_source / name) == digest, name
    data = json.loads(args.data_manifest.read_text())
    assert sha(args.data_manifest) == prior['data_manifest_sha256']
    for name, digest in data['files'].items():
        assert sha(name) == digest, name
    original_sha = sha(args.original / 'last_training.pt')
    assert original_sha == prior['artifact_sha256']['last_training.pt']
    sys.path.insert(0, str(args.historical_source.resolve()))
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from gcnet_missing_m3.training_resume import TrainingState, _atomic, _json
    torch.set_num_threads(1)
    cfg = TrainConfig(**prior['config'])
    assert cfg.epochs == 100 and cfg.lr_schedule == 'constant'
    assert cfg.osram_meaningful_block == 'nested_gnn_rooted_evidence'
    assert cfg.training_objective == 'emotion-only'
    before = asdict(cfg)
    cfg.epochs = 150
    assert {k for k in before if before[k] != asdict(cfg)[k]} == {'epochs'}
    state = torch.load(args.original / 'last_training.pt', map_location='cpu', weights_only=False)
    assert state['next_epoch'] == 100 and len(state['history']) == 100
    assert state['optimizer']['state'] and set(state['best_references']) == {str(i / 10) for i in range(8)}
    assert state['scheduler'] is None and state['scaler'] is None
    original_identity = state['identity'].copy()
    identity = {**original_identity,
                'config': hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest(),
                'extension': '100-to-150-budget-only', 'wrapper': sha(Path(__file__))}
    # Copy independently: no hard links and no writes to original checkpoints.
    shutil.copytree(args.original / 'training_versions', args.output / 'training_versions')
    shutil.copy2(args.original / 'metrics.json', args.output / 'ORIGINAL_100_METRICS.json')
    state['identity'] = identity
    _atomic(args.output / 'last_training.pt', lambda stream: torch.save(state, stream))
    record = dict(status='running', label='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle',
                  seed=cfg.seed, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  original=str(args.original), original_last_sha256=original_sha,
                  original_identity=original_identity, identity=identity,
                  historical_source=str(args.historical_source), historical_commit=prior['code_commit'],
                  wrapper_commit=args.wrapper_commit, wrapper_sha256=sha(Path(__file__)),
                  gpu_uuid=args.gpu_uuid, effective_config=asdict(cfg),
                  configuration_delta={'epochs': {'original': 100, 'continued': 150}},
                  resumed_model_optimizer_rng_selection_schedule=True)
    _json(args.output / 'PROVENANCE.json', record)
    _json(args.output / 'RAW_CONFIG.json', asdict(cfg))
    print(f'Resuming original Nested seed{cfg.seed} at epoch101 with full optimizer/RNG/BEST state', flush=True)
    try:
        trainer_state = TrainingState(args.output, identity=identity, schedule_identity=state['schedule_identity'])
        del state
        metrics = run_experiment(cfg, *data['feature_roots'], args.output, training_state=trainer_state)
        assert len(json.loads((args.output / 'history.json').read_text())) == 150
        original_metrics = json.loads((args.output / 'ORIGINAL_100_METRICS.json').read_text())
        assert metrics['mask_sha256'] == original_metrics['mask_sha256']
        for rate in (str(i / 10) for i in range(8)):
            assert metrics['selected_weighted_f1_by_rate'][rate] >= original_metrics['selected_weighted_f1_by_rate'][rate]
        artifacts = ['config.json', 'metrics.json', 'history.json', 'last_training.pt']
        artifacts += [f'best_miss_0p{i}.pt' for i in range(8)]
        artifacts += [f'predictions_miss_0p{i}.npz' for i in range(8)]
        assert all((args.output / name).is_file() for name in artifacts)
        assert sha(args.original / 'last_training.pt') == original_sha
        record.update(status='complete', outputs_verified=True, exit_code=0,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifact_sha256={name: sha(args.output / name) for name in artifacts})
    except BaseException as error:
        record.update(status='failed', error=repr(error), exit_code=1)
        raise
    finally:
        _json(args.output / 'PROVENANCE.json', record)


if __name__ == '__main__':
    main()
