"""Full-state epoch100 continuation with a deliberate constant-LR reduction."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def continued_config(cfg, lr):
    if cfg.epochs != 100 or cfg.lr_schedule != 'constant' or cfg.learning_rate != .001:
        raise ValueError('Requires original constant100epoch config')
    if not 0 < lr < cfg.learning_rate:
        raise ValueError('Continuation LR must be strictly lower')
    return replace(cfg, epochs=150, learning_rate=lr)


def lower_optimizer_lr(state, lr):
    if state['next_epoch'] != 100 or len(state['history']) != 100:
        raise ValueError('Requires complete epoch100 state')
    if not state['optimizer']['state'] or state['scheduler'] is not None or state['scaler'] is not None:
        raise ValueError('Requires original Adam moments and no scheduler/scaler')
    old = [g['lr'] for g in state['optimizer']['param_groups']]
    if not old or any(value != .001 for value in old) or not 0 < lr < .001:
        raise ValueError('Unexpected optimizer group LR')
    for group in state['optimizer']['param_groups']:
        group['lr'] = lr
    return old


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--historical-source', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--wrapper-commit', required=True)
    p.add_argument('--lr', type=float, default=.0001)
    args = p.parse_args()
    assert args.output.resolve() != args.original.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
    assert any(row.strip().endswith(args.gpu_uuid) for row in rows.splitlines()) and f'4, {args.gpu_uuid}' not in rows
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (args.output/'last_training.pt').exists(), 'Already initialized; inspect before resubmitting'
    prior = json.loads((args.original/'PROVENANCE.json').read_text())
    assert prior['status'] == 'complete' and prior['outputs_verified'] and prior['exit_code'] == 0
    historical = json.loads(Path('/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66/PROVENANCE.json').read_text())
    for name, digest in historical['source_sha256'].items():
        if name.endswith('.py'):
            assert sha(args.historical_source/name) == digest
    assert prior['identity']['source'] == historical['identity']['source']
    assert sha(args.data_manifest) == prior['identity']['data']
    data = json.loads(args.data_manifest.read_text())
    for name, digest in data['files'].items():
        assert sha(name) == digest
    original_sha = sha(args.original/'last_training.pt')
    assert original_sha == prior['artifact_sha256']['last_training.pt']
    sys.path.insert(0, str(args.historical_source.resolve()))
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.training_resume import TrainingState, _atomic, _json
    torch.set_num_threads(1)
    original_cfg = trainer.TrainConfig(**prior['effective_config'])
    assert asdict(original_cfg) == prior['effective_config'], 'Historical defaults changed'
    assert original_cfg.seed in (66,67,68) and original_cfg.osram_meaningful_block in ('none','nested_gnn_rooted_evidence')
    assert original_cfg.training_objective == 'emotion-only'
    cfg = continued_config(original_cfg, args.lr)
    assert {k for k in asdict(cfg) if asdict(cfg)[k] != asdict(original_cfg)[k]} == {'epochs','learning_rate'}
    state = torch.load(args.original/'last_training.pt', map_location='cpu', weights_only=False)
    old_lrs = lower_optimizer_lr(state, args.lr)
    original_history = state['history']
    assert set(state['best_references']) == {str(i/10) for i in range(8)}
    original_steps = sorted(set(float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v))
    identity = {**state['identity'], 'config': hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest(),
                'continuation': 'epoch100-to150-lr1e-4', 'wrapper': sha(Path(__file__))}
    state['identity'] = identity
    shutil.copytree(args.original/'training_versions', args.output/'training_versions')
    shutil.copy2(args.original/'metrics.json', args.output/'ORIGINAL_100_METRICS.json')
    _atomic(args.output/'last_training.pt', lambda handle: torch.save(state, handle))
    record = dict(status='running', label='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle',
                  model=prior.get('model', 'nested' if cfg.osram_meaningful_block != 'none' else 'flat'), seed=cfg.seed, pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  original=str(args.original), original_last_sha256=original_sha,
                  original_identity=prior['identity'], identity=identity,
                  historical_source=str(args.historical_source), historical_commit=prior.get('historical_commit',prior.get('code_commit')),
                  wrapper_commit=args.wrapper_commit, wrapper_sha256=sha(Path(__file__)), gpu_uuid=args.gpu_uuid,
                  effective_config=asdict(cfg), configuration_delta={'epochs':[100,150], 'learning_rate':[.001,args.lr]},
                  original_optimizer_lrs=old_lrs, original_optimizer_steps=original_steps,
                  full_state_restored_with_explicit_lr_override=True)
    _json(args.output/'PROVENANCE.json',record)
    _json(args.output/'RAW_CONFIG.json',asdict(cfg))
    original_train = trainer.train_epoch
    signature = inspect.signature(original_train)
    def observed_train(*positional, **keywords):
        bound = signature.bind(*positional, **keywords).arguments
        epoch, optimizer = bound['epoch'], bound['optimizer']
        actual_lrs = [float(g['lr']) for g in optimizer.param_groups]
        assert actual_lrs == [args.lr]*3 and 100 <= epoch < 150
        steps = sorted(set(float(v['step']) for v in optimizer.state.values() if 'step' in v))
        if epoch == 100:
            assert steps == original_steps, 'Optimizer moments/steps were not restored'
        _json(args.output/'learning_rates'/f'epoch_{epoch+1:03d}.json',
              dict(epoch=epoch+1, rates=actual_lrs, optimizer_steps=steps,
                   original_steps_verified=(epoch==100)))
        return original_train(*positional, **keywords)
    trainer.train_epoch = observed_train
    try:
        recovery = TrainingState(args.output, identity=identity, schedule_identity=state['schedule_identity'])
        del state
        metrics = trainer.run_experiment(cfg,*data['feature_roots'],args.output,training_state=recovery)
        history = json.loads((args.output/'history.json').read_text())
        assert len(history) == 150 and history[:100] == original_history
        original_metrics = json.loads((args.output/'ORIGINAL_100_METRICS.json').read_text())
        assert metrics['mask_sha256'] == original_metrics['mask_sha256']
        for rate in [str(i/10) for i in range(8)]:
            assert metrics['selected_weighted_f1_by_rate'][rate] >= original_metrics['selected_weighted_f1_by_rate'][rate]
        traces = list((args.output/'learning_rates').glob('epoch_*.json'))
        assert len(traces) == 50
        artifacts = ['config.json','metrics.json','history.json','last_training.pt']
        artifacts += [f'best_miss_0p{i}.pt' for i in range(8)]
        artifacts += [f'predictions_miss_0p{i}.npz' for i in range(8)]
        assert all((args.output/name).is_file() for name in artifacts)
        assert sha(args.original/'last_training.pt') == original_sha
        record.update(status='complete', outputs_verified=True, exit_code=0,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifact_sha256={name:sha(args.output/name) for name in artifacts},
                      learning_rate_sha256={path.name:sha(path) for path in traces})
    except BaseException as error:
        record.update(status='failed', error=repr(error), exit_code=1)
        raise
    finally:
        _json(args.output/'PROVENANCE.json',record)


if __name__ == '__main__':
    main()
