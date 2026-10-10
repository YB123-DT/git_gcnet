"""Train original Flat/Nested with passive statistics using sealed old source."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from monitor import GradientMonitor


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_config(cfg, model, schedule, seed=None):
    """Preserve every historical setting except the explicit experimental axes."""
    if model not in ('flat', 'nested') or schedule not in ('constant', 'cosine'):
        raise ValueError('Unknown model or schedule')
    return replace(cfg, seed=cfg.seed if seed is None else seed, lr_schedule=schedule, osram_meaningful_block=(
        'none' if model == 'flat' else 'nested_gnn_rooted_evidence'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('flat','nested'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--wrapper-commit', required=True)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--seed', type=int, choices=(65,66), default=66)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert f'4, {args.gpu_uuid}' not in gpu_rows
    assert any(row.strip().endswith(args.gpu_uuid) for row in gpu_rows.splitlines())
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Historical source from original Nested seed66, not current mutable worktree.
    historical = Path('/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66/PROVENANCE.json')
    old = json.loads(historical.read_text())
    for name,digest in old['source_sha256'].items():
        if name.endswith('.py'):
            assert sha(args.source/name)==digest,name
    data = json.loads(args.data_manifest.read_text())
    assert sha(args.data_manifest)==old['data_manifest_sha256']
    for name,digest in data['files'].items():
        assert sha(name)==digest,name
    sys.path.insert(0,str(args.source.resolve()))
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.training_resume import TrainingState,_json
    torch.set_num_threads(1)
    cfg = trainer.TrainConfig(**json.loads((args.reference/'config.json').read_text()))
    assert cfg.seed==66 and cfg.epochs==100 and cfg.osram_meaningful_block=='none'
    assert cfg.training_objective=='emotion-only' and cfg.train_rate_mode=='cyclic'
    assert cfg.emotion_loss_mode=='sample-mean' and cfg.task_regression_loss=='mse'
    assert cfg.osram_readout_fusion=='flat' and cfg.gradient_clip_norm==1.
    cfg=select_config(cfg,args.model,args.lr_schedule,seed=args.seed)
    identity=dict(source=old['identity']['source'], data=sha(args.data_manifest),
                  config=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest(),
                  monitor=sha(Path(__file__).with_name('monitor.py')),wrapper=sha(Path(__file__)))
    provenance_path=args.output/'PROVENANCE.json'
    if provenance_path.exists():
        prior=json.loads(provenance_path.read_text())
        assert prior['identity']==identity
        if prior['status']=='complete':
            print('already complete',flush=True)
            return
    monitor=GradientMonitor(args.model,latent_dim=cfg.latent_dim)
    original_train=trainer.train_epoch
    signature=inspect.signature(original_train)
    def observed_train(*positional,**keywords):
        bound=signature.bind(*positional,**keywords).arguments
        model,epoch=bound['model'],bound['epoch']
        if args.lr_schedule == 'cosine':
            _json(args.output/'learning_rates'/f'epoch_{epoch+1:03d}.json',
                  dict(epoch=epoch+1, rates=[float(g['lr']) for g in bound['optimizer'].param_groups]))
        monitor.start_epoch(model,epoch,bound['config'])
        original_clip=torch.nn.utils.clip_grad_norm_
        def observed_clip(parameters,max_norm,*clip_args,**clip_kwargs):
            monitor.before_clip(model,max_norm)
            return original_clip(parameters,max_norm,*clip_args,**clip_kwargs)
        torch.nn.utils.clip_grad_norm_=observed_clip
        try:
            result=original_train(*positional,**keywords)
            rows=monitor.finish_epoch(expected_steps=result['optimizer_steps'])
            _json(args.output/'gradients'/f'epoch_{epoch+1:03d}.json',rows)
            return result
        finally:
            monitor.active=False
            torch.nn.utils.clip_grad_norm_=original_clip
    trainer.train_epoch=observed_train
    provenance=dict(status='running',label='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle',
        model=args.model,seed=cfg.seed,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        source=str(args.source),historical_commit=old['code_commit'],wrapper_commit=args.wrapper_commit,
        identity=identity,gpu_uuid=args.gpu_uuid,effective_config=asdict(cfg),
        monitor_only=True,from_scratch=True,resumed=(args.output/'last_training.pt').exists())
    _json(provenance_path,provenance)
    _json(args.output/'RAW_CONFIG.json',asdict(cfg))
    try:
        metrics=trainer.run_experiment(cfg,*data['feature_roots'],args.output,
            training_state=TrainingState(args.output,identity=identity))
        history=json.loads((args.output/'history.json').read_text())
        assert len(history)==100
        reference=json.loads((args.reference/'metrics.json').read_text())
        if cfg.seed == 66:
            assert metrics['mask_sha256']==reference['mask_sha256']
        for i,row in enumerate(history,1):
            saved=json.loads((args.output/'gradients'/f'epoch_{i:03d}.json').read_text())
            assert len(saved)==row['train']['optimizer_steps']
        if args.lr_schedule == 'cosine':
            trace=[json.loads((args.output/'learning_rates'/f'epoch_{i:03d}.json').read_text()) for i in range(1,101)]
            assert len(trace)==100 and trace[0]['rates']==[.0002]*3 and trace[4]['rates']==[.001]*3
            assert trace[-1]['rates']==[0.]*3
        artifacts=['config.json','metrics.json','history.json','last_training.pt']
        artifacts += [f'best_miss_0p{i}.pt' for i in range(8)]
        artifacts += [f'predictions_miss_0p{i}.npz' for i in range(8)]
        assert all((args.output/name).exists() for name in artifacts)
        provenance.update(status='complete',outputs_verified=True,exit_code=0,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            artifact_sha256={name:sha(args.output/name) for name in artifacts},
            gradient_sha256={f'epoch_{i:03d}.json':sha(args.output/'gradients'/f'epoch_{i:03d}.json') for i in range(1,101)})
        if args.lr_schedule == 'cosine':
            provenance['learning_rate_sha256']={f'epoch_{i:03d}.json':sha(args.output/'learning_rates'/f'epoch_{i:03d}.json') for i in range(1,101)}
    except BaseException as error:
        provenance.update(status='failed',exit_code=1,error=repr(error))
        raise
    finally:
        _json(provenance_path,provenance)


if __name__=='__main__':
    main()
