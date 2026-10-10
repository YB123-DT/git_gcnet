"""Original Nested with local warmup/lower LR and random residual decoders."""
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

from variant import random_nested_adapter, separate_nested_group, apply_nested_lr


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'data-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--wrapper-commit', required=True)
    parser.add_argument('--nested-lr', type=float, default=.0005)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    args = parser.parse_args()
    assert 0 < args.nested_lr < .001 and 1 <= args.warmup_epochs <= 100
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert any(row.strip().endswith(args.gpu_uuid) for row in rows.splitlines())
    assert f'4, {args.gpu_uuid}' not in rows
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    old = json.loads(Path('/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66/PROVENANCE.json').read_text())
    for name, digest in old['source_sha256'].items():
        if name.endswith('.py'):
            assert sha(args.source/name) == digest
    assert sha(args.data_manifest) == old['data_manifest_sha256']
    data = json.loads(args.data_manifest.read_text())
    for name, digest in data['files'].items():
        assert sha(name) == digest
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3 import meaningful_input_new40 as input_module
    from gcnet_missing_m3.training_resume import TrainingState, _json
    torch.set_num_threads(1)
    cfg = trainer.TrainConfig(**json.loads((args.reference/'config.json').read_text()))
    assert cfg.seed == 66 and cfg.epochs == 100 and cfg.osram_meaningful_block == 'none'
    assert cfg.learning_rate == .001 and cfg.lr_schedule == 'constant'
    assert cfg.training_objective == 'emotion-only' and cfg.pretrained_learning_rate is None
    assert cfg.gradient_clip_norm == 1. and cfg.osram_readout_fusion == 'flat'
    cfg = replace(cfg, osram_meaningful_block='nested_gnn_rooted_evidence')
    policy = dict(nested_target_lr=args.nested_lr, nested_warmup_epochs=args.warmup_epochs,
                  nested_residual_decoder_initialization='nn.Linear default random',
                  residual=True, other_lr=.001, global_gradient_clip_norm=1.)
    identity = dict(source=old['identity']['source'], data=sha(args.data_manifest),
        config=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest(),
        policy=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest(),
        wrapper=sha(Path(__file__)), variant=sha(Path(__file__).with_name('variant.py')))
    provenance_path = args.output/'PROVENANCE.json'
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text())
        assert prior['identity'] == identity
        if prior['status'] == 'complete':
            print('already complete',flush=True)
            return
    record = dict(status='running', label='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle',
        pid=os.getpid(), seed=cfg.seed, gpu_uuid=args.gpu_uuid, identity=identity,
        source=str(args.source), historical_commit=old['code_commit'],
        wrapper_commit=args.wrapper_commit, effective_config=asdict(cfg), runtime_policy=policy,
        started_utc=datetime.now(timezone.utc).isoformat(),
        resumed=(args.output/'last_training.pt').exists())
    _json(provenance_path, record)
    _json(args.output/'RAW_CONFIG.json',asdict(cfg))
    _json(args.output/'RUNTIME_POLICY.json',policy)
    original_build = input_module.build_new40
    def build(method, latent_dim=256, num_heads=8, value_dim=64):
        if method == 'nested_gnn_rooted_evidence':
            return random_nested_adapter(latent_dim, num_heads, value_dim)
        return original_build(method, latent_dim, num_heads, value_dim)
    input_module.build_new40 = build
    original_groups = trainer._optimizer_parameter_groups
    def groups(model, config):
        initial = model.osram.meaningful_block.core
        decoders = [initial.local_decoder, *initial.memory_decoders]
        assert getattr(initial,'residual',True) and all(layer.weight.ne(0).any() for layer in decoders)
        record['initial_decoder_weight_norms'] = [float(layer.weight.detach().norm()) for layer in decoders]
        _json(provenance_path,record)
        return separate_nested_group(*original_groups(model, config), model, args.nested_lr)
    trainer._optimizer_parameter_groups = groups
    original_schedule = trainer._apply_epoch_learning_rate
    def schedule(optimizer, config, epoch):
        original_schedule(optimizer, config, epoch)
        apply_nested_lr(optimizer, epoch, args.nested_lr, args.warmup_epochs)
    trainer._apply_epoch_learning_rate = schedule
    original_train = trainer.train_epoch
    signature = inspect.signature(original_train)
    def observed_train(*positional, **keywords):
        bound = signature.bind(*positional, **keywords).arguments
        epoch, optimizer = bound['epoch'], bound['optimizer']
        actual = [dict(name=g.get('name','original'),lr=float(g['lr']),
                       parameter_count=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]
        assert len(actual) == 4 and all(g['lr'] == .001 for g in actual if g['name'] != 'nested')
        expected = args.nested_lr * min(epoch+1,args.warmup_epochs)/args.warmup_epochs
        assert [g['lr'] for g in actual if g['name']=='nested'] == [expected]
        _json(args.output/'learning_rates'/f'epoch_{epoch+1:03d}.json',dict(epoch=epoch+1,groups=actual))
        return original_train(*positional, **keywords)
    trainer.train_epoch = observed_train
    try:
        metrics = trainer.run_experiment(cfg,*data['feature_roots'],args.output,
            training_state=TrainingState(args.output,identity=identity))
        history = json.loads((args.output/'history.json').read_text())
        assert len(history) == 100 and len(list((args.output/'learning_rates').glob('*.json'))) == 100
        reference = json.loads((args.reference/'metrics.json').read_text())
        assert metrics['mask_sha256'] == reference['mask_sha256']
        artifacts = ['config.json','metrics.json','history.json','last_training.pt','RUNTIME_POLICY.json']
        artifacts += [f'best_miss_0p{i}.pt' for i in range(8)]
        artifacts += [f'predictions_miss_0p{i}.npz' for i in range(8)]
        assert all((args.output/name).is_file() for name in artifacts)
        record.update(status='complete',outputs_verified=True,exit_code=0,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            artifact_sha256={name:sha(args.output/name) for name in artifacts})
    except BaseException as error:
        record.update(status='failed',exit_code=1,error=repr(error))
        raise
    finally:
        _json(provenance_path,record)


if __name__ == '__main__':
    main()
