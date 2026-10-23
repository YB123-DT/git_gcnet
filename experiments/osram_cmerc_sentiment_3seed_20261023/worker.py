"""Original Flat cmerc controls: MOSI keeps its schedule; MOSEI uses constant1e-4."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NUMERICAL_COMMIT = '0da38b3326fdad09da0195e49b8b7d6424b4f613'
RATES = tuple(f'{i/10:.1f}' for i in range(8))
MOSI66 = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')
MOSI_SEEDS = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/execution_v1/runs/MOSI')
MOSEI66 = Path('/data2/yb/remote_experiments/osram_mosei_flat_lr1e4_seed66_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat')
MOSEI_SEEDS = Path('/data2/yb/remote_experiments/osram_mosei_gcnet_flat_lr1e4_20261021/execution_v1/runs/CMUMOSEI')
DISABLED_SCHEMA = dict(osram_source_relation=False, osram_source_relation_layout='source-mean',
    osram_source_role_v3=False, osram_source_role_v3_mode='residual',
    osram_source_role_v3_pair_hidden=128, osram_source_role_v3_variant='standard',
    osram_source_role_v3_dim=64, osram_conditional_reread=False,
    osram_reread_mode='free', osram_reread_rank_rtol=1e-5, osram_reread_rank_atol=1e-7)


def epochs(dataset):
    if dataset not in ('CMUMOSI', 'CMUMOSEI'):
        raise ValueError('Only MOSI/MOSEI approved')
    return 150 if dataset == 'CMUMOSI' else 60


def learning_rate(dataset, epoch):
    if not 0 <= epoch < epochs(dataset):
        raise ValueError('Outside approved training budget')
    return (3e-4 if epoch < 100 else 3e-4/10) if dataset == 'CMUMOSI' else 1e-4


def reference_path(dataset, seed):
    epochs(dataset)
    if seed not in (66,67,68):
        raise ValueError('Only seeds66/67/68 approved')
    if dataset == 'CMUMOSI':
        return MOSI66 if seed == 66 else MOSI_SEEDS/f'seed_{seed}'/'flat'
    return MOSEI66 if seed == 66 else MOSEI_SEEDS/f'seed_{seed}'/'flat'


def validate_reference_config(reference, expected):
    """Older canonical Flat lacks later disabled extensions; reject anything else."""
    reference = dict(reference)
    for key in set(reference)-set(expected):
        if key not in DISABLED_SCHEMA or reference[key] != DISABLED_SCHEMA[key]:
            raise ValueError('Unverified schema/config difference: ' + key)
        reference.pop(key)
    if json.loads(json.dumps(reference)) != json.loads(json.dumps(expected)):
        raise ValueError('Same-seed original Flat config differs')


def setup(dataset, seed, uuid, output, prior):
    import numpy as np
    import sklearn
    import torch
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as mosi
    from experiments.osram_mosei_gcnet_router_20261020 import worker as mosei
    from experiments.osram_mosi_recent_history_20261018 import run as mosi_data
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    environment = dict(python=sys.executable, python_version=sys.version,
        torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    if (environment['torch'],environment['numpy'],environment['sklearn']) != ('1.13.1+cu117','1.26.4','1.3.2'):
        raise ValueError('Use unchanged cmerc-repro environment')
    gpu = guard_gpu(uuid)
    if dataset == 'CMUMOSI':
        manifest, proof = mosi.load_manifest(mosi_data.DATA, mosi_data.DATA_SHA)
        for path, digest in manifest['files'].items():
            if h.sha(path) != digest: raise ValueError('MOSI feature changed: '+path)
    else:
        manifest, proof = mosei.load_manifest()
    if manifest['files'] != prior['identity']['data_files']:
        raise ValueError('Reference feature manifest differs')
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output/'feature_cache'/dataset)
    from gcnet_missing_m3 import train_gcnet as trainer
    # The old canonical model sources are identical to the completed cmerc MOSI66.
    template_path = MOSI66 if dataset == 'CMUMOSI' else MOSEI_SEEDS/'seed_67'/'flat'
    template = json.loads((template_path/'SUMMARY.json').read_text())
    raw = dict(template['config'],seed=seed)
    validate_reference_config(prior['config'], raw)
    cfg = trainer.TrainConfig(**raw)
    required = dict(dataset=dataset,seed=seed,epochs=epochs(dataset),batch_size=32,
        learning_rate=learning_rate(dataset,0),weight_decay=1e-5,optimizer='adam',
        missing_mask_protocol='gcnet-random-mask-v1',checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1',evaluate_test=True,training_objective='emotion-only',
        train_rate_mode='cyclic',task_regression_loss='mse',mosi_task_mode='regression',
        osram_output_dim=1600,osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,
        osram_write_step=.6,osram_bidirectional=False,osram_readout_fusion='flat',
        osram_meaningful_block='none',osram_tasr=False,osram_source_role=False,
        paired_history_views=False,completion_path='none',osram_disable_local_skip=False)
    for key, value in required.items():
        if getattr(cfg,key) != value: raise ValueError('Original protocol changed: '+key)
    if tuple(cfg.train_missing_rates) != tuple(float(r) for r in RATES):
        raise ValueError('Original cyclic rates changed')
    # Compare canonical numerical model/sampler bytes, not a later runner's schema.
    reference_core = {p:v for p,v in template['identity']['source_hashes'].items()
                      if p.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/'))}
    if {p:h.sha(ROOT/p) for p in reference_core} != reference_core:
        raise ValueError('Canonical model/sampler source changed')
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset=dataset,num_folder=1,batch_size=32,num_workers=0,seed=66,
        validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    bundle = dict(trainer=trainer,cfg=cfg,dimensions=tuple(dimensions),
        train_loader=train[0],validation_loader=validation[0],test_loader=test[0],
        environment=environment,gpu=gpu,data_manifest=proof,data_files=manifest['files'],
        split={name:loader.protocol_metadata for name,loader in
               (('train',train[0]),('validation',validation[0]),('test',test[0]))})
    bundle['seed_protocol'] = seed_training_loader(bundle,seed)
    factory = mosi.fresh_backbone if dataset == 'CMUMOSI' else mosei.fresh_backbone
    model = scalar.wrap_model(factory(cfg,tuple(dimensions),torch.device('cuda:0')),'flat')
    if sum(p.numel() for p in model.parameters()) != 13509793:
        raise ValueError('Original Flat parameter count changed')
    if h.backbone_digest(model) != prior['initial_backbone_sha256']:
        raise ValueError('Same-seed original initialization differs')
    for key in ('split','seed_protocol'):
        if bundle[key] != prior['identity'][key]: raise ValueError('Reference differs: '+key)
    bundle.update(model=model,initial_backbone_sha256=h.backbone_digest(model),canonical_core=reference_core,
                  canonical_template=str(template_path),s0_same_seed_core_exact=all(
                      prior['identity']['source_hashes'].get(p)==v for p,v in reference_core.items()))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('CMUMOSI','CMUMOSEI'),required=True)
    parser.add_argument('--seed',type=int,choices=(66,67,68),required=True)
    parser.add_argument('--phase',choices=('train',),required=True)
    parser.add_argument('--mode',choices=('flat',),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.dataset == 'CMUMOSI' and args.seed == 66:
        raise ValueError('Reuse the completed cmerc MOSI66; no duplicate training')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists() and not args.resume: raise FileExistsError('Independent fresh output required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path,tick = args.output/'SUMMARY.json',time.monotonic()
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':
        raise ValueError('Completed run cannot repeat silently')
    summary = dict(status='preflight',dataset=args.dataset,seed=args.seed,mode='flat',phase='train',
        source_commit=args.source_commit,numerical_commit=NUMERICAL_COMMIT,pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN; unified cmerc original Flat')
    try:
        torch.set_num_threads(1);h.write_json(path,summary)
        reference = reference_path(args.dataset,args.seed)
        prior = json.loads((reference/'SUMMARY.json').read_text())
        ref_history = json.loads((reference/'history.json').read_text())
        budget = epochs(args.dataset)
        if prior['status']!='complete' or prior['epoch']!=budget or len(ref_history)!=budget:
            raise ValueError('Require complete same-seed reference')
        bundle = setup(args.dataset,args.seed,args.gpu_uuid,args.output,prior)
        model,cfg = bundle['model'],bundle['cfg']
        banks = torch.load(reference/'fixed_masks.pt',map_location='cpu',weights_only=False)
        test_bank = banks.get('test',banks)
        hashes = {rate:hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in bank)).hexdigest()
                  for rate,bank in test_bank.items()}
        reference_hashes = prior['identity']['mask_hashes']
        if hashes != reference_hashes.get('test',reference_hashes):
            raise ValueError('Exact reference evaluation masks differ')
        sources = dict(bundle['canonical_core'])
        for p in ('experiments/osram_text_path_router_joint_mosi_20261019/run.py',
                  'experiments/osram_text_path_router_joint_20261019/run.py',
                  'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
                  'experiments/osram_cmerc_sentiment_3seed_20261023/worker.py',
                  'experiments/osram_tasr_20261018/runner_helpers.py'):
            sources[p] = h.sha(ROOT/p)
        identity = dict(source_commit=args.source_commit,numerical_commit=NUMERICAL_COMMIT,
            source_hashes=sources,config=asdict(cfg),dataset=args.dataset,seed=args.seed,mode='flat',
            environment=bundle['environment'],split=bundle['split'],seed_protocol=bundle['seed_protocol'],
            mask_hashes={'test':hashes},data_manifest=bundle['data_manifest'],data_files=bundle['data_files'],
            reference=dict(path=str(reference),summary_sha256=h.sha(reference/'SUMMARY.json'),
                history_sha256=h.sha(reference/'history.json'),initialization_exact=True,
                canonical_core_exact=True,s0_same_seed_core_exact=bundle['s0_same_seed_core_exact'],
                canonical_template=bundle['canonical_template'],
                disabled_schema_fields={k:prior['config'][k] for k in set(prior['config'])-set(asdict(cfg))}),
            old_weights_loaded=False,selection='per-rate TEST W-F1; earliest strict-best ties')
        optimizer = scalar.make_optimizer(model,cfg,'flat')
        summary.update(status='running',config=asdict(cfg),identity=identity,environment=bundle['environment'],
            gpu=bundle['gpu'],initial_backbone_sha256=bundle['initial_backbone_sha256'],
            total_parameters=sum(p.numel() for p in model.parameters()),router_parameters=0)
        torch.save({'test':test_bank},args.output/'fixed_masks.pt')
        h.write_json(args.output/'config.json',asdict(cfg));h.write_json(path,summary)
        reset_training_stochasticity(args.seed,cfg.fold)
        best,history,start = {},[],0
        if args.resume:
            state = torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity)
            start,history,best = state['epoch'],state['history'],state['best_by_rate']
            selected.validate_selected(args.output,best)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start,budget):
            for group in optimizer.param_groups: group['lr'] = learning_rate(args.dataset,epoch)
            row = scalar.train_epoch(bundle,model,optimizer,epoch,'flat')
            for key in ('epoch','mask_trace_sha256','optimizer_steps','learning_rates'):
                if row[key]!=ref_history[epoch][key]: raise ValueError('Training protocol differs: '+key)
            evaluated = {r:scalar.evaluate(bundle,model,'flat','test',test_bank[r]) for r in RATES}
            row['test'] = {r:v[0] for r,v in evaluated.items()};history.append(row)
            improved = selected.persist_epoch(args.output,model,optimizer,epoch=epoch+1,
                identity=identity,history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
                training_inputs_exact=True,seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary)
            print(json.dumps(dict(epoch=epoch+1,dataset=args.dataset,seed=args.seed,task_MSE=row['task_MSE'],
                learning_rates=row['learning_rates'],improved_rates=improved)),flush=True)
        summary['evaluation'] = selected.publish_selected(args.output,best)
        summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        if {p:h.sha(ROOT/p) for p in sources} != sources: raise ValueError('Immutable source changed')
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:
        h.write_json(path,summary)


if __name__ == '__main__':
    main()
