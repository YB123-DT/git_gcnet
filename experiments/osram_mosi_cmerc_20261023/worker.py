"""One MOSI66 environment control; reuse the exact s0 numerical source snapshot."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')
NUMERICAL_COMMIT = '5877678ce3fac9358a94f1217e3dbac5509f6f58'
ROOT = Path(__file__).resolve().parents[2]


def validate_environment(environment):
    expected = dict(torch='1.13.1+cu117', numpy='1.26.4', sklearn='1.3.2')
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError('Use the existing cmerc-repro environment without dependency changes')


def validate_source_hashes(actual, reference):
    if actual != reference:
        raise ValueError('The environment-only control must reuse exact reference numerical sources')


def check_training_parity(actual, reference):
    # Different software can change loss/gradients, but not our experimental inputs.
    for key in ('epoch', 'mask_trace_sha256', 'optimizer_steps', 'learning_rates'):
        if actual[key] != reference[key]:
            raise ValueError('Training input/protocol changed: ' + key)


def setup(uuid, output, prior):
    import numpy as np
    import sklearn
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
    from experiments.osram_mosi_recent_history_20261018 import run as data_reference
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu

    environment = dict(python=sys.executable, python_version=sys.version,
                       torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    validate_environment(environment)
    gpu = guard_gpu(uuid)
    manifest, proof = original.load_manifest(data_reference.DATA, data_reference.DATA_SHA)
    if manifest['files'] != prior['identity']['data_files']:
        raise ValueError('Reference feature manifest differs')
    for path, digest in manifest['files'].items():
        if h.sha(path) != digest:
            raise ValueError('Original MOSI feature changed: ' + path)
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent/'feature_cache'/'CMUMOSI')
    cfg = trainer.TrainConfig(**prior['identity']['config'])
    if (cfg.dataset != 'CMUMOSI' or cfg.seed != 66 or cfg.epochs != 150
            or cfg.learning_rate != 3e-4 or cfg.weight_decay != 1e-5 or cfg.batch_size != 32
            or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'
            or cfg.checkpoint_selection != 'test-oracle-per-rate'
            or cfg.checkpoint_selection_metric != 'weighted_f1'):
        raise ValueError('Require the accepted original MOSI66 Flat150 protocol')
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset='CMUMOSI', num_folder=1, batch_size=cfg.batch_size, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    model = scalar.wrap_model(original.fresh_backbone(cfg, tuple(dimensions), torch.device('cuda:0')), 'flat')
    if sum(p.numel() for p in model.parameters()) != 13509793:
        raise ValueError('Original Flat parameter count changed')
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), model=model,
        train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
        environment=environment, gpu=gpu, data_manifest=proof,
        initial_backbone_sha256=h.backbone_digest(model),
        split={name: loader.protocol_metadata for name, loader in
               (('train', train[0]), ('validation', validation[0]), ('test', test[0]))})
    bundle['seed_protocol'] = seed_training_loader(bundle, 66)
    for key in ('split', 'seed_protocol'):
        if bundle[key] != prior['identity'][key]:
            raise ValueError('Reference differs: ' + key)
    return bundle


def publish(output, summary, prior):
    """Use the saved selected predictions; do not resample masks or run inference."""
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_tasr_20261018 import runner_helpers as h
    summary['evaluation'] = selected.publish_selected(output, summary['best_by_rate'])
    summary['aggregate'] = scalar.aggregate(summary['evaluation'])
    rows = ['# MOSI Flat seed66 environment control', '', 'INTERNAL TEST-ORACLE SCREEN', '',
            'Same numerical source, data, config, seed, training masks and exact test masks; only environment changes.',
            '150epochs, LR3e-4 for100 then3e-5 for50, batch32, WD1e-5, original GCNet random missing.',
            'Fresh initialization, not continuation; no old trained weights are loaded.', '',
            '| Environment | Eight-rate ACC | W-F1 | High ACC | High W-F1 |',
            '|---|---:|---:|---:|---:|']
    for name, run in (('s0 reference', prior), ('cmerc-repro', summary)):
        a, b = run['aggregate']['router']['mean'], run['aggregate']['router']['high']
        rows.append(f"| {name} | {a['ACC']:.4f} | {a['WF1']:.4f} | {b['ACC']:.4f} | {b['WF1']:.4f} |")
    rows += ['', '| Rate | s0 W-F1 | cmerc W-F1 | Delta(pp) | Selected epoch |',
             '|---|---:|---:|---:|---:|']
    for rate, best in summary['best_by_rate'].items():
        old = prior['best_by_rate'][rate]['metrics']['router']['WF1']
        new = best['metrics']['router']['WF1']
        rows.append(f"| {rate} | {old:.4f} | {new:.4f} | {new-old:+.4f} | {best['epoch']} |")
    rows += ['', 'Eight-rate selection can combine different epochs; not single-checkpoint deployment performance.',
             'One seed and bundled software-version change, not a causal attribution to a particular package.',
             'No automatic extra seeds, Nested, V3 or hyperparameter search.', '']
    (output/'RESULT.md').write_text('\n'.join(rows))
    h.write_json(output/'RESULT.json', dict(status='complete', label='INTERNAL TEST-ORACLE SCREEN',
        reference=str(REFERENCE), reference_aggregate=prior['aggregate'],
        aggregate=summary['aggregate'], initial_parameters_exact=summary['initial_parameters_exact'],
        best_by_rate=summary['best_by_rate']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import shutil
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_mosi_router_lr1e5_20261020 import worker as lr_common
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib

    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh independent output directory')
    if available_gib() < 32 or shutil.disk_usage(args.output.parent).free < 8*2**30:
        raise RuntimeError('Insufficient RAM/disk; no settings or other tasks were changed')
    # guard_gpu excludes physical GPU4 before any CUDA model is built.
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    guard_gpu(args.gpu_uuid)
    free_mib = int(subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free_mib < 6144:
        raise RuntimeError('Require6GiB free on the healthy selected GPU')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, path = time.monotonic(), args.output/'SUMMARY.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run cannot repeat silently')
    summary = dict(status='preflight', pid=os.getpid(), seed=66, mode='flat', dataset='CMUMOSI',
        source_commit=args.source_commit, numerical_commit=NUMERICAL_COMMIT,
        label='INTERNAL TEST-ORACLE SCREEN; environment-only control')
    try:
        torch.set_num_threads(1)
        prior = json.loads((REFERENCE/'SUMMARY.json').read_text())
        reference_history = json.loads((REFERENCE/'history.json').read_text())
        if prior['status'] != 'complete' or prior['epoch'] != 150 or len(reference_history) != 150:
            raise ValueError('Require complete reference')
        sources = {p: h.sha(ROOT/p) for p in prior['identity']['source_hashes']}
        validate_source_hashes(sources, prior['identity']['source_hashes'])
        bundle = setup(args.gpu_uuid, args.output, prior)
        model, cfg = bundle['model'], bundle['cfg']
        masks = torch.load(REFERENCE/'fixed_masks.pt', map_location='cpu', weights_only=False)
        hashes = {split: {rate: hashlib.sha256(b''.join(r['availability'].numpy().tobytes()
                  for r in bank)).hexdigest() for rate, bank in rates.items()} for split, rates in masks.items()}
        if hashes != prior['identity']['mask_hashes']:
            raise ValueError('Original exact masks changed')
        identity = dict(source_commit=args.source_commit, numerical_commit=NUMERICAL_COMMIT,
            source_hashes=sources, runner_sha256=h.sha(Path(__file__)), config=asdict(cfg),
            seed=66, mode='flat', environment=bundle['environment'], split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            data_manifest=bundle['data_manifest'], reference_summary_sha256=h.sha(REFERENCE/'SUMMARY.json'),
            reference_history_sha256=h.sha(REFERENCE/'history.json'), old_weights_loaded=False)
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        summary.update(status='running', identity=identity, config=asdict(cfg), environment=bundle['environment'],
            gpu=bundle['gpu'], mask_hashes=hashes, total_parameters=sum(p.numel() for p in model.parameters()),
            initial_backbone_sha256=bundle['initial_backbone_sha256'],
            initial_parameters_exact=bundle['initial_backbone_sha256'] == prior['initial_backbone_sha256'])
        h.write_json(path, summary)
        h.write_json(args.output/'config.json', asdict(cfg))
        torch.save(masks, args.output/'fixed_masks.pt')
        reset_training_stochasticity(66, cfg.fold)
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
            for i, row in enumerate(history):
                check_training_parity(row, reference_history[i])
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, 150):
            lr_common.set_epoch_lrs(optimizer, epoch, backbone_lr=3e-4, final_backbone_lr=3e-5)
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat')
            check_training_parity(row, reference_history[epoch])
            evaluated = {rate: scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][rate])
                         for rate in selected.RATES}
            row['test'] = {rate: value[0] for rate, value in evaluated.items()}
            history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best,
                selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()}, training_inputs_exact=True,
                seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(epoch=epoch+1, task_MSE=row['task_MSE'],
                learning_rates=row['learning_rates'], improved_rates=improved)), flush=True)
        validate_source_hashes({p:h.sha(ROOT/p) for p in sources}, sources)
        publish(args.output, summary, prior)
        summary.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
