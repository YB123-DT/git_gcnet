"""MOSEI60 seed66 Flat/one-stage Router, original GCNet missing and Test-best.

INTERNAL TEST-ORACLE SCREEN. Eight selected checkpoints may have different
epochs. No protected model weights, MOSI setup, or MOSI150 LR schedule is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(str(i / 10) for i in range(8))
PROTOCOL = 'gcnet-random-mask-v1'
EPOCHS = 60
SMOKE_BATCHES = 3
CONFIG = Path('/data2/yb/remote_experiments/osram_mosei_cfg84_nojepa_20260919/seed_66/config.json')
CONFIG_SHA = '2b612c7093c696121f6e92762b24a75acf6db316106f2ff07b240603df529155'
INPUTS = Path('/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json')
INPUTS_SHA = 'fd03c4a6ca12abe8a55bde88531b7103761c4ed263c9fe381e30809aabd2a46c'


def configs(raw, seed=66):
    """Adopt accepted MOSEI cfg84 settings and explicitly record the new budget."""
    from gcnet_missing_m3.train_gcnet import TrainConfig
    if seed != 66:
        raise ValueError('Only MOSEI seed66 is approved for this paired screening')
    required = dict(dataset='CMUMOSEI', fold=1, seed=66, epochs=100, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, optimizer='adam', lr_schedule='constant',
        dropout=.5, projector_dropout=.1, latent_dim=256, backbone_type='osram',
        osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, osram_bidirectional=False, osram_readout_fusion='flat',
        fusion_type='mean', training_objective='emotion-only', train_rate_mode='cyclic',
        checkpoint_selection='test-oracle-per-rate', evaluation_protocol='official',
        emotion_loss_mode='sample-mean', mosi_task_mode='regression', task_regression_loss='mse',
        osram_meaningful_block='none', osram_relation_block=False, osram_post_grn=False,
        osram_base_first_gap=False, osram_memory_only_adapter=False,
        osram_history_recency=False, osram_causal_gru_history=False,
        osram_history_query_adapter=False, osram_history_input_gate=False,
        osram_local_evidence_gate=False, osram_hierarchical_evidence_gate=False,
        osram_local_skip_gate=False, osram_disable_local_skip=False,
        osram_decision_correction=False, osram_gap_increment_filter=False,
        osram_ced_block=False, osram_tasr=False, paired_history_views=False,
        completion_path='none', classification_completion=False,
        initial_backbone_checkpoint=None, disable_unused_aux_modules=True)
    for key, expected in required.items():
        if raw.get(key, expected) != expected:
            raise ValueError('Accepted MOSEI configuration changed: ' + key)
    cfg = TrainConfig(**dict(raw, seed=seed, epochs=EPOCHS, device='cuda:0',
        missing_mask_protocol=PROTOCOL, checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', evaluate_test=True))
    if tuple(cfg.train_missing_rates) != tuple(float(rate) for rate in RATES):
        raise ValueError('MOSEI train_missing_rates must remain cyclic 0.0 through 0.7')
    return cfg


def load_manifest(path=INPUTS, expected_sha256=INPUTS_SHA):
    """Hash the exact manifest AND every declared feature/label, without mutation."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    path = Path(path)
    if h.sha(path) != expected_sha256:
        raise ValueError('MOSEI data manifest SHA256 changed')
    data = json.loads(path.read_text())['data']['CMUMOSEI']
    if len(data['feature_roots']) != 3 or not data['files']:
        raise ValueError('MOSEI manifest must specify A/T/V and verified input files')
    for name, digest in data['files'].items():
        if h.sha(name) != digest:
            raise ValueError('MOSEI feature/label changed: ' + name)
    return data, dict(path=str(path), sha256=expected_sha256,
        dataset='CMUMOSEI', files_verified=len(data['files']), all_file_hashes_exact=True)


def build_loaders(manifest, cfg, trainer):
    """Official train/dev/test only: do not subtract any adapter holdout."""
    return trainer.get_loaders(*manifest['feature_roots'], dataset='CMUMOSEI',
        num_folder=1, batch_size=cfg.batch_size, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol='official')


def fresh_backbone(cfg, dimensions, device):
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    return _build_model(cfg, dimensions, device).requires_grad_(True).train()


def setup(gpu_uuid, output, seed=66):
    import torch
    import sklearn
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    if (torch.__version__ != '2.2.2+cu121' or np.__version__ != '1.24.3'
            or sklearn.__version__ != '1.3.2'):
        raise ValueError('MOSEI requires archived s0 Torch2.2.2/NumPy1.24.3/sklearn1.3.2')
    gpu = guard_gpu(gpu_uuid)
    manifest, proof = load_manifest()
    if h.sha(CONFIG) != CONFIG_SHA:
        raise ValueError('Accepted MOSEI baseline config SHA256 changed')
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent / 'feature_cache' / 'CMUMOSEI')
    # Dataset-root variables must precede the trainer's config-module import.
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg = configs(json.loads(CONFIG.read_text()), seed)
    train, validation, test, *dimensions = build_loaders(manifest, cfg, trainer)
    dimensions, device = tuple(dimensions), torch.device('cuda:0')
    backbone = fresh_backbone(cfg, dimensions, device)
    if sum(p.numel() for p in backbone.parameters()) != 13509793:
        raise ValueError('MOSEI cfg84 Large Flat parameter count differs from accepted architecture')
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=dimensions, model=backbone,
        train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
        initial_backbone_sha256=h.backbone_digest(backbone), gpu=gpu,
        data_manifest=proof, data_manifest_sha256=INPUTS_SHA, data_files=manifest['files'],
        baseline_config=dict(path=str(CONFIG), sha256=CONFIG_SHA,
            changes=dict(epochs=[100, EPOCHS], missing_mask_protocol=['historical-default', PROTOCOL],
                         checkpoint_selection_metric=['dataset-default', 'weighted_f1'],
                         device=['cuda', 'cuda:0'])),
        environment=dict(python=sys.executable, python_version=sys.version,
            torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        split={name: loader.protocol_metadata for name, loader in
               (('train', train[0]), ('validation', validation[0]), ('test', test[0]))})
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    return bundle


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('smoke', 'train'))
    parser.add_argument('--mode', required=True, choices=('flat', 'joint'))
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import (
        persist_epoch, validate_selected, publish_selected)
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output directory required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only full training runs support exact resume')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if args.resume and summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run cannot be repeated silently')
    summary = dict(status='preflight', dataset='CMUMOSEI', mode=args.mode, phase=args.phase,
        seed=args.seed, source_commit=args.source_commit, pid=os.getpid(),
        experiment_label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1)
        bundle = setup(args.gpu_uuid, args.output, args.seed)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model = scalar.wrap_model(bundle['model'], args.mode)
        optimizer = scalar.make_optimizer(model, bundle['cfg'], args.mode)
        # Both arms use this deterministic common bank; no fresh mask inference
        # is performed when a selected epoch's artifacts are published.
        masks = {'test': scalar.build_masks(bundle, 'test')}
        hashes = {'test': {rate: hashlib.sha256(b''.join(
            row['availability'].numpy().tobytes() for row in bank)).hexdigest()
            for rate, bank in masks['test'].items()}}
        source_files = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py',
            'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
            'gcnet_missing_m3/mixed_rate.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
            'gcnet_modality_jepa/mask_schedule.py', 'gcnet_modality_jepa/protocol.py',
            'gcnet_modality_jepa/train_gcnet.py',
            'experiments/osram_text_path_router_joint_20261019/joint.py',
            'experiments/osram_text_path_router_joint_20261019/run.py',
            'experiments/osram_text_path_router_20261019/router.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/router.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
            'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
            'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
            'experiments/osram_tasr_20261018/runner_helpers.py',
            str(Path(__file__).relative_to(ROOT)))
        identity = dict(source_commit=args.source_commit,
            source_hashes={name: h.sha(ROOT / name) for name in source_files},
            mode=args.mode, seed=args.seed, config=asdict(bundle['cfg']),
            split=bundle['split'], seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            dataset_manifest_sha256=INPUTS_SHA, data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'], baseline_config=bundle['baseline_config'],
            initialization='fresh-original-missing_m3_model_init:fold:1; no checkpoint loaded',
            selection='test-oracle-per-rate; router weighted_f1; strict greater; earliest ties',
            backbone_learning_rates=dict(constant=.001),
            router_learning_rates=[.0003] if args.mode == 'joint' else None,
            missing_protocol=PROTOCOL, no_adapter_holdout=True,
            test_masks='same fixed GCNet-legal banks across Flat/Router; no resampling each epoch')
        torch.save(masks, args.output / 'fixed_masks.pt')
        summary.update(identity=identity, config=identity['config'],
            environment=bundle['environment'], gpu=bundle['gpu'], mask_hashes=hashes,
            initial_backbone_sha256=bundle['initial_backbone_sha256'],
            total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            status='running')
        h.write_json(args.output / 'config.json', identity['config'])
        h.write_json(summary_path, summary)
        reset_training_stochasticity(args.seed, bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            validate_selected(args.output, best)
        initial_model = h.backbone_digest(model) if args.phase == 'smoke' else None
        initial_router = (h.backbone_digest(model.router)
                          if args.phase == 'smoke' and args.mode == 'joint' else None)
        for epoch in (range(1) if args.phase == 'smoke' else range(start, EPOCHS)):
            # Constant MOSEI LR: intentionally do NOT call MOSI150 set_epoch_lrs.
            expected_lrs = [.001, .0003] if args.mode == 'joint' else [.001]
            if [group['lr'] for group in optimizer.param_groups] != expected_lrs:
                raise ValueError('MOSEI constant learning rates changed')
            row = scalar.train_epoch(bundle, model, optimizer, epoch, args.mode,
                max_batches=SMOKE_BATCHES if args.phase == 'smoke' else None)
            if args.mode == 'joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('No nonzero Router task gradient reached the added module')
            evaluated = {rate: scalar.evaluate(bundle, model, args.mode, 'test', masks['test'][rate],
                max_batches=1 if args.phase == 'smoke' else None) for rate in RATES}
            row['test'] = {rate: result[0] for rate, result in evaluated.items()}
            history.append(row)
            improved = persist_epoch(args.output, model, optimizer, epoch=epoch + 1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch + 1, best_by_rate=best,
                selected_epoch_by_rate={rate: value['epoch'] for rate, value in best.items()},
                seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            h.write_json(summary_path, summary)
            print(json.dumps(dict(event='epoch', epoch=epoch + 1, task_MSE=row['task_MSE'],
                test_wf1={rate: result[0]['router']['WF1'] for rate, result in evaluated.items()},
                improved_rates=improved, selected_epochs=summary['selected_epoch_by_rate'],
                learning_rates=row['learning_rates'])), flush=True)
        if args.phase == 'smoke':
            if h.backbone_digest(model) == initial_model:
                raise RuntimeError('Smoke made no parameter update')
            if args.mode == 'joint' and h.backbone_digest(model.router) == initial_router:
                raise RuntimeError('Smoke Router parameters did not update')
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            if not all(torch.equal(value.cpu(), state['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full model checkpoint reload failed')
            validate_selected(args.output, best)
            metrics, arrays = scalar.evaluate(bundle, model, args.mode, 'test', masks['test']['0.3'], max_batches=1)
            with np.load(args.output / best['0.3']['predictions'], allow_pickle=False) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], saved[key], err_msg='Smoke exact selected replay: ' + key)
            summary.update(smoke=history[0], smoke_only=True, full_checkpoint_reloaded=True,
                selected_artifacts_exact_reload=True, smoke_test=metrics)
        else:
            summary['evaluation'] = publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for filename, digest in identity['source_hashes'].items():
            if h.sha(ROOT / filename) != digest:
                raise RuntimeError('Immutable training source changed: ' + filename)
        summary.update(status='complete', seconds=time.monotonic() - tick,
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
