"""Fresh one-stage V3 classification; INTERNAL per-rate TEST-oracle ACC.

Only seed66: five original LOSO folds per IEMOCAP task and one official MELD
split. UA/WF1 use the same ACC-selected prediction. No trained weight is loaded.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import (
    RATES, full_training_split, persist_epoch, publish_selected, validate_selected,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
BASELINE_ROOT = Path('/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1')
DATASETS = ('IEMOCAPFour', 'IEMOCAPSix', 'MELD')


def source_catalog():
    """Pin current numerical code and every reused experiment helper."""
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import SOURCES
    additions = ('gcnet_missing_m3/source_role_v3.py',
        'experiments/osram_source_role_v3_20261021/worker.py',
        'experiments/osram_source_relation_attention_20261021/worker.py',
        'experiments/osram_v3_classification_20261021/worker.py')
    numerical = tuple(str(path.relative_to(ROOT)) for package in ('gcnet_missing_m3', 'gcnet_modality_jepa')
                      for path in sorted((ROOT / package).glob('*.py')))
    return tuple(dict.fromkeys((*SOURCES, *additions, *numerical)))


def make_config(dataset, fold=1, seed=66, *, baseline_root=BASELINE_ROOT):
    if dataset not in DATASETS or seed != 66 or fold not in range(1, 6) or (dataset == 'MELD' and fold != 1):
        raise ValueError('Only seed66 IEMOCAP4/6 folds1..5 and MELD fold1 are authorized')
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if dataset == 'MELD':
        source = ROOT / 'experiments/osram_meld_wd_seed66_20261018/BASELINE.json'
        raw = dict(json.loads(source.read_text())['config_without_seed'], seed=seed)
    else:
        classes = 4 if dataset == 'IEMOCAPFour' else 6
        source = Path(baseline_root) / f'iemocap{classes}/seed_66/fold_{fold}/config.json'
        raw = json.loads(source.read_text())
    base = TrainConfig(**raw)
    expected = dict(dataset=dataset, fold=fold, seed=seed, epochs=50 if dataset == 'MELD' else 100,
        learning_rate=1e-5 if dataset == 'MELD' else .001, weight_decay=1e-5,
        batch_size=32, optimizer='adam', lr_schedule='constant', evaluation_protocol='official',
        training_objective='emotion-only', emotion_loss_mode='sample-mean',
        missing_mask_protocol='gcnet-random-mask-v1', train_rate_mode='cyclic',
        train_missing_rates=tuple(i/10 for i in range(8)), osram_readout_fusion='flat',
        osram_output_dim=1600, osram_write_step=.6, osram_disable_local_skip=False,
        osram_bidirectional=False, paired_history_views=False, osram_meaningful_block='none',
        osram_tasr=False, osram_source_role=False, osram_source_relation=False, osram_source_role_v3=False,
        initial_backbone_checkpoint=None, teacher_checkpoint=None, backbone_type='osram')
    if any(getattr(base, name) != value for name, value in expected.items()):
        raise ValueError('Accepted original Flat configuration changed')
    cfg = replace(base, osram_source_role_v3=True, preserve_selection_artifacts=True,
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy',
        evaluate_test=True, osram_rpca_lr_multiplier=1., device='cuda:0')
    reference = dict(source=str(source), source_sha256=h.sha(source), original_config=raw,
        config_diff={key: dict(original=getattr(base, key), effective=value)
                     for key, value in asdict(cfg).items() if getattr(base, key) != value},
        numerical_source='current immutable run source; historical 785a012 is not imported',
        historical_training_sha='785a01242ef1a82b97d5337acbb33ba963df5537' if dataset != 'MELD' else None,
        trajectory_parity_to_historical_training_claimed=False)
    return cfg, reference


def setup(dataset, fold, seed, uuid, output):
    """Install accepted data paths before importing trainers, then build fresh."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import build_cases
    from experiments.osram_iemocap_b6_transfer_20261015.worker import LABEL_HASHES
    case = build_cases()['MELD' if dataset == 'MELD' else 'IEMOCAPSix']
    if dataset == 'MELD':
        feature = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
        if h.sha(feature) != case['dataset_sha256']:
            raise ValueError('Accepted MELD data changed')
        os.environ['GCNET_MELD_PICKLE'] = str(feature)
        os.environ['GCNET_DATASET_ROOT'] = str(Path(case['numerical_source']['source']) / 'dataset')
        roots = ('MELD-pickle',) * 3
        data_manifest = dict(dataset_sha256=case['dataset_sha256'], feature=str(feature))
    else:
        filename, digest = LABEL_HASHES[dataset]
        if h.sha(DATA_ROOT / 'IEMOCAP' / filename) != digest:
            raise ValueError('Accepted IEMOCAP labels changed')
        os.environ['GCNET_DATASET_ROOT'] = str(DATA_ROOT)
        os.environ.pop('GCNET_MELD_PICKLE', None)
        roots = tuple(str(DATA_ROOT / 'IEMOCAP/features' / name)
                      for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT'))
        if not all(Path(root).is_dir() for root in roots):
            raise FileNotFoundError('Accepted IEMOCAP features unavailable')
        data_manifest = dict(label_sha256=digest, label_file=str(DATA_ROOT / 'IEMOCAP' / filename),
            feature_roots=roots)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent / 'feature_cache' / dataset)
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    from experiments.osram_source_role_v3_20261021.worker import original_digest
    environment = require_replication_environment(torch)
    repair_uuid_rng(torch)
    gpu = guard_gpu(uuid)
    cfg, reference = make_config(dataset, fold, seed)
    train, validation, test, *dimensions = get_loaders(*roots, dataset=dataset,
        num_folder=1 if dataset == 'MELD' else 5, batch_size=32, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    train_loader, validation_loader, test_loader = train[fold-1], validation[fold-1], test[fold-1]
    if dataset == 'MELD':
        split = dict(method='official-train-dev-test', no_extra_training_holdout=True,
            train_indices=sorted(train_loader.sampler.indices),
            validation_indices=sorted(validation_loader.sampler.indices),
            test_indices=sorted(test_loader.sampler.indices))
        ids = [set(split[name + '_indices']) for name in ('train', 'validation', 'test')]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError('Official MELD split overlap')
    else:
        split = full_training_split(train_loader, test_loader, fold)
    baseline = _build_model(replace(cfg, osram_source_role_v3=False), tuple(dimensions), torch.device('cuda:0'))
    initial = h.backbone_digest(baseline)
    model = _build_model(cfg, tuple(dimensions), torch.device('cuda:0')).requires_grad_(True).train()
    if original_digest(model) != initial:
        raise RuntimeError('V3 initialization altered original Flat parameters')
    bundle = dict(model=model, trainer=trainer, cfg=cfg, dimensions=tuple(dimensions),
        case=dict(dataset=dataset, classes={'MELD': 7, 'IEMOCAPFour': 4, 'IEMOCAPSix': 6}[dataset],
            original_reference=reference, data_manifest=data_manifest),
        split=split, train_loader=train_loader, validation_loader=validation_loader, test_loader=test_loader,
        environment=environment, gpu=gpu, initial_backbone_sha256=initial)
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    return bundle, baseline


def replay_selected_bank(rows, accepted):
    """Replay an exact selected mask onto identical original-loader batches."""
    import torch
    result, cursor = [], 0
    for row in rows:
        size = len(row['labels'])
        for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices'):
            np.testing.assert_array_equal(row[key], accepted[key][cursor:cursor+size])
        availability = row['availability'].clone().transpose(0, 1)
        valid = availability.sum(-1) > 0
        replacement = torch.as_tensor(accepted['availability'][cursor:cursor+size], dtype=availability.dtype)
        if replacement.shape != (size, 3) or not torch.all((replacement == 0) | (replacement == 1)) or replacement.sum(-1).lt(1).any():
            raise ValueError('Accepted masks must be binary nonempty modality observations')
        availability[valid] = replacement
        result.append(dict(row, availability=availability.transpose(0, 1)))
        cursor += size
    if cursor != len(accepted['labels']):
        raise ValueError('Exact accepted mask cohort was not fully replayed')
    return result


def build_test_masks(bundle):
    from experiments.osram_iemocap6_router_fivefold_20261019.worker import build_mask_bank, selected_from_bank
    from experiments.osram_tasr_20261018 import runner_helpers as h
    cfg = bundle['cfg']
    bank = build_mask_bank(bundle, 'test')
    origin = dict(kind='regenerated-current-fold-GCNet-seed66-epoch0', historical_selected_mask_reuse=False,
        evaluation='fixed bank; does not claim historical per-epoch mask resampling')
    if cfg.dataset == 'MELD':
        from experiments.osram_tasr_20261018.protocol import build_cases
        case = build_cases()['MELD']
        sources = {}
        for rate in RATES:
            ref = case['evaluation_references'][rate]
            path = Path(ref['selection'])
            if h.sha(path) != ref['selection_sha256']:
                raise ValueError('Accepted MELD selected mask artifact hash changed: ' + rate)
            with np.load(path, allow_pickle=False) as stored:
                accepted = {key: stored[key].copy() for key in
                    ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability')}
            if hashlib.sha256(accepted['availability'].tobytes()).hexdigest() != ref['mask_sha256']:
                raise ValueError('Accepted MELD exact mask changed: ' + rate)
            bank[rate] = replay_selected_bank(bank[rate], accepted)
            sources[rate] = dict(path=str(path), sha256=ref['selection_sha256'], mask_sha256=ref['mask_sha256'])
        origin.update(kind='accepted-MELD-baseline-selected-mask-only', historical_selected_mask_reuse=True, sources=sources)
    elif cfg.dataset == 'IEMOCAPSix':
        import torch
        directory = Path('/data2/yb/remote_experiments/osram_iemocap6_gcnet_testbest_20261020/execution_v1/runs/IEMOCAPSix') / f'fold_{cfg.fold}/seed_66/flat'
        path = directory / 'test_masks.pt'
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=False)
            actual, refs = selected_from_bank(old)
            summary = json.loads((directory / 'SUMMARY.json').read_text())
            if summary.get('status') != 'complete':
                raise ValueError('Corrected IEMOCAP6 Flat reference is not complete')
            for rate in RATES:
                if refs[rate]['mask_sha256'] != summary['identity']['test_masks'][rate]:
                    raise ValueError('Corrected IEMOCAP6 reference mask hash changed')
                bank[rate] = replay_selected_bank(bank[rate], actual[rate])
            origin.update(kind='accepted-corrected-IEMOCAP6-flat-fixed-test-bank', historical_selected_mask_reuse=True,
                source=str(path), source_sha256=h.sha(path), summary_sha256=h.sha(directory / 'SUMMARY.json'))
    selected, references = selected_from_bank(bank)
    for row in references.values():
        row['origin'] = origin['kind']
    return bank, selected, references, origin


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', choices=DATASETS, required=True)
    result.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    result.add_argument('--seed', type=int, choices=(66,), default=66)
    result.add_argument('--phase', choices=('smoke', 'train'), required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--resume', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.gpu_uuid.startswith('GPU-') or (args.dataset == 'MELD' and args.fold != 1):
        raise ValueError('Healthy UUID and authorized dataset fold required')
    if args.resume and args.phase != 'train':
        raise ValueError('Resume is only supported for full training state')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019 import run as joint
    from experiments.osram_source_role_v3_20261021.worker import original_digest, validate_gradients
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick, hooks = args.output / 'SUMMARY.json', time.monotonic(), []
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed experiment must not repeat')
    summary = dict(status='preflight', dataset=args.dataset, fold=args.fold, seed=args.seed, mode='v3',
        phase=args.phase, source_commit=args.source_commit, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(2)
        h.write_json(path, summary)
        bundle, baseline = setup(args.dataset, args.fold, args.seed, args.gpu_uuid, args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        cfg, model = bundle['cfg'], bundle['model']
        module = model.osram.source_role_v3
        test_masks, selected, references, mask_origin = build_test_masks(bundle)
        _, old_arrays = joint.evaluate(bundle, baseline, 'flat', bundle['test_loader'], batches=test_masks['0.3'], max_batches=1)
        _, new_arrays = joint.evaluate(bundle, model, 'flat', bundle['test_loader'], batches=test_masks['0.3'], max_batches=1)
        for key in ('labels', 'availability', 'log_probs_router'):
            np.testing.assert_array_equal(old_arrays[key], new_arrays[key], err_msg='V3 zero-init parity: ' + key)
        del baseline, old_arrays, new_arrays
        torch.cuda.empty_cache()
        optimizer = joint.make_optimizer(model, cfg, 'flat')
        if {id(p) for p in model.parameters()} != {id(p) for group in optimizer.param_groups for p in group['params']}:
            raise RuntimeError('One-stage optimizer must contain every original and V3 parameter')
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in source_catalog()},
            dataset=args.dataset, fold=args.fold, seed=args.seed, mode='flat-v3', config=asdict(cfg), split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], original_reference_case=bundle['case'],
            initialization='canonical fresh factory; no trained weights; original parameters exact',
            one_stage=True, all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
            selection='per-rate TEST ACC; strict improvement; earliest ties; same-call UA/WF1',
            test_masks={r: row['mask_sha256'] for r, row in references.items()}, test_mask_origin=references,
            evaluation_bank=mask_origin, historical_selected_mask_reuse=mask_origin['historical_selected_mask_reuse'],
            historical_trajectory_parity_claimed=False)
        if not identity['all_parameters_trainable']:
            raise RuntimeError('One-stage classification screen cannot freeze parameters')
        torch.save(test_masks, args.output / 'test_masks.pt')
        torch.save(selected, args.output / 'selected_masks.pt')
        h.write_json(args.output / 'config.json', identity['config'])
        summary.update(status='running', identity=identity, config=identity['config'], environment=bundle['environment'],
            gpu=bundle['gpu'], zero_init_predictions_exact=True, initial_backbone_sha256=bundle['initial_backbone_sha256'],
            total_parameters=sum(p.numel() for p in model.parameters()), source_role_v3_parameters=sum(p.numel() for p in module.parameters()))
        h.write_json(path, summary)
        observed = {}
        def capture(name):
            def check(grad):
                if not torch.isfinite(grad).all():
                    raise RuntimeError('Nonfinite V3 task gradient: ' + name)
                observed[name] = observed.get(name, False) or bool(grad.ne(0).any())
            return check
        for name, parameter in module.named_parameters():
            hooks.append(parameter.register_hook(capture(name)))
        def watch_inputs(unused, inputs):
            for name, value in zip(('source_base', 'source_gap'), inputs[1:3]):
                if value.requires_grad:
                    value.register_hook(capture('input.' + name))
        hooks.append(module.register_forward_pre_hook(watch_inputs))
        reset_training_stochasticity(args.seed, args.fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            joint.restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            validate_selected(args.output, best)
        before_v3, before_original = h.backbone_digest(module), original_digest(model)
        cumulative = {}
        for row in history:
            for name, value in row.get('gradient_paths', {}).items():
                cumulative[name] = cumulative.get(name, False) or value
        for epoch in (range(1) if args.phase == 'smoke' else range(start, cfg.epochs)):
            observed.clear()
            row = joint.train_epoch(bundle, model, optimizer, epoch, 'flat', max_batches=3 if args.phase == 'smoke' else None)
            for name, value in observed.items():
                cumulative[name] = cumulative.get(name, False) or value
            validate_gradients(cumulative)
            row.update(gradient_paths=copy.deepcopy(observed), learning_rates=[g['lr'] for g in optimizer.param_groups])
            evaluated = {r: joint.evaluate(bundle, model, 'flat', bundle['test_loader'], batches=test_masks[r],
                         max_batches=1 if args.phase == 'smoke' else None) for r in RATES}
            row['test'] = {r: item[0] for r, item in evaluated.items()}
            history.append(row)
            improved = persist_epoch(args.output, model, optimizer, epoch=epoch+1, identity=identity,
                history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best, selected_epoch_by_rate={r: item['epoch'] for r, item in best.items()},
                seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', dataset=args.dataset, fold=args.fold, epoch=epoch+1,
                task_NLL=row['task_NLL'], improved_rates=improved, test_ACC={r: item[0]['router']['ACC'] for r, item in evaluated.items()})), flush=True)
        validate_selected(args.output, best)
        if args.phase == 'smoke':
            if sum(row['optimizer_steps'] for row in history) < 3:
                raise RuntimeError('Smoke requires three genuine optimizer updates')
            if before_v3 == h.backbone_digest(module) or before_original == original_digest(model):
                raise RuntimeError('Smoke must update both original and V3 parameters')
            stored = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            joint.restore(model, optimizer, stored, identity)
            if any(not torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full-state checkpoint model restoration differs')
            _, replay = joint.evaluate(bundle, model, 'flat', bundle['test_loader'], batches=test_masks['0.3'], max_batches=1)
            with np.load(args.output / best['0.3']['predictions'], allow_pickle=False) as saved:
                for key, value in replay.items():
                    np.testing.assert_array_equal(value, saved[key], err_msg='Selected checkpoint replay: ' + key)
            summary.update(smoke_only=True, smoke=history[0], v3_and_backbone_updated=True,
                full_checkpoint_reloaded=True, selected_artifacts_exact_reload=True)
        else:
            summary['evaluation'] = publish_selected(args.output, best)
            summary['aggregate'] = h.aggregate_rates({r: item['router'] for r, item in summary['evaluation'].items()})
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary.update(status='complete', seconds=time.monotonic()-tick, final_backbone_sha256=original_digest(model))
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
