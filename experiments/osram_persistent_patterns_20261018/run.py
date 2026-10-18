"""Finite frozen Persistent7 evaluation, one dataset per independent process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HELPERS = Path('/home/qinjinghui/remote_experiments/osram_meld_related_history_diagnostic_20261018/source_470e653')
GPU = 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'
MELD_REGISTRY = ROOT/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json'
MELD_PRIOR = Path('/home/qinjinghui/remote_experiments/osram_meld_persistent_unimodal_20261018/runs_v1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def check_source():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Immutable evaluation source changed: '+name)
    return manifest


def meld_sources():
    baseline = json.loads(MELD_REGISTRY.read_text())
    cases = []
    for run in baseline['runs']:
        seed = run['seed']
        config = dict(baseline['config_without_seed'], seed=seed)
        if config['learning_rate'] != 1e-5 or config['weight_decay'] != 1e-5:
            raise ValueError('Accepted MELD optimizer configuration changed')
        references = {}
        for rate in ('0.0', '0.3', '0.7'):
            ref = run['artifacts'][rate]
            references[rate] = dict(checkpoint=ref['checkpoint'], checkpoint_sha256=ref['checkpoint_sha256'],
                selection=ref['selected_predictions'], selection_sha256=ref['prediction_sha256'],
                mask_sha256=ref['availability_sha256'], epoch=ref['epoch'],
                expected_metrics=run['testbest']['rates'][rate], expected_config=config)
        cases.append(dict(dataset='MELD', classes=7, seed=seed, fold=1, output=run['output'],
            expected_config=config, references=references, baseline_id=baseline['baseline_id']))
    if [c['seed'] for c in cases] != [66, 67, 68]:
        raise ValueError('Require all accepted MELD seeds')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('MELD', 'IEMOCAPFour', 'IEMOCAPSix'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite an existing evaluation')
    manifest = check_source()
    sys.path.append(str(HELPERS))
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    # Numerical package binding happens before model/trainer imports. Dataset
    # workers are separate processes so the two historical sources never mix.
    if args.dataset != 'MELD':
        from experiments.osram_persistent_patterns_20261018 import iemocap_source as adapter
        classes = 4 if args.dataset == 'IEMOCAPFour' else 6
        os.environ['GCNET_DATASET_ROOT'] = str(adapter.DATA_ROOT)
        os.environ.pop('GCNET_MELD_PICKLE', None)
        numerical_hashes = adapter.activate_numeric_source()
        sources = adapter.get_sources(classes)
        from gcnet_missing_m3 import train_gcnet as trainer
    else:
        from experiments.osram_meld_related_history_diagnostic_20261018.run import check_sources
        from experiments.osram_meld_optimizer_screen_20261018 import run as screen
        check_sources()
        os.environ.update(GCNET_MELD_PICKLE=str(screen.original.DATASET),
            GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'))
        if sha(screen.original.DATASET) != screen.original.DATA_SHA:
            raise ValueError('Accepted MELD feature hash changed')
        trainer = screen.original.load_historical(screen.MODEL_SOURCE)
        numerical_hashes = {'source': str(screen.MODEL_SOURCE), 'git_sha': screen.original.HISTORICAL}
        sources = meld_sources()
    import numpy as np
    import torch
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu, state_digest
    from experiments.osram_persistent_patterns_20261018.evaluate import PATTERNS, CHECKPOINT_RATE, evaluate_condition, classification_metrics
    from experiments.osram_persistent_patterns_20261018.analysis import aggregate_runs
    environment = require_replication_environment(torch)
    repair_uuid_rng(torch)
    gpu = guard_gpu(GPU)
    if gpu['host_index'] != 3 or gpu['uuid'] != GPU:
        raise ValueError('Require healthy physical GPU3')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    result = dict(status='running', dataset=args.dataset, training=False, runs=[],
        source_commit=args.commit, manifest=manifest, numerical_source=numerical_hashes,
        environment=environment, gpu=gpu, pid=os.getpid(), source_hashes={})
    hashes = result['source_hashes']
    tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    try:
        prior = None
        if args.dataset == 'MELD':
            prior_path = MELD_PRIOR/'RESULTS.json'
            hashes[str(prior_path)] = sha(prior_path)
            prior = json.loads(prior_path.read_text())
            if prior['status'] != 'complete' or not prior['source_inputs_unchanged']:
                raise ValueError('Original persistent singletons are incomplete')
        for case in sources:
            seed, fold, classes = case['seed'], case['fold'], case['classes']
            if args.dataset == 'MELD':
                from gcnet_modality_jepa.train_gcnet import get_loaders
                from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
                cfg = trainer.TrainConfig(**case['expected_config'])
                _, _, tests, *dimensions = get_loaders(audio_root='MELD-pickle', text_root='MELD-pickle',
                    video_root='MELD-pickle', dataset='MELD', num_folder=1, batch_size=32, num_workers=0,
                    seed=seed, validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
                model, loader = _build_model(cfg, dimensions, torch.device('cuda:0')), tests[0]
            else:
                model, loader, cfg, dimensions = adapter.build_model_and_loader(case, 'cuda:0', args.output/'feature_cache')
            model.eval().requires_grad_(False)
            row = dict(seed=seed, fold=fold, classes=classes, patterns={}, selected_replays={}, source=case)
            for rate in ('0.7', '0.3', '0.0'):
                ref = case['references'][rate]
                checkpoint_path = Path(ref['checkpoint'])
                if sha(checkpoint_path) != ref['checkpoint_sha256']:
                    raise ValueError('Checkpoint hash changed')
                hashes[str(checkpoint_path)] = ref['checkpoint_sha256']
                if prior is not None and rate == '0.7':
                    old = prior['seeds'][str(seed)]
                    if (old['source']['checkpoint_sha256'] != ref['checkpoint_sha256']
                            or prior['baseline_id'] != case['baseline_id']):
                        raise ValueError('Persistent singleton source differs from accepted baseline')
                    row['selected_replays'][rate] = dict(old['original_selected_replay'], reused=True)
                    for pattern in ('A', 'T', 'V'):
                        artifact_path = MELD_PRIOR/f'seed{seed}_{pattern}.npz'
                        hashes[str(artifact_path)] = sha(artifact_path)
                        with np.load(artifact_path, allow_pickle=False) as values:
                            actual = classification_metrics(values['labels'], values['logits'], classes)
                        if any(abs(actual[m]-old['patterns'][pattern]['metrics'][m]) > 1e-10 for m in ('ACC','UA','WF1','CE')):
                            raise ValueError('Reused singleton artifact metrics differ')
                        row['patterns'][pattern] = dict(old['patterns'][pattern], source_checkpoint=ref,
                            reused=True, artifact=str(artifact_path))
                    continue
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if checkpoint['config'] != case['expected_config'] or checkpoint['epoch'] != ref['epoch']:
                    raise ValueError('Wrong effective configuration or selected epoch')
                artifact_ref = checkpoint['selection_artifact']
                selected_path = (Path(case['output'])/artifact_ref['path']).resolve()
                if (not selected_path.is_relative_to(Path(case['output']).resolve())
                        or sha(selected_path) != ref['selection_sha256']
                        or artifact_ref['sha256'] != ref['selection_sha256']):
                    raise ValueError('Wrong exact selected artifact')
                hashes[str(selected_path)] = ref['selection_sha256']
                with np.load(selected_path, allow_pickle=False) as file:
                    saved = {key: file[key].copy() for key in file.files}
                if hashlib.sha256(saved['availability'].tobytes()).hexdigest() != ref['mask_sha256']:
                    raise ValueError('Selected availability hash differs')
                model.load_state_dict(checkpoint['model'], strict=True)
                before = state_digest(model)
                replay, replay_artifacts = evaluate_condition(model, loader, dimensions, saved, trainer, None, classes)
                for metric in ('ACC', 'UA', 'WF1'):
                    if abs(replay['metrics'][metric]-ref['expected_metrics'][metric]) > 1e-8:
                        raise ValueError('Original selected metrics do not reproduce')
                row['selected_replays'][rate] = replay
                for pattern in (p for p in PATTERNS if CHECKPOINT_RATE[p] == rate):
                    if pattern == 'ATV':
                        if not np.all(saved['availability'] == 1):
                            raise ValueError('Rate0 full-observation reference is not all visible')
                        value, predictions = dict(replay, full_reuses_selected_replay=True), replay_artifacts
                    else:
                        value, predictions = evaluate_condition(model, loader, dimensions, saved, trainer, pattern, classes)
                    path = args.output/f'seed{seed}_fold{fold}_{pattern}.npz'
                    np.savez_compressed(path, **predictions)
                    row['patterns'][pattern] = dict(value, source_checkpoint=ref, artifact=str(path), reused=False)
                    print(args.dataset, seed, fold, pattern, value['metrics'], flush=True)
                if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Frozen evaluation modified the model')
                del checkpoint
            result['runs'].append(row)
            write(args.output/'RESULTS.json', result)
            print('COMPLETE_UNIT', args.dataset, seed, fold, flush=True)
            del model
        result['aggregation'] = aggregate_runs(result['runs'], (1,) if args.dataset == 'MELD' else (1,2,3,4,5))
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('Read-only source input changed')
        result.update(status='complete', source_inputs_unchanged=True,
            seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output/'RESULTS.json', result)


if __name__ == '__main__':
    main()
