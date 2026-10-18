"""Read-only adapters for the completed original Flat IEMOCAP4/6 sources.

These remain internal per-rate Test-oracle checkpoints. Source discovery reads
retained metadata, never constructs a model, initializes CUDA, or samples masks.
The caller must hash the checkpoint before loading its model tensors, then
resolve its selection_artifact path and verify the selected NPZ and mask hashes.
Source discovery leaves this path unresolved to avoid reading weights twice.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_PATH = ROOT/'experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json'
TRAINING_SHA = '785a01242ef1a82b97d5337acbb33ba963df5537'
SOURCE_ROOT = Path('/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012')
DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
LABEL_HASHES = {4: '072cffa9c528dd0536792de7ac85d5a0efaf25d4a9e3ed01f7e45453547cc9fd',
               6: 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'}
DATASETS = {4: 'IEMOCAPFour', 6: 'IEMOCAPSix'}
RATES = ('0.0', '0.3', '0.7')
BUILDER = 'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py'
FLAT_CONFIG = dict(epochs=100, batch_size=32, optimizer='adam', learning_rate=.001,
    weight_decay=1e-5, dropout=.5, projector_dropout=.1, local_fusion_dropout=.2,
    lr_schedule='constant', gradient_clip_norm=1., emotion_loss_mode='sample-mean',
    training_objective='emotion-only', missing_mask_protocol='gcnet-random-mask-v1',
    train_rate_mode='cyclic', train_missing_rates=[i/10 for i in range(8)],
    checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy',
    preserve_selection_artifacts=True, evaluation_protocol='official',
    backbone_type='osram', osram_output_dim=1600, osram_adapter_hidden_dim=0,
    osram_readout_fusion='flat', osram_disable_local_skip=False,
    osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
    osram_bidirectional=False, osram_meaningful_block='none', osram_rpca_lr_multiplier=1.)
_ACTIVE_SOURCE = None
_NUMERICAL_HASHES = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config, classes, seed, fold):
    expected = dict(FLAT_CONFIG, dataset=DATASETS[classes], seed=seed, fold=fold)
    for key, value in expected.items():
        default = False if key in ('osram_disable_local_skip', 'osram_bidirectional') else None
        if config.get(key, default) != value:
            raise ValueError(f'Original Flat config mismatch: {key}')


def _reference(output, rate, expected, config):
    checkpoint = output/f'best_miss_{rate.replace(".", "p")}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=expected['checkpoint_sha256'],
        selection=None, selection_sha256=expected['selection_sha256'],
        mask_sha256=expected['mask_sha256'],
        epoch=expected['epoch'], expected_metrics=dict(expected), expected_config=dict(config))


def get_sources(classes: int) -> list[dict]:
    """Return all 15 retained sources, ordered seed then LOSO fold.

    Run on biggpu where the recorded absolute output paths exist. No path is
    synthesized for a selected NPZ: resolve it from the loaded checkpoint.
    Checkpoint/NPZ SHA verification is the caller's pre-inference responsibility.
    """
    if type(classes) is not int or classes not in DATASETS:
        raise ValueError('IEMOCAP classes must be 4 or 6')
    aggregate = json.loads(AGGREGATE_PATH.read_text())
    jobs = aggregate['jobs']
    all_ids = {f'iemocap{c}_Original_Flat_seed{s}_fold{f}'
               for c in DATASETS for s in (66, 67, 68) for f in range(1, 6)}
    if (aggregate.get('status') != 'complete' or aggregate.get('completed_fold_runs') != 30
            or len(jobs) != 30 or {job['id'] for job in jobs} != all_ids
            or any(job.get('status') != 'complete' or not job.get('summary_valid')
                   or job.get('failures') for job in jobs)):
        raise ValueError('Require all 30 completed original Flat folds')
    by_id = {job['id']: job for job in jobs}
    dataset = DATASETS[classes]
    sources = []
    for seed in (66, 67, 68):
        for fold in range(1, 6):
            job = by_id[f'iemocap{classes}_Original_Flat_seed{seed}_fold{fold}']
            output = Path(job['output']).resolve(strict=True)
            summary = json.loads((output/'SUMMARY.json').read_text())
            config = json.loads((output/'config.json').read_text())
            if (summary.get('status') != 'complete' or summary.get('method') != 'Original_Flat'
                    or summary.get('smoke_only') is not False
                    or (summary.get('dataset'), summary.get('seed'), summary.get('fold')) != (dataset, seed, fold)
                    or summary['identity']['source_git_sha'] != TRAINING_SHA
                    or summary['identity']['label_sha256'] != LABEL_HASHES[classes]):
                raise ValueError('Retained source identity is not completed original Flat')
            # The trainer writes constructor-expanded config.json; the worker
            # identity hashes the original launch config. Preserve both file
            # identities and require the exact recorded effective payload.
            if (config != summary['effective_config']
                    or any(config.get(key) != value for key, value in summary['config'].items())):
                raise ValueError('Retained source config differs from its training identity')
            _validate_config(config, classes, seed, fold)
            environment = summary['provenance']['environment']
            if environment.get('torch') != '1.13.1+cu117' or environment.get('numpy') != '1.26.4':
                raise ValueError('Retained source does not use accepted cmerc-repro numerics')
            references = {}
            for rate in RATES:
                expected = aggregate['datasets'][dataset]['folds'][str(seed)][str(fold)]['rates'][rate]
                if (summary['testbest']['rates'][rate] != expected
                        or len(expected['recall']) != classes or len(expected['support']) != classes):
                    raise ValueError('Retained per-class metrics differ from completed aggregate')
                references[rate] = _reference(output, rate, expected, config)
            sources.append(dict(dataset=dataset, classes=classes, seed=seed, fold=fold,
                output=str(output), source_root=str(SOURCE_ROOT), references=references,
                expected_config=config, effective_config=summary['effective_config'],
                source_git_sha=TRAINING_SHA, source_summary_sha256=sha256(output/'SUMMARY.json'),
                source_config_sha256=sha256(output/'config.json'),
                launch_config_sha256=summary['identity']['config_sha256'],
                aggregate_sha256=sha256(AGGREGATE_PATH), label_sha256=LABEL_HASHES[classes],
                feature_roots=summary['provenance'].get('feature_roots'),
                protocol='INTERNAL TEST-ORACLE; full .0, pairs .3, singles .7'))
    return sources


def activate_numeric_source(source=SOURCE_ROOT):
    """Bind original packages before numerical imports; reuse tested activation."""
    global _ACTIVE_SOURCE, _NUMERICAL_HASHES
    source = Path(source).resolve(strict=True)
    if _ACTIVE_SOURCE is not None:
        if source != _ACTIVE_SOURCE:
            raise ValueError('Cannot mix numerical checkouts in one process')
        for name in ('config', 'gcnet_missing_m3', 'gcnet_modality_jepa'):
            module = sys.modules.get(name)
            if module is None or not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError('Numerical package binding changed: '+name)
        return dict(_NUMERICAL_HASHES)
    from experiments.osram_iemocap_information_availability_20261018.run import activate_source
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', BUILDER],
                               cwd=source, text=True).strip():
        raise ValueError('Original model builder has tracked edits')
    hashes = activate_source(source)
    _ACTIVE_SOURCE, _NUMERICAL_HASHES = source, dict(hashes, **{BUILDER: sha256(source/BUILDER)})
    return dict(_NUMERICAL_HASHES)


def build_model_and_loader(source, device, cache_root):
    """Return (frozen model, held-out loader, original config, dimensions).

    No checkpoint is loaded and no batch is evaluated. The caller loads each
    already verified reference strictly before frozen inference.
    """
    os.environ['GCNET_DATASET_ROOT'] = str(DATA_ROOT)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(cache_root).resolve())
    os.environ.pop('GCNET_MELD_PICKLE', None)
    numerical_root = Path(source['source_root']).resolve(strict=True)
    activate_numeric_source(numerical_root)
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment
    require_replication_environment(torch)
    classes, seed, fold = source['classes'], source['seed'], source['fold']
    config = TrainConfig(**source['expected_config'])
    _validate_config(asdict(config), classes, seed, fold)
    if asdict(config) != source['effective_config'] or _dataset_shape(config.dataset)['num_classes'] != classes:
        raise ValueError('Original effective config or IEMOCAP class count changed')
    label = DATA_ROOT/'IEMOCAP'/f'IEMOCAP_features_raw_{classes}way.pkl'
    if sha256(label) != LABEL_HASHES[classes]:
        raise ValueError('Original IEMOCAP label source changed')
    roots = [str(DATA_ROOT/'IEMOCAP/features'/name)
             for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    if source.get('feature_roots') != roots or not all(Path(root).is_dir() for root in roots):
        raise ValueError('Original feature roots changed or unavailable')
    train, _, test, *dimensions = get_loaders(*roots, num_folder=5, dataset=config.dataset,
        batch_size=config.batch_size, num_workers=0, seed=seed,
        validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol)
    if len(train) != 5 or len(test) != 5:
        raise ValueError('Original IEMOCAP requires five LOSO folds')
    train_loader, test_loader = train[fold-1], test[fold-1]
    train_ids = {train_loader.dataset.vids[i] for i in train_loader.sampler.indices}
    test_ids = {test_loader.dataset.vids[i] for i in test_loader.sampler.indices}
    if train_ids & test_ids:
        raise ValueError('Original LOSO train/test conversation overlap')
    # Load the original builder itself by file path, bypassing the current
    # experiments package while preserving its already bound numeric imports.
    spec = importlib.util.spec_from_file_location('_persistent_original_iemocap_builder', numerical_root/BUILDER)
    builder = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    try:
        spec.loader.exec_module(builder)
    finally:
        sys.path[:] = previous_path
    model = builder._build_model(config, dimensions, torch.device(device))
    return model, test_loader, config, tuple(dimensions)
