"""CPU metadata checks for the retained original IEMOCAP Flat sources."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def adapter():
    return importlib.import_module(
        'experiments.osram_persistent_patterns_20261018.iemocap_source')


@pytest.fixture
def retained_sources(tmp_path, monkeypatch):
    module = adapter()
    aggregate = dict(status='complete', completed_fold_runs=30, jobs=[], datasets={})
    for classes, dataset in ((4, 'IEMOCAPFour'), (6, 'IEMOCAPSix')):
        folds = {}
        for seed in (66, 67, 68):
            folds[str(seed)] = {}
            for fold in range(1, 6):
                output = tmp_path / f'original_{classes}_{seed}_{fold}'
                output.mkdir()
                config = dict(module.FLAT_CONFIG, dataset=dataset, seed=seed, fold=fold)
                (output / 'config.json').write_text(json.dumps(config))
                rates = {}
                for rate in module.RATES:
                    artifact = output / 'selection_versions' / f'actual_selected_{rate}.npz'
                    artifact.parent.mkdir(exist_ok=True)
                    artifact.write_bytes(b'retained selection fixture')
                    row = dict(ACC=75., UA=72., WF1=74., CE=.6, N=100, epoch=8,
                               recall=[72.] * classes, support=[25] * classes,
                               selection_sha256=module.sha256(artifact), mask_sha256='a' * 64)
                    checkpoint = output / f'best_miss_{rate.replace(".", "p")}.pt'
                    checkpoint.write_bytes(b'weights are never loaded during source discovery')
                    row['checkpoint_sha256'] = module.sha256(checkpoint)
                    rates[rate] = row
                folds[str(seed)][str(fold)] = dict(rates=copy.deepcopy(rates))
                summary = dict(status='complete', method='Original_Flat', smoke_only=False,
                    dataset=dataset, seed=seed, fold=fold, config=config, effective_config=config,
                    testbest=dict(rates=rates), identity=dict(source_git_sha=module.TRAINING_SHA,
                        config_sha256=module.sha256(output/'config.json'),
                        label_sha256=module.LABEL_HASHES[classes]),
                    provenance=dict(environment=dict(torch='1.13.1+cu117', numpy='1.26.4')))
                (output/'SUMMARY.json').write_text(json.dumps(summary))
                aggregate['jobs'].append(dict(id=f'iemocap{classes}_Original_Flat_seed{seed}_fold{fold}',
                    output=str(output), status='complete', summary_valid=True, failures=[]))
        aggregate['datasets'][dataset] = dict(folds=folds)
    path = tmp_path/'aggregate.json'
    path.write_text(json.dumps(aggregate))
    monkeypatch.setattr(module, 'AGGREGATE_PATH', path)
    return module, aggregate, path


@pytest.mark.parametrize('classes', [4, 6])
def test_sources_cover_exact_three_seed_fivefold_and_defer_artifact_paths(retained_sources, classes):
    module, _, _ = retained_sources
    sources = module.get_sources(classes)
    assert [(s['seed'], s['fold']) for s in sources] == [
        (seed, fold) for seed in (66, 67, 68) for fold in range(1, 6)]
    for source in sources:
        assert source['classes'] == classes
        assert set(source['references']) == {'0.0', '0.3', '0.7'}
        assert source['expected_config']['osram_meaningful_block'] == 'none'
        for rate, reference in source['references'].items():
            assert reference['selection'] is None
            assert len(reference['expected_metrics']['recall']) == classes
            artifact = Path(source['output'])/'selection_versions'/f'actual_selected_{rate}.npz'
            assert reference['selection_sha256'] == module.sha256(artifact)


def test_sources_reject_missing_fold_instead_of_partial_average(retained_sources):
    module, report, path = retained_sources
    report['jobs'].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='30|fold'):
        module.get_sources(4)


def test_sources_reject_retained_result_that_differs_from_completed_aggregate(retained_sources):
    module, report, _ = retained_sources
    path = Path(report['jobs'][0]['output'])/'SUMMARY.json'
    summary = json.loads(path.read_text())
    summary['testbest']['rates']['0.7']['UA'] = 99.
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='aggregate'):
        module.get_sources(4)


def test_sources_reject_changed_source_config(retained_sources):
    module, report, _ = retained_sources
    path = Path(report['jobs'][0]['output'])/'config.json'
    config = json.loads(path.read_text())
    config['osram_meaningful_block'] = 'other'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='config'):
        module.get_sources(4)


def test_retained_config_format_may_differ_from_original_launch_config(retained_sources):
    module, report, _ = retained_sources
    path = Path(report['jobs'][0]['output'])/'config.json'
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
    sources = module.get_sources(4)
    assert sources[0]['source_config_sha256'] == module.sha256(path)
    assert sources[0]['launch_config_sha256'] != sources[0]['source_config_sha256']


def test_retained_config_contains_explicit_original_constructor_defaults(retained_sources):
    module, report, _ = retained_sources
    output = Path(report['jobs'][0]['output'])
    summary = json.loads((output/'SUMMARY.json').read_text())
    summary['config'].pop('osram_disable_local_skip')
    (output/'SUMMARY.json').write_text(json.dumps(summary))
    sources = module.get_sources(4)
    assert sources[0]['expected_config']['osram_disable_local_skip'] is False


@pytest.mark.parametrize('classes', [3, 7, True])
def test_unsupported_class_counts_are_rejected(classes):
    with pytest.raises(ValueError, match='4 or 6'):
        adapter().get_sources(classes)


def test_numeric_activation_reuses_existing_binding_and_rejects_other_checkout(tmp_path, monkeypatch):
    module = adapter()
    helper = importlib.import_module('experiments.osram_iemocap_information_availability_20261018.run')
    builder = tmp_path/module.BUILDER
    builder.parent.mkdir(parents=True)
    builder.write_text('# original model builder\n')
    calls = []
    def activate(root):
        calls.append(root)
        for name in ('config', 'gcnet_missing_m3', 'gcnet_modality_jepa'):
            bound = ModuleType(name)
            bound.__file__ = str(root/f'{name}.py')
            monkeypatch.setitem(sys.modules, name, bound)
        return {'original.py': 'original hash'}
    monkeypatch.setattr(helper, 'activate_source', activate)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(module, '_ACTIVE_SOURCE', None)
    monkeypatch.setattr(module, '_NUMERICAL_HASHES', None)
    assert module.activate_numeric_source(tmp_path) == module.activate_numeric_source(tmp_path)
    assert calls == [tmp_path]
    alternate = tmp_path/'other'
    alternate.mkdir()
    with pytest.raises(ValueError, match='mix numerical'):
        module.activate_numeric_source(alternate)


@pytest.mark.parametrize('classes', [4, 6])
def test_builder_uses_original_file_and_original_fivefold_loader(retained_sources, tmp_path, monkeypatch, classes):
    module, _, _ = retained_sources
    source = module.get_sources(classes)[-1]
    numerical_root = tmp_path/'original_numeric'
    builder = numerical_root/module.BUILDER
    builder.parent.mkdir(parents=True)
    builder.write_text('def _build_model(config, dimensions, device):\n'
                       '    return (__file__, config.dataset, dimensions, device)\n')
    source['source_root'] = str(numerical_root)
    for key in ('expected_config', 'effective_config'):
        source[key]['validation_fraction'] = .1
    data = tmp_path/'data'
    label = data/'IEMOCAP'/f'IEMOCAP_features_raw_{classes}way.pkl'
    label.parent.mkdir(parents=True)
    label.write_bytes(b'original labels')
    feature_roots = [data/'IEMOCAP/features'/name for name in
                     ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    for root in feature_roots:
        root.mkdir(parents=True)
    source['feature_roots'] = list(map(str, feature_roots))
    monkeypatch.setattr(module, 'DATA_ROOT', data)
    monkeypatch.setitem(module.LABEL_HASHES, classes, module.sha256(label))
    activation = []
    monkeypatch.setattr(module, 'activate_numeric_source', lambda root: activation.append(root))
    monkeypatch.setattr(module, 'asdict', vars)
    fake_torch = ModuleType('torch')
    fake_torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    trainer = ModuleType('gcnet_missing_m3.train_gcnet')
    trainer.TrainConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    trainer._dataset_shape = lambda dataset: {'num_classes': classes}
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)
    def loader(identifier):
        return SimpleNamespace(dataset=SimpleNamespace(vids=[identifier]),
                               sampler=SimpleNamespace(indices=[0]))
    train = [loader(f'train{i}') for i in range(5)]
    test = [loader(f'test{i}') for i in range(5)]
    calls = []
    def get_loaders(*roots, **kwargs):
        calls.append((roots, kwargs))
        return train, [], test, 10, 20, 30
    loaders = ModuleType('gcnet_modality_jepa.train_gcnet')
    loaders.get_loaders = get_loaders
    monkeypatch.setitem(sys.modules, loaders.__name__, loaders)
    worker = ModuleType('experiments.osram_iemocap_b6_transfer_20261015.worker')
    worker.require_replication_environment = lambda torch: None
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    model, test_loader, config, dimensions = module.build_model_and_loader(source, 'cpu', tmp_path/'cache')
    assert activation == [numerical_root]
    assert model == (str(builder), source['dataset'], [10, 20, 30], 'cpu')
    assert test_loader is test[4]
    assert dimensions == (10, 20, 30)
    assert (config.seed, config.fold) == (68, 5)
    assert calls[0][1] == dict(num_folder=5, dataset=source['dataset'], batch_size=32,
                               num_workers=0, seed=68, validation_fraction=.1,
                               evaluation_protocol='official')
