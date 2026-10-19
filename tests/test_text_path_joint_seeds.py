"""Seed expansion must not change the established split or evaluation masks."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


class Conversations(torch.utils.data.Dataset):
    def __len__(self):
        return 30

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(rows):
        return rows


def bundle_fixture():
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import _build_protocol_loader
    loader = _build_protocol_loader(Conversations(), list(range(20)), 'MELD', 1,
        'train', 4, 0, SeedBundle(66), 'unchanged-split', 'official')
    return dict(train_loader=loader, validation_loader=object(), test_loader=object(),
        cfg=SimpleNamespace(dataset='MELD'), case={'fold': 1},
        split={'train_indices': list(range(20)), 'validation_indices': [20], 'test_indices': [21]})


def test_default_seed_keeps_original_loader_and_rng():
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundle = bundle_fixture()
    old_loader = bundle['train_loader']
    before = torch.get_rng_state().clone()
    metadata = seed_training_loader(bundle)
    assert bundle['train_loader'] is old_loader
    assert torch.equal(torch.get_rng_state(), before)
    assert metadata['training_seed'] == metadata['split_seed'] == metadata['evaluation_seed'] == 66
    assert metadata['train_loader'] == old_loader.protocol_metadata


@pytest.mark.parametrize('seed', (67, 68))
def test_seed_changes_only_training_order_and_preserves_split_and_eval(seed):
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundle = bundle_fixture()
    old = bundle['train_loader']
    validation, test, split = bundle['validation_loader'], bundle['test_loader'], bundle['split']
    before = torch.get_rng_state().clone()
    metadata = seed_training_loader(bundle, seed)
    new = bundle['train_loader']
    assert new.dataset is old.dataset
    assert new.sampler.indices == old.sampler.indices
    assert list(new.sampler) != list(old.sampler)
    assert new.batch_size == old.batch_size and new.num_workers == old.num_workers
    assert new.protocol_metadata['split_hash'] == old.protocol_metadata['split_hash']
    assert new.protocol_metadata['evaluation_protocol'] == old.protocol_metadata['evaluation_protocol']
    assert bundle['validation_loader'] is validation and bundle['test_loader'] is test
    assert bundle['split'] is split
    assert torch.equal(torch.get_rng_state(), before)
    assert metadata['training_seed'] == seed
    assert metadata['split_seed'] == metadata['evaluation_seed'] == 66
    assert metadata['train_loader'] == new.protocol_metadata


def test_expanded_training_seed_is_reproducible_and_distinct_between_67_68():
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundles = [bundle_fixture() for _ in range(3)]
    for bundle, seed in zip(bundles, (67, 67, 68)):
        seed_training_loader(bundle, seed)
    for bundle in bundles:
        bundle['train_loader'].sampler.set_epoch(7)
    orders = [list(bundle['train_loader'].sampler) for bundle in bundles]
    assert orders[0] == orders[1] and orders[0] != orders[2]


@pytest.mark.parametrize('seed', (0, 65, 69, 666))
def test_unapproved_seed_rejected_without_mutation(seed):
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundle = bundle_fixture()
    loader = bundle['train_loader']
    with pytest.raises(ValueError, match='66.*67.*68'):
        seed_training_loader(bundle, seed)
    assert bundle['train_loader'] is loader


@dataclass
class Config:
    dataset: str = 'MELD'
    seed: int = 66
    learning_rate: float = 1e-5
    epochs: int = 50
    weight_decay: float = 1e-5
    checkpoint_selection: str = 'test-oracle-per-rate'
    checkpoint_selection_metric: str = 'accuracy'
    evaluate_test: bool = True


@pytest.mark.parametrize('seed', (66, 67, 68))
def test_fresh_factory_receives_requested_seed_and_default_matches_66(monkeypatch, seed):
    from experiments.osram_text_path_router_joint_20261019 import run
    from experiments.osram_meld_flat_source_combinations_20261013 import evaluate
    bundle = bundle_fixture()
    bundle.update(cfg=Config(), model=torch.nn.Linear(2, 2), dimensions=(2, 2, 2))
    seen = []

    def build(cfg, dimensions, device):
        seen.append(cfg)
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(evaluate, '_build_model', build)
    monkeypatch.setattr(run, 'wrap_model', lambda model, mode: model)
    old_cfg = bundle['cfg']
    if seed == 66:
        run.prepare_fresh(bundle, 'flat')
    else:
        run.prepare_fresh(bundle, 'flat', seed=seed)
    assert seen[0].seed == seed and bundle['cfg'].seed == seed
    assert old_cfg.seed == 66
    assert bundle['cfg'].checkpoint_selection == 'validation'
    assert bundle['seed_protocol']['training_seed'] == seed


def test_queue_plans_exactly_eight_new_jobs_without_seed66(tmp_path):
    from experiments.osram_text_path_router_3seed_20261019.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'source', ['GPU-meld', 'GPU-iemo'])
    assert len(jobs) == 8
    assert {(j['dataset'], j['seed'], j['mode']) for j in jobs} == {
        (d, s, m) for d in ('MELD', 'IEMOCAPSix') for s in (67, 68) for m in ('flat', 'joint')}
    assert sum(j['gpu_uuid'] == 'GPU-meld' for j in jobs) == 4
    assert sum(j['gpu_uuid'] == 'GPU-iemo' for j in jobs) == 4
    assert all('--seed' in j['command'] and '--resume' not in j['command'] for j in jobs)
    assert len({j['output'] for j in jobs}) == 8


def test_queue_memory_preflight_respects_peak_margin():
    from experiments.osram_text_path_router_3seed_20261019.run import memory_requirements
    good = {'GPU-meld': {'total_mib': 32768, 'used_mib': 0},
            'GPU-iemo': {'total_mib': 32768, 'used_mib': 0}}
    requirements = memory_requirements(good, ['GPU-meld', 'GPU-iemo'])
    assert 24000 < requirements['IEMOCAPSix']['required_free_mib'] < 28000
    good['GPU-iemo']['used_mib'] = 16000
    with pytest.raises(RuntimeError, match='IEMOCAPSix'):
        memory_requirements(good, ['GPU-meld', 'GPU-iemo'])


def test_queue_reuses_only_completed_seed66_with_unchanged_core(tmp_path, monkeypatch):
    import hashlib
    import json
    from experiments.osram_text_path_router_3seed_20261019 import run
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.py').write_text('unchanged model')
    expected = hashlib.sha256((source / 'model.py').read_bytes()).hexdigest()
    monkeypatch.setattr(run, 'ROOT', source)
    references = tmp_path / 'references'
    for dataset in run.DATASETS:
        for mode in ('flat', 'joint'):
            path = references / dataset / mode / 'SUMMARY.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='complete', seed=66, mode=mode,
                dataset=dataset, source_commit='old', identity=dict(source_hashes={
                    'model.py': expected,
                    'experiments/osram_text_path_router_joint_20261019/run.py': 'old-wrapper'},
                    config={'seed': 66}, validation_mask_hashes={'0.3': 'val'},
                    test_masks={'0.3': 'test'}, split={'train_indices': [0]}))))
    result = run.verify_seed66(references)
    assert result['MELD']['flat']['config']['seed'] == 66
    (source / 'model.py').write_text('changed algorithm')
    with pytest.raises(ValueError, match='Changed seed66 algorithm source'):
        run.verify_seed66(references)
