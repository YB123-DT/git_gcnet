"""Finite classification V3 screen keeps original Flat data and ACC selection."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def worker():
    path = ROOT / 'experiments/osram_v3_classification_20261021/worker.py'
    assert path.exists(), 'The classification V3 worker is not implemented'
    from experiments.osram_v3_classification_20261021 import worker
    return worker


@pytest.fixture
def baseline_root(tmp_path):
    from experiments.osram_tasr_20261018.protocol import build_cases
    template = build_cases()['IEMOCAPSix']['expected_config']
    for classes, dataset in ((4, 'IEMOCAPFour'), (6, 'IEMOCAPSix')):
        for fold in range(1, 6):
            path = tmp_path / f'iemocap{classes}/seed_66/fold_{fold}/config.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(template, dataset=dataset, fold=fold)))
    return tmp_path


@pytest.mark.parametrize('dataset,epochs,lr,folds', (
    ('IEMOCAPFour', 100, .001, range(1, 6)),
    ('IEMOCAPSix', 100, .001, range(1, 6)), ('MELD', 50, 1e-5, (1,))))
def test_configs_keep_accepted_flat_protocol_and_enable_only_v3(baseline_root, dataset, epochs, lr, folds):
    for fold in folds:
        cfg, reference = worker().make_config(dataset, fold, 66, baseline_root=baseline_root)
        assert (cfg.dataset, cfg.fold, cfg.seed) == (dataset, fold, 66)
        assert (cfg.epochs, cfg.learning_rate, cfg.weight_decay, cfg.batch_size) == (epochs, lr, 1e-5, 32)
        assert cfg.osram_source_role_v3 and not cfg.osram_source_role and not cfg.osram_source_relation
        assert not cfg.osram_tasr and not cfg.osram_disable_local_skip
        assert cfg.osram_readout_fusion == 'flat' and cfg.osram_output_dim == 1600
        assert cfg.osram_write_step == .6 and cfg.lr_schedule == 'constant'
        assert cfg.missing_mask_protocol == 'gcnet-random-mask-v1'
        assert cfg.checkpoint_selection == 'test-oracle-per-rate' and cfg.checkpoint_selection_metric == 'accuracy'
        assert tuple(cfg.train_missing_rates) == tuple(i / 10 for i in range(8)) and cfg.train_rate_mode == 'cyclic'
        assert reference['config_diff']['osram_source_role_v3'] == {'original': False, 'effective': True}
        assert reference['original_config']['dataset'] == dataset
        assert reference['source_sha256'] == hashlib.sha256(Path(reference['source']).read_bytes()).hexdigest()


@pytest.mark.parametrize('dataset,fold,seed', (('CMUMOSEI', 1, 66), ('MELD', 2, 66),
    ('IEMOCAPFour', 0, 66), ('IEMOCAPSix', 6, 66), ('MELD', 1, 67)))
def test_out_of_budget_jobs_rejected(dataset, fold, seed):
    with pytest.raises(ValueError):
        worker().make_config(dataset, fold, seed)


def test_changed_original_protocol_is_rejected(baseline_root):
    path = baseline_root / 'iemocap4/seed_66/fold_1/config.json'
    cfg = json.loads(path.read_text())
    cfg['osram_write_step'] = .3
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Flat'):
        worker().make_config('IEMOCAPFour', 1, 66, baseline_root=baseline_root)


def evaluation(acc, ua, marker):
    availability = np.array([[1., 0., 0.]], dtype=np.float32)
    return (dict(router=dict(ACC=acc, UA=ua, WF1=ua + 1., CE=.5),
        mask_sha256=hashlib.sha256(availability.tobytes()).hexdigest()),
        dict(labels=np.array([0]), availability=availability,
             log_probs_router=np.array([[marker, -marker]], dtype=np.float32)))


def test_acc_selection_keeps_same_call_and_full_recovery_state(tmp_path):
    w = worker()
    model = torch.nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    best = {}
    identity = dict(config=dict(osram_source_role_v3=True))
    w.persist_epoch(tmp_path, model, optimizer, epoch=1, identity=identity, history=[{'epoch': 1}], best=best,
        evaluated={r: evaluation(60., 30., 1.) for r in w.RATES})
    rows = {r: evaluation(60., 99., 2.) for r in w.RATES}
    rows['0.3'] = evaluation(61., 25., 3.)
    w.persist_epoch(tmp_path, model, optimizer, epoch=2, identity=identity, history=[{'epoch': 1}, {'epoch': 2}],
        best=best, evaluated=rows)
    assert best['0.3']['epoch'] == 2 and best['0.3']['metrics']['router']['UA'] == 25.
    assert best['0.0']['epoch'] == 1 and best['0.0']['metrics']['router']['UA'] == 30.
    state = torch.load(tmp_path / 'last.pt', map_location='cpu', weights_only=False)
    assert state['best_by_rate'] == best and state['optimizer']['state']
    assert {'rng', 'model', 'optimizer', 'identity', 'history', 'epoch'} <= state.keys()
    for row in best.values():
        selected = torch.load(tmp_path / row['checkpoint'], map_location='cpu', weights_only=False)
        assert selected['optimizer']['state'] and selected['selection']['same_evaluation_call']
    w.validate_selected(tmp_path, best)


def test_source_catalog_covers_current_v3_and_reused_training_factories():
    paths = worker().source_catalog()
    assert len(paths) == len(set(paths))
    for name in ('gcnet_missing_m3/source_role_v3.py', 'gcnet_missing_m3/model.py',
        'experiments/osram_text_path_router_joint_20261019/run.py',
        'experiments/osram_v3_classification_20261021/worker.py'):
        assert name in paths and (ROOT / name).is_file()


def test_selected_mask_replay_replaces_only_valid_slots_and_requires_same_cohort():
    row = dict(labels=np.array([0, 1]), speakers=np.array([0, 1]),
        conversation_ids=np.array(['c', 'c']), utterance_indices=np.array([0, 1]),
        availability=torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 0.]]]))
    accepted = {key: value for key, value in row.items() if key != 'availability'}
    accepted['availability'] = np.array([[0., 0., 1.], [1., 1., 0.]], dtype=np.float32)
    assert hasattr(worker(), 'replay_selected_bank'), 'Accepted exact-mask replay is missing'
    replayed = worker().replay_selected_bank([row], accepted)
    assert torch.equal(replayed[0]['availability'][:2, 0], torch.tensor(accepted['availability']))
    assert replayed[0]['availability'][2].sum() == 0
    assert torch.equal(row['availability'][0, 0], torch.tensor([1., 0., 0.]))
    wrong = dict(accepted, labels=np.array([1, 0]))
    with pytest.raises(AssertionError):
        worker().replay_selected_bank([row], wrong)
