"""Regression contract for the finite, original-Flat MELD optimizer screen."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
EXPECTED_SPECS = {
    'O0': (2e-4, 2e-4), 'O1': (2e-4, 1e-3),
    'O2': (1e-4, 2e-4), 'O3': (1e-4, 1e-3),
    'O4': (3e-5, 2e-4), 'O5': (3e-5, 1e-3),
    'O6': (1e-5, 2e-4), 'O7': (1e-5, 1e-3),
}


def runner():
    name = 'experiments.osram_meld_optimizer_screen_20261018.run'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and name.startswith(error.name):
            pytest.fail('Finite MELD optimizer screen is not implemented')
        raise


def baseline():
    path = ROOT / 'experiments/osram_meld_memory_sensitivity_20261017/FROZEN_SUMMARY.json'
    return json.loads(path.read_text())['source_config']


def test_exact_eight_unique_cartesian_configs_only_change_lr_and_weight_decay():
    run = runner()
    assert run.JOB_SPECS == EXPECTED_SPECS
    source = baseline()
    original = copy.deepcopy(source)
    configs = {job: run.make_config(source, job) for job in EXPECTED_SPECS}
    assert source == original, 'Configuration construction must not mutate the reference'
    assert len({(cfg['learning_rate'], cfg['weight_decay']) for cfg in configs.values()}) == 8
    for job, cfg in configs.items():
        lr, wd = EXPECTED_SPECS[job]
        assert cfg == dict(original, learning_rate=lr, weight_decay=wd)
        assert cfg['seed'] == 66 and cfg['epochs'] == 50 and cfg['batch_size'] == 32
        assert cfg['lr_schedule'] == 'constant'
        assert cfg['osram_write_step'] == .6
        assert cfg['checkpoint_selection'] == 'test-oracle-per-rate'


@pytest.mark.parametrize('job', ['O8', 'W0', 'O-1', '', None])
def test_unapproved_optimizer_jobs_are_rejected(job):
    with pytest.raises(ValueError):
        runner().make_config(baseline(), job)


@pytest.mark.parametrize('change', [
    {'missing_mask_protocol': 'conversation-mask-v2'},
    {'missing_mask_protocol': 'gcnet-legacy-fixed-v1'},
    {'lr_schedule': 'cosine'},
    {'seed': 67}, {'epochs': 100}, {'batch_size': 64},
    {'optimizer': 'adamw'}, {'dropout': .2},
    {'learning_rate': 1e-4}, {'weight_decay': 1e-3},
    {'train_rate_mode': 'fixed', 'fixed_missing_rate': .7},
    {'training_objective': 'future-state'},
    {'checkpoint_selection': 'validation'},
    {'osram_disable_local_skip': True},
    {'osram_meaningful_block': 'mbt_role_bottleneck'},
    {'initial_backbone_checkpoint': '/some/pretrained/model.pt'},
    {'backbone_lr_multiplier': .1},
    {'osram_write_step': .3},
])
def test_changed_source_protocol_cannot_silently_enter_the_screen(change):
    with pytest.raises(ValueError):
        runner().make_config(dict(baseline(), **change), 'O0')


def optimizer_config():
    return dict(learning_rate=2e-4, weight_decay=1e-3,
                backbone_lr_multiplier=.5, projector_lr_multiplier=2.,
                classifier_lr_multiplier=3., lr_schedule='constant')


def optimizer_groups():
    return [dict(name='backbone', lr=1e-4, weight_decay=1e-3),
            dict(name='projector', lr=4e-4, weight_decay=1e-3),
            dict(name='classifier', lr=6e-4, weight_decay=1e-3),
            dict(name='predictor', lr=2e-4, weight_decay=1e-3),
            dict(name='other', lr=2e-4, weight_decay=1e-3)]


def test_optimizer_check_reads_actual_group_lrs_and_decay_without_modifying_them():
    groups = optimizer_groups()
    previous = copy.deepcopy(groups)
    result = runner().constant_optimizer_check(groups, optimizer_config())
    assert result['rates'] == pytest.approx([1e-4, 4e-4, 6e-4, 2e-4, 2e-4])
    assert result['weight_decay'] == [1e-3] * 5
    assert groups == previous


def test_old_trainer_unnamed_groups_are_audited_by_parameter_ownership_without_mutation():
    parameters = [object(), object(), object()]
    names = ['osram.key.weight', 'observed_set.projector.weight', 'smax_fc.weight']
    model = SimpleNamespace(named_parameters=lambda:zip(names,parameters))
    groups = [dict(params=[p],lr=1e-5,weight_decay=2e-4) for p in parameters]
    observed = runner().audit_optimizer_groups(model,groups)
    assert [g['name'] for g in observed] == ['backbone','projector','classifier']
    assert all('name' not in group for group in groups)
    assert [g['params'] for g in observed] == [g['params'] for g in groups]


def test_mixed_or_foreign_parameter_groups_are_rejected():
    a,b=object(),object()
    model = SimpleNamespace(named_parameters=lambda:iter([('osram.a',a),('smax_fc.b',b)]))
    with pytest.raises(ValueError):
        runner().audit_optimizer_groups(model,[dict(params=[a,b])])
    with pytest.raises(ValueError):
        runner().audit_optimizer_groups(model,[dict(params=[object()])])


@pytest.mark.parametrize('index, field, value', [
    (0, 'lr', 2e-4), (1, 'lr', 2e-4), (2, 'lr', 2e-4),
    (3, 'lr', 1e-4), (4, 'lr', 0.),
    (0, 'weight_decay', 1e-5), (4, 'weight_decay', 0.),
    (0, 'lr', float('nan')), (1, 'weight_decay', float('inf')),
    (0, 'name', 'unknown_group'),
])
def test_optimizer_check_rejects_actual_lr_or_decay_mismatch(index, field, value):
    groups = optimizer_groups()
    groups[index][field] = value
    with pytest.raises(ValueError):
        runner().constant_optimizer_check(groups, optimizer_config())


def test_optimizer_check_rejects_missing_groups_or_group_names():
    with pytest.raises(ValueError):
        runner().constant_optimizer_check([], optimizer_config())
    groups = optimizer_groups()
    del groups[0]['name']
    with pytest.raises(ValueError):
        runner().constant_optimizer_check(groups, optimizer_config())


def mask_trace():
    def row(name, count, batches):
        return dict(sha256=hashlib.sha256(name.encode()).hexdigest(), N=count, batches=batches)

    return dict(seed=66,
        train={'1': row('actual-training-masks', 9989, 33)},
        selection={'1': {f'{i / 10:.1f}': row(f'actual-test-masks-{i}', 2610, 9)
                         for i in range(8)}})


def test_actual_mask_trace_can_match_despite_unrelated_score_differences():
    reference = mask_trace()
    trace = copy.deepcopy(reference)
    trace['accuracy'] = .1
    reference['accuracy'] = .9
    original = copy.deepcopy(trace)
    runner().assert_epoch_masks(trace, reference, 1)
    assert trace == original


@pytest.mark.parametrize('changed', ['train_sha', 'selection_sha', 'train_count',
                                  'selection_count', 'missing_train', 'missing_rate',
                                  'missing_reference_epoch'])
def test_same_seed_is_not_sufficient_for_actual_mask_parity(changed):
    trace, reference = mask_trace(), mask_trace()
    if changed == 'train_sha':
        trace['train']['1']['sha256'] = hashlib.sha256(b'different-train-mask').hexdigest()
    elif changed == 'selection_sha':
        trace['selection']['1']['0.7']['sha256'] = hashlib.sha256(b'different-test-mask').hexdigest()
    elif changed == 'train_count':
        trace['train']['1']['N'] -= 1
    elif changed == 'selection_count':
        trace['selection']['1']['0.6']['batches'] += 1
    elif changed == 'missing_train':
        del trace['train']['1']
    elif changed == 'missing_rate':
        del trace['selection']['1']['0.4']
    else:
        del reference['selection']['1']
    with pytest.raises(ValueError):
        runner().assert_epoch_masks(trace, reference, 1)


def test_queue_uses_only_two_authorized_gpus_and_independent_worker_outputs(tmp_path):
    run = runner()
    scheduler = run.OptimizerScheduler(tmp_path / 'queue', model_source=tmp_path / 'b4',
                                       control=tmp_path / 'W0')
    assert scheduler.gpus == (1, 2)
    assert scheduler.capacity == {1: 4, 2: 4}
    commands = []
    for job in EXPECTED_SPECS:
        output = tmp_path / job
        config = tmp_path / f'{job}.json'
        command = scheduler.command({'id': job, 'output': str(output)}, config, 'GPU-safe')
        assert '--worker' in command
        assert command[command.index('--job') + 1] == job
        assert command[command.index('--config') + 1] == str(config)
        assert command[command.index('--output') + 1] == str(output)
        assert command[command.index('--model-source') + 1] == str(tmp_path / 'b4')
        assert command[command.index('--control') + 1] == str(tmp_path / 'W0')
        assert command[command.index('--gpu-uuid') + 1] == 'GPU-safe'
        commands.append(tuple(command))
    assert len(set(commands)) == 8
