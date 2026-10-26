import importlib
from pathlib import Path

import pytest

PACKAGE = 'experiments.osram_frozen_nested_3seed_20261026'


def protocol():
    return importlib.import_module(PACKAGE + '.protocol')


def test_exact_128_new_branches_and_mosi_lr_expression():
    p = protocol()
    cells = [(d, s, f, r) for d in p.DATASETS for s in p.SEEDS
             for f in (range(1, 6) if d == 'IEMOCAPSix' else (1,)) for r in p.RATES]
    assert len(cells) == 128
    for d, s, f, r in cells:
        assert p.validate_scope(d, s, f, r) == r
    for args in [('CMUMOSI', 66, 1, '.3'), ('MELD', 67, 2, '.3'),
                 ('IEMOCAPSix', 68, 6, '.3'), ('MELD', 67, 1, True)]:
        with pytest.raises(ValueError): p.validate_scope(*args)
    assert p.learning_rate('CMUMOSI', 100) == 3e-4 / 10
    assert p.learning_rate('CMUMOSI', 100) != 3e-5
    assert p.learning_rate('MELD', 49) == 3e-4


def test_sources_and_reference_absence_are_truthful():
    p = protocol()
    assert p.source_root('CMUMOSI', 68).name == 'source_4d5aa09'
    assert p.source_root('CMUMOSI', 67).name == 'source_e0ca94c'
    assert p.source_root('IEMOCAPSix', 67).name == 'source_21e46f2'
    assert p.reference_dir('CMUMOSI', 68, 1, '.7').name == '0.7'
    assert p.reference_dir('IEMOCAPSix', 68, 5, '.7').name == 'IEMOCAPSix_68_fold5_0.7'
    assert p.reference_dir('MELD', 67, 1, '.7') is None


def test_trace_validates_without_fabricating_signature_reference():
    p = protocol()
    row = dict(epoch=1, optimizer_steps=2, training_rates=[0., .1], mask_trace_sha256='a'*64,
               N=5, learning_rate=3e-4)
    p.validate_epoch_trace('MELD', row)
    p.validate_epoch_trace('MELD', row, row)
    for changes in [dict(mask_trace_sha256='bad'), dict(N=0), dict(optimizer_steps=1),
                    dict(training_rates=[.8, .1]), dict(learning_rate=1e-5)]:
        with pytest.raises(ValueError): p.validate_epoch_trace('MELD', dict(row, **changes))


def test_reference_checks_scientific_parent_fields_not_whole_new_proof():
    p = protocol()
    cfg = dict(dataset='CMUMOSI', seed=68, fold=1)
    old_parent = dict(parent_id='ckpt', checkpoint_sha256='ckpt', config=cfg,
                      masks_parent_path='/same/mask', saved_prediction_replay={'max_absolute_error': 0.})
    parent = dict(old_parent, additional_provenance='new')
    identity = dict(mode='signature', dataset='CMUMOSI', seed=68, evaluation_rate='0.7',
                    source_commit=p.source_commit('CMUMOSI', 68), smoke=False, stage2_epochs=150,
                    cyclic_training=True, training_rates=list(p.RATES),
                    mask_protocol='gcnet-random-mask-v1', effective_stage2_config=cfg,
                    test_mask_sha256='mask', parent=old_parent)
    reference = dict(status='complete', epoch=150, total_parameters=281856,
                     parent_id='ckpt', identity=identity)
    p.validate_reference(reference, parent, cfg, 'mask', 'CMUMOSI', 68, 1, '0.7')
    with pytest.raises(ValueError):
        p.validate_reference(reference, dict(parent, checkpoint_sha256='wrong'), cfg,
                             'mask', 'CMUMOSI', 68, 1, '0.7')


def test_meld_registry_uses_own_seed_parent_masks_and_original_config():
    prepare = importlib.import_module(PACKAGE + '.prepare')
    registry_path = Path(__file__).resolve().parents[1]/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json'
    case67 = prepare.meld_case(67, registry_path)
    case68 = prepare.meld_case(68, registry_path)
    assert case67['expected_config']['seed'] == 67
    assert case68['expected_config']['seed'] == 68
    assert case67['output'].endswith('/S67_WD1')
    assert case68['output'].endswith('/S68_WD1')
    assert case67['evaluation_references']['0.7']['mask_sha256'] != case68['evaluation_references']['0.7']['mask_sha256']
    assert case67['evaluation_references']['0.7']['checkpoint'] != case68['evaluation_references']['0.7']['checkpoint']


def test_flat_saved_mask_uses_recorded_conversation_and_utterance_order():
    import numpy as np
    import torch
    p = importlib.import_module(PACKAGE + '.prepare')
    availability = torch.arange(18).reshape(3, 2, 3)
    record = dict(availability=availability, conversation_ids=np.array(['b', 'b', 'a']),
                  utterance_indices=np.array([0, 1, 0]))
    np.testing.assert_array_equal(p.flat_record_availability(record),
                                  np.array([[0, 1, 2], [6, 7, 8], [3, 4, 5]]))
    with pytest.raises(ValueError):
        p.flat_record_availability(dict(record, utterance_indices=np.array([0, 4, 0])))
