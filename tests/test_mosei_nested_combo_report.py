import copy
import pytest


def module():
    from experiments.osram_mosei_nested_combinations_20261024 import report
    return report


def test_partial_not_ranked(tmp_path):
    with pytest.raises(ValueError):
        module().write_report(tmp_path, {'jobs': [], 'source_commit': 'sha'})


def test_pair_requires_identical_masks_training_initialization_and_numerical_source():
    base = dict(config={'osram_meaningful_block':'none','learning_rate':1e-4},
        initial_backbone_sha256='same', identity=dict(mask_reference='mask',split='split',
        source_hashes={'gcnet_missing_m3/osram.py':'same',
                      'gcnet_missing_m3/meaningful_new40_registry.py':'old'}))
    candidate = copy.deepcopy(base)
    candidate['config']['osram_meaningful_block'] = 'new'
    candidate['identity']['source_hashes']['gcnet_missing_m3/meaningful_new40_registry.py'] = 'new'
    trace = [dict(epoch=0,mask_trace_sha256='trace',optimizer_steps=2,learning_rates=[1e-4])]
    module().validate_pair(candidate,trace,base,trace)
    for key in ('mask_reference','split'):
        bad = copy.deepcopy(candidate); bad['identity'][key] = 'changed'
        with pytest.raises(ValueError): module().validate_pair(bad,trace,base,trace)
    bad = copy.deepcopy(candidate)
    bad['identity']['source_hashes']['gcnet_missing_m3/osram.py'] = 'different'
    with pytest.raises(ValueError): module().validate_pair(bad,trace,base,trace)
    changed = copy.deepcopy(trace); changed[0]['mask_trace_sha256'] = 'different'
    with pytest.raises(ValueError): module().validate_pair(candidate,changed,base,trace)
