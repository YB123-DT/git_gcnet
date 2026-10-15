"""Extending exact selection bookkeeping must not change model/mask RNG."""
import numpy as np
import pytest
import torch

from gcnet_missing_m3 import train_gcnet as trainer
from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
from tests.test_meld_query_decision_trace import fixture


@pytest.mark.parametrize('dataset,classes', [('IEMOCAPFour', 4), ('IEMOCAPSix', 6)])
def test_exact_iemocap_artifacts_do_not_change_forward_or_rng(dataset, classes):
    model, raw, *_ = fixture()
    model.smax_fc = torch.nn.Linear(4, classes)
    raw[8] = raw[8].clamp_min(0) % classes
    schedule = GCNetLegacyRandomMaskSchedule(dataset, 'test', 5, .7, 66)
    results = []
    with torch.no_grad():
        for collecting in (False, True):
            np.random.seed(91)
            torch.manual_seed(75)
            metrics, artifact = trainer.evaluate_rate(model, [raw], schedule, dataset,
                (4, 4, 4), torch.device('cpu'), collect=collecting, collect_identity=collecting)
            results.append((metrics, artifact, np.random.get_state(), torch.get_rng_state()))
    assert results[0][0]['accuracy'] == results[1][0]['accuracy']
    assert results[0][0]['weighted_f1'] == results[1][0]['weighted_f1']
    assert results[0][0]['loss'] == results[1][0]['loss']
    assert np.array_equal(results[0][2][1], results[1][2][1])
    assert results[0][2][2:] == results[1][2][2:]
    assert torch.equal(results[0][3], results[1][3])
    artifact = results[1][1]
    assert artifact['logits'].shape == (6, classes)
    assert artifact['conversation_ids'].tolist() == ['a']*4 + ['b']*2
    np.testing.assert_array_equal(artifact['predictions'], artifact['logits'].argmax(1))
    np.testing.assert_array_equal(artifact['labels'], raw[8][raw[7].bool()].numpy())
