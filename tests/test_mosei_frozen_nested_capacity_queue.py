import copy
import pytest

from experiments.osram_mosei_frozen_nested_capacity_20261027 import scheduler as s
from experiments.osram_mosei_frozen_nested_capacity_20261027.protocol import CONFIGS,jobs_for


def test_finite_queue_recovery_rejects_missing_runs_source_or_new_config():
    d=dict(source_commit='source',smoke=jobs_for('smoke'),train=jobs_for('train'))
    s.validate_state(d,'source')
    assert set(s.GPUS)=={0,1,2}
    with pytest.raises(ValueError):s.validate_state(d,'other')
    for phase in ('smoke','train'):
        bad=copy.deepcopy(d);bad[phase].pop()
        with pytest.raises(ValueError):s.validate_state(bad,'source')
    bad=copy.deepcopy(d);bad['train'][0]['config']='dim128'
    with pytest.raises(ValueError):s.validate_state(bad,'source')


def test_aggregate_requires_complete_rate_set_for_each_structure():
    from experiments.osram_mosei_frozen_nested_capacity_20261027.summarize import aggregates
    rows=[dict(config='dim32',rate=str(i/10),ACC=80+i,WF1=79+i,MSE=1.) for i in range(8)]
    assert aggregates(rows)['dim32']['mean']['ACC']==83.5
    assert 'mlp128' not in aggregates(rows)
    assert aggregates(rows[:-1])=={} and aggregates(rows+[rows[0]])=={}


def test_selected_metadata_uses_actual_variant_not_original_parameter_count():
    from experiments.osram_mosei_frozen_nested_capacity_20261027.summarize import validate_selected_metadata
    c=CONFIGS['mlp128']
    identity=dict(source_commit='source',config='mlp128',method=c['method'],dataset='CMUMOSEI',seed=66,
        fold=1,stage='frozen',smoke=False,evaluation_rate='0.3',learning_rate=3e-4,
        architecture={k:c[k] for k in ('dim','depth','mlp_hidden','groups')})
    summary=dict(status='complete',epoch=60,total_parameters=c['parameters'],checks={'frozen':True},identity=identity)
    job=dict(config='mlp128',rate='0.3')
    validate_selected_metadata(summary,job,'source')
    bad=copy.deepcopy(summary);bad['total_parameters']=159235
    with pytest.raises(ValueError):validate_selected_metadata(bad,job,'source')
    bad=copy.deepcopy(summary);bad['identity']['architecture']['depth']=2
    with pytest.raises(ValueError):validate_selected_metadata(bad,job,'source')
