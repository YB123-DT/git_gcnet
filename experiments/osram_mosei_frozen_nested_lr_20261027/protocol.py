"""Finite learning-rate-only follow-up to the completed frozen Nested66."""
import hashlib
import json
from pathlib import Path

LRS={'lr1e4':1e-4,'lr1e3':1e-3}
RATES=tuple(str(i/10) for i in range(8))
EPOCHS=60
PARAMETERS=159235
METHOD='nested_gnn_rooted_evidence'
NUMERICAL_SHA='e0ca94c4c33de0efda2c2f4786fd289067e1ab2c'
NUMERIC=Path('/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c')
REFERENCE=Path('/data2/yb/remote_experiments/osram_frozen_nested_cross_20261025/execution_v1/train')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def validate_scope(config,rate):
    if config not in LRS or isinstance(rate,bool):
        raise ValueError('Only two approved MOSEI66 learning rates')
    try:rate=str(float(rate))
    except (ValueError,TypeError):raise ValueError('Require one of eight original rates') from None
    if rate not in RATES:raise ValueError('Require one of eight original rates')
    return rate


def jobs_for(phase):
    if phase not in ('smoke','train'):raise ValueError('Only smoke/train')
    rates=('0.7',) if phase=='smoke' else RATES
    return [dict(id=f'{config}_rate{rate}',config=config,rate=rate,status='pending')
        for rate in rates for config in LRS]


def reference_dir(rate):
    rate=validate_scope('lr1e4',rate)
    return REFERENCE/f'CMUMOSEI_66_fold1_{rate}'


def validate_reference(reference,parent,effective_config,mask_sha,rate):
    rate=validate_scope('lr1e4',rate)
    identity=reference.get('identity',{})
    expected=dict(stage='frozen',mode='nested',method=METHOD,dataset='CMUMOSEI',seed=66,
        fold=1,evaluation_rate=rate,smoke=False,stage2_epochs=EPOCHS,cyclic_training=True,
        training_rates=list(RATES),mask_protocol='gcnet-random-mask-v1',parent=parent,
        test_mask_sha256=mask_sha)
    expected=json.loads(json.dumps(expected))
    if (reference.get('status')!='complete' or reference.get('epoch')!=EPOCHS
        or reference.get('total_parameters')!=PARAMETERS or reference.get('parent_id')!=parent['parent_id']
        or any(identity.get(k)!=v for k,v in expected.items())):
        raise ValueError('Completed default Nested parent/config/seed/mask reference differs')
    old=dict(identity['effective_stage2_config'])
    new=json.loads(json.dumps(effective_config))
    if old.pop('learning_rate')!=3e-4 or new.pop('learning_rate') not in LRS.values() or old!=new:
        raise ValueError('Only second-stage learning_rate may differ from default Nested')


def validate_epoch_trace(row,reference,lr):
    for key in ('epoch','optimizer_steps','training_rates','mask_trace_sha256','N'):
        if row.get(key)!=reference.get(key):raise ValueError('Original sampling/update trace differs: '+key)
    if row.get('learning_rate')!=lr:raise ValueError('Actual optimizer learning rate differs')
