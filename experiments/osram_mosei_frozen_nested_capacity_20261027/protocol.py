"""Three approved single-factor Nested structures; frozen MOSEI66 only."""
import json

from experiments.osram_mosei_frozen_nested_lr_20261027.protocol import (
    EPOCHS,RATES,NUMERIC,NUMERICAL_SHA,REFERENCE,sha,validate_epoch_trace)

LR=3e-4
METHOD='nested_gnn_rooted_evidence'
CONFIGS={
    'dim32':dict(method='mosei_nested_d32_l3_g8_m64',dim=32,depth=3,mlp_hidden=64,groups=8,parameters=80163),
    'mlp128':dict(method='mosei_nested_mlp128',dim=64,depth=3,mlp_hidden=128,groups=8,parameters=216899),
    'depth2':dict(method='mosei_nested_depth2',dim=64,depth=2,mlp_hidden=64,groups=8,parameters=146690),
}


def validate_scope(config,rate):
    if config not in CONFIGS or isinstance(rate,bool):raise ValueError('Only three approved capacity settings')
    try:rate=str(float(rate))
    except (TypeError,ValueError):raise ValueError('Only original eight missing rates') from None
    if rate not in RATES:raise ValueError('Only original eight missing rates')
    return rate


def jobs_for(phase):
    if phase not in ('smoke','train'):raise ValueError('Only smoke/train phases')
    return [dict(id=f'{config}_rate{rate}',config=config,rate=rate,status='pending')
        for rate in (('0.7',) if phase=='smoke' else RATES) for config in CONFIGS]


def reference_dir(rate):
    rate=validate_scope('dim32',rate)
    return REFERENCE/f'CMUMOSEI_66_fold1_{rate}'


def validate_reference(reference,parent,effective_config,mask_sha,rate):
    rate=validate_scope('dim32',rate)
    expected=dict(stage='frozen',mode='nested',method=METHOD,dataset='CMUMOSEI',seed=66,fold=1,
        evaluation_rate=rate,smoke=False,stage2_epochs=EPOCHS,cyclic_training=True,training_rates=list(RATES),
        mask_protocol='gcnet-random-mask-v1',parent=parent,test_mask_sha256=mask_sha,
        effective_stage2_config=effective_config)
    expected=json.loads(json.dumps(expected));identity=reference.get('identity',{})
    if (reference.get('status')!='complete' or reference.get('epoch')!=EPOCHS
        or reference.get('total_parameters')!=159235 or reference.get('parent_id')!=parent['parent_id']
        or effective_config.get('learning_rate')!=LR or any(identity.get(k)!=v for k,v in expected.items())):
        raise ValueError('Require same original parent, effective training config, seed and mask')
