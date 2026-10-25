"""Pinned completed Flat parents and mandatory same-mask frozen replay.

Sentiment reuses the preceding audited bridge. MELD loads the accepted original
lr1e-5/wd1e-5 parent, never the no-skip or later module variants. No old run is
modified, and an exact selected-logit replay is required before stage two.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from osram_four_block_wiring import FrozenFlatReader
from experiments.osram_signature_stage2_extension_20261025 import prepare as previous
from experiments.osram_tasr_20261018 import runner_helpers as h

ROOT = Path(__file__).resolve().parents[2]
RATES = previous.RATES
validate_rate = previous.validate_rate
MELD_FEATURE = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
MELD_ADDED_DEFAULTS = dict(previous.NEW_DEFAULTS,
    checkpoint_selection_metric='dataset-default', preserve_selection_artifacts=False,
    osram_causal_gru_history=False, osram_source_tracking=False, osram_tasr=False,
    osram_tasr_source_mode='real', osram_source_role=False, osram_source_role_mode='residual',
    osram_rpca_rank=1, osram_rpca_steps=14, osram_rpca_threshold_init=.1,
    osram_rpca_lr_multiplier=1., osram_mbt_bottlenecks=4, osram_mbt_rounds=2,
    osram_mbt_dim=128, osram_mbt_heads=4, osram_mbt_ffn_hidden=256,
    osram_mbt_lr_multiplier=1., osram_disable_local_skip=False)


def meld_case():
    from experiments.osram_tasr_20261018.protocol import build_cases
    return build_cases()['MELD']


def configure_environment(dataset, output):
    """Bind MELD paths before any trainer imports the process-global config."""
    if dataset == 'MELD':
        case = meld_case()
        os.environ.update(GCNET_MELD_PICKLE=str(MELD_FEATURE),
            GCNET_DATASET_ROOT=str(Path(case['numerical_source']['source'])/'dataset'),
            GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/'MELD'))


def parent_path(dataset, seed):
    if dataset == 'CMUMOSI' and seed in (66,67):
        return previous.parent_path(dataset,seed)
    if dataset == 'CMUMOSEI' and seed in (66,67,68):
        return previous.MOSEI66 if seed == 66 else previous.UNIFIED/dataset/f'seed_{seed}'/'flat'
    if dataset == 'MELD' and seed == 66:
        return Path(meld_case()['output'])
    raise ValueError('Only MOSI66/67, MOSEI66/67/68 and MELD66 are authorized')


def mask_path(dataset, seed):
    parent_path(dataset,seed)
    return previous.UNIFIED/dataset/'seed_66'/'flat' if dataset == 'CMUMOSEI' else parent_path(dataset,seed)


def meld_config(raw):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg=TrainConfig(**raw)
    required=dict(dataset='MELD',seed=66,fold=1,epochs=50,learning_rate=1e-5,
        weight_decay=1e-5,batch_size=32,optimizer='adam',missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate',training_objective='emotion-only',
        train_rate_mode='cyclic',evaluation_protocol='official',osram_output_dim=1600,
        osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,osram_write_step=.6,
        osram_bidirectional=False,osram_readout_fusion='flat',osram_meaningful_block='none',
        osram_disable_local_skip=False,osram_tasr=False,osram_source_role=False,
        osram_source_relation=False,osram_source_role_v3=False,osram_source_moe_mode='off',
        osram_conditional_reread=False,paired_history_views=False,completion_path='none')
    for key,value in required.items():
        if getattr(cfg,key) != value:
            raise ValueError('Protected MELD original Flat protocol differs: '+key)
    added={k:v for k,v in asdict(cfg).items() if k not in raw}
    if any(k not in MELD_ADDED_DEFAULTS or v != MELD_ADDED_DEFAULTS[k] for k,v in added.items()):
        raise ValueError('Unreviewed MELD added default config')
    return cfg,added


def selected_meld_arrays(ref):
    if h.sha(ref['selection']) != ref['selection_sha256']:
        raise ValueError('Protected MELD selected prediction hash differs')
    with np.load(ref['selection'],allow_pickle=False) as stored:
        arrays={k:stored[k].copy() for k in stored.files}
    if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != ref['mask_sha256']:
        raise ValueError('Protected MELD selected mask hash differs')
    np.testing.assert_array_equal(arrays['predictions'],arrays['logits'].argmax(-1))
    return arrays


def _sentiment_setup(dataset,seed,uuid,output,rate):
    path=parent_path(dataset,seed)
    accepted=json.loads((path/'SUMMARY.json').read_text())
    if accepted['status'] != 'complete' or accepted['epoch'] != (150 if dataset == 'CMUMOSI' else 60):
        raise ValueError('Completed same-seed original Flat required')
    selected=accepted['best_by_rate'][rate]
    checkpoint=path/selected['checkpoint']
    digest=h.sha(checkpoint)
    if digest != selected['checkpoint_sha256']:
        raise ValueError('Selected original Flat checkpoint SHA differs')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['epoch'] != selected['epoch'] or previous._canonical(state['identity']) != accepted['identity']:
        raise ValueError('Selected original Flat epoch/identity differs')
    if 'selection' in state and validate_rate(state['selection']['rate']) != rate:
        raise ValueError('Selected original Flat rate differs')
    bundle=previous._build_bundle(dataset,seed,uuid,output,accepted)
    bundle['model'].load_state_dict(state['model'],strict=True)
    reader=FrozenFlatReader(bundle['model'])
    core={p:sha for p,sha in accepted['identity']['source_hashes'].items()
          if p.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/'))}
    changes=previous._core_differences(core,bundle['cfg'])
    proof=dict(dataset=dataset,seed=seed,rate=rate,parent_path=str(path),parent_id=digest,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,epoch=state['epoch'],
        summary_sha256=h.sha(path/'SUMMARY.json'),config=asdict(bundle['cfg']),
        environment=bundle['environment'],parent_frozen=True,masks_parent_path=str(mask_path(dataset,seed)),
        train_rates=list(map(float,RATES)),test_rates=[rate],saved_prediction_replay_required=True,
        parent_rng_repair=accepted['identity'].get('rng_repair'),
        added_default_config_fields=bundle['added_default_config_fields'],core_source_differences=changes,
        source_bridge='Reuse reviewed disabled-extension bridge plus mandatory saved prediction replay',
        parent_comparison='Own original per-rate selected parent; MOSEI66 repaired parent differs from older unified66.')
    bundle['parent_identity']=proof
    return bundle,reader,proof


def _meld_setup(uuid,output,rate):
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    case=meld_case()
    cfg,added=meld_config(case['expected_config'])
    environment=require_replication_environment(torch)
    gpu=guard_gpu(uuid)
    if h.sha(MELD_FEATURE) != case['dataset_sha256']:
        raise ValueError('Protected MELD feature SHA differs')
    path=parent_path('MELD',66)
    registry=json.loads((ROOT/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json').read_text())
    registered=next(r for r in registry['runs'] if r['seed'] == 66)
    if h.sha(path/'SUMMARY.json') != registered['summary_sha256']:
        raise ValueError('Protected MELD completed summary hash differs')
    accepted=json.loads((path/'SUMMARY.json').read_text())
    if accepted['status'] != 'complete' or accepted['config'] != case['expected_config']:
        raise ValueError('Completed protected MELD original Flat config differs')
    for key in ('torch','numpy','sklearn'):
        if environment[key] != registered['environment'][key]:
            raise ValueError('Protected MELD environment differs: '+key)
    ref=case['evaluation_references'][rate]
    if h.sha(ref['checkpoint']) != ref['checkpoint_sha256']:
        raise ValueError('Protected MELD checkpoint hash differs')
    state=torch.load(ref['checkpoint'],map_location='cpu',weights_only=False)
    if state['config'] != case['expected_config'] or state['epoch'] != ref['epoch']:
        raise ValueError('Protected MELD checkpoint config/epoch differs')
    artifact=state['selection_artifact']
    if (artifact['sha256'] != ref['selection_sha256'] or
        artifact['availability_sha256'] != ref['mask_sha256'] or
        str((path/artifact['path']).resolve()) != str(Path(ref['selection']).resolve())):
        raise ValueError('Protected MELD exact selected artifact differs')
    selected=selected_meld_arrays(ref)
    os.environ.update(GCNET_MELD_PICKLE=str(MELD_FEATURE),
        GCNET_DATASET_ROOT=str(Path(case['numerical_source']['source'])/'dataset'),
        GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/'MELD'))
    from gcnet_missing_m3 import train_gcnet as trainer
    train,val,test,*dimensions=trainer.get_loaders(*('MELD-pickle',)*3,dataset='MELD',num_folder=1,
        batch_size=32,num_workers=0,seed=66,validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol)
    model=_build_model(cfg,tuple(dimensions),torch.device('cuda:0'))
    model.load_state_dict(state['model'],strict=True)
    if sum(p.numel() for p in model.parameters()) != accepted['parameters']:
        raise ValueError('Protected MELD architecture parameter count differs')
    proof=dict(dataset='MELD',seed=66,rate=rate,parent_path=str(path),parent_id=ref['checkpoint_sha256'],
        checkpoint=ref['checkpoint'],checkpoint_sha256=ref['checkpoint_sha256'],epoch=ref['epoch'],
        summary_sha256=registered['summary_sha256'],config=asdict(cfg),environment=environment,
        parent_frozen=True,train_rates=list(map(float,RATES)),test_rates=[rate],
        baseline_registry_sha256=case['registry_sha256'],dataset_sha256=case['dataset_sha256'],
        original_selection=case['source_selection'],selected_prediction=ref['selection'],
        selected_prediction_sha256=ref['selection_sha256'],test_mask_sha256=ref['mask_sha256'],
        added_default_config_fields=added,saved_prediction_replay_required=True,
        numerical_source=case['numerical_source'],
        source_bridge='Current strict-load + pinned old checkpoint/config/data/environment + mandatory bit-exact selected logits')
    bundle=dict(model=model,cfg=cfg,trainer=trainer,dimensions=tuple(dimensions),
        train_loader=train[0],validation_loader=val[0],test_loader=test[0],environment=environment,
        gpu=gpu,parent_identity=proof,selected_meld_arrays=selected,
        data_files={str(MELD_FEATURE):case['dataset_sha256']},
        split={n:loader.protocol_metadata for n,loader in [('train',train[0]),('validation',val[0]),('test',test[0])]})
    bundle['seed_protocol']=seed_training_loader(bundle,66)
    return bundle,FrozenFlatReader(model),proof


def setup(dataset,seed,uuid,output,rate):
    parent_path(dataset,seed)
    rate=validate_rate(rate)
    return _meld_setup(uuid,output,rate) if dataset == 'MELD' else _sentiment_setup(dataset,seed,uuid,output,rate)


def test_cache(bundle,reader,rate):
    rate=validate_rate(rate)
    if bundle['cfg'].dataset != 'MELD':
        return previous.test_cache(bundle,reader,rate)
    proof=bundle['parent_identity']
    if proof['rate'] != rate:
        raise ValueError('Test rate must match frozen MELD parent')
    selected=bundle['selected_meld_arrays']
    trainer,loader=bundle['trainer'],bundle['test_loader']
    device=next(reader.parent.parameters()).device
    rows,cursor=[],0
    with h.isolated_rng(66+99193),torch.no_grad():
        loader.sampler.set_epoch(0)
        for raw in loader:
            data=trainer._move_batch(raw,device)
            meta=h.metadata(data)
            n=len(meta['labels'])
            for key,value in meta.items():
                np.testing.assert_array_equal(value,selected[key][cursor:cursor+n])
            a=torch.zeros((*data[7].shape,3),device=device,dtype=torch.float32)
            a[data[7].bool()]=torch.as_tensor(selected['availability'][cursor:cursor+n],device=device,dtype=a.dtype)
            a=a.transpose(0,1)
            view=trainer._prepare_view_from_primary_masks(data,a,a,bundle['dimensions'])
            frame=reader(view['incomplete'],a,view['qmask'],view['umask'],view['lengths'])
            torch.testing.assert_close(reader.decode_delta(frame,torch.zeros_like(frame.anchor)),frame.prediction,rtol=0,atol=0)
            rows.append(dict(frame=frame.to('cpu'),labels=view['labels'].cpu(),umask=view['umask'].cpu(),
                conversation_ids=list(map(str,view['conversation_ids'])),metadata=meta,
                availability=a.transpose(0,1)[data[7].bool()].cpu().numpy()))
            cursor+=n
    if cursor != len(selected['labels']) or not rows:
        raise ValueError('Protected MELD selected test sample count differs')
    predictions=np.concatenate([row['frame'].prediction.transpose(0,1)[row['umask'].bool()].numpy() for row in rows])
    np.testing.assert_array_equal(predictions,selected['logits'])
    np.testing.assert_array_equal(predictions.argmax(-1),selected['predictions'])
    proof['saved_prediction_replay']=dict(rate=rate,max_absolute_error=float(np.max(np.abs(predictions-selected['logits']))),
        logits_exact=True,classes_exact=True,prediction_sha256=proof['selected_prediction_sha256'],
        mask_bank_sha256=proof['test_mask_sha256'])
    return rows


test_cache.__test__=False
