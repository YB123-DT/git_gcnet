"""Pinned original IEMOCAP-6 LOSO parents and bit-exact selected-mask replay.

No new validation partition or parent training is introduced. The historical
official protocol aliases validation to the held-out session, which is retained
and explicitly recorded as internal test-oracle exploration.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from osram_four_block_wiring import FrozenFlatReader
from experiments.osram_signature_cyclic_multi_20261025 import prepare as previous
from experiments.osram_tasr_20261018 import runner_helpers as h

ROOT = Path(__file__).resolve().parents[2]
RATES = previous.RATES
validate_rate = previous.validate_rate
DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
AGGREGATE = ROOT/'experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json'
TRAINING_SHA = '785a01242ef1a82b97d5337acbb33ba963df5537'
LABEL_SHA = 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'
FEATURE_NAMES = ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')
# Reviewed old->current differences only add configurable, disabled MBT paths.
# These adapters are not instantiated for original meaningful_block='none'.
DISABLED_MBT_HASHES = {
    'gcnet_missing_m3/meaningful_input.py': dict(
        parent='0c3a4b8f78d50bc11983997b5cfe125dfc8ef5f2539f9c5356e8cfa9511fa770',
        current='e4a86fe4aabd42ca9d0b2ccd503e9d588afa9f6f803d8e74748fe6970c1037f8'),
    'gcnet_missing_m3/meaningful_input_representation.py': dict(
        parent='e4ee8076009ded8f46d827b2c51aa960284acaa2c12a9cacdc28cd502ef5f756',
        current='ecbe48a04c49577bf5d305271ffadef9a70877d7e7774ca227c529d4629b0ff3'),
}


def configure_environment(output):
    """Must precede trainer/config imports; MELD overrides must not leak in."""
    os.environ['GCNET_DATASET_ROOT'] = str(DATA_ROOT)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent/'feature_cache'/'IEMOCAPSix')
    os.environ.pop('GCNET_MELD_PICKLE', None)


def registered_parent(seed, fold):
    """Resolve a target from the accepted registry without loading any weights."""
    if type(seed) is not int or seed not in (66, 67, 68) or type(fold) is not int or fold not in range(1, 6):
        raise ValueError('Only IEMOCAP-6 seeds66/67/68 and LOSO folds1..5 are authorized')
    aggregate = json.loads(AGGREGATE.read_text())
    if aggregate.get('status') != 'complete' or aggregate.get('completed_fold_runs') != 30:
        raise ValueError('Require accepted complete original Flat three-seed registry')
    job_id = f'iemocap6_Original_Flat_seed{seed}_fold{fold}'
    jobs = [j for j in aggregate['jobs'] if j['id'] == job_id]
    if len(jobs) != 1 or jobs[0]['status'] != 'complete' or not jobs[0]['summary_valid'] or jobs[0]['failures']:
        raise ValueError('Protected original Flat LOSO result is not complete')
    output = Path(jobs[0]['output'])
    rates = aggregate['datasets']['IEMOCAPSix']['folds'][str(seed)][str(fold)]['rates']
    refs = {r: dict(checkpoint=str(output/f'best_miss_{r.replace(".","p")}.pt'),
                   checkpoint_sha256=v['checkpoint_sha256'], selection_sha256=v['selection_sha256'],
                   mask_sha256=v['mask_sha256'], epoch=v['epoch'], expected_metrics=v)
            for r,v in rates.items()}
    if set(refs) != set(RATES):
        raise ValueError('Protected parent needs all eight selected rates')
    return dict(dataset='IEMOCAPSix', classes=6, seed=seed, fold=fold, output=str(output),
                references=refs, aggregate_sha256=h.sha(AGGREGATE))


def parent_config(raw, seed, fold):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = TrainConfig(**raw)
    expected = dict(dataset='IEMOCAPSix', seed=seed, fold=fold, epochs=100, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, optimizer='adam', dropout=.5, latent_dim=256,
        lr_schedule='constant', missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy',
        training_objective='emotion-only', train_rate_mode='cyclic', evaluation_protocol='official',
        evaluate_test=True, osram_readout_fusion='flat', osram_meaningful_block='none',
        osram_output_dim=1600, osram_adapter_hidden_dim=0, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
        osram_disable_local_skip=False, paired_history_views=False, completion_path='none',
        osram_tasr=False, osram_source_tracking=False, osram_source_role=False,
        osram_source_relation=False, osram_source_role_v3=False, osram_source_moe_mode='off',
        osram_conditional_reread=False, initial_backbone_checkpoint=None)
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError('Protected IEMOCAP original Flat protocol differs: '+key)
    if tuple(cfg.train_missing_rates) != tuple(map(float,RATES)):
        raise ValueError('Protected original cyclic rates differ')
    added = {k:v for k,v in asdict(cfg).items() if k not in raw}
    if any(k not in previous.MELD_ADDED_DEFAULTS or v != previous.MELD_ADDED_DEFAULTS[k] for k,v in added.items()):
        raise ValueError('Unreviewed added IEMOCAP config defaults')
    return cfg, added


def core_differences(hashes, cfg):
    remaining = dict(hashes)
    reviewed = {}
    for path, pair in DISABLED_MBT_HASHES.items():
        old, new = remaining.get(path), h.sha(ROOT/path)
        if old != new:
            if cfg.osram_meaningful_block != 'none' or dict(parent=old,current=new) != pair:
                raise ValueError('Unreviewed disabled MBT source drift: '+path)
            reviewed[path] = dict(pair)
            remaining.pop(path)
    return dict(previous.previous._core_differences(remaining,cfg), **reviewed)


def selected_arrays(ref):
    if h.sha(ref['selection']) != ref['selection_sha256']:
        raise ValueError('Protected selected prediction hash differs')
    with np.load(ref['selection'],allow_pickle=False) as stored:
        selected = {k:stored[k].copy() for k in stored.files}
    if hashlib.sha256(selected['availability'].tobytes()).hexdigest() != ref['mask_sha256']:
        raise ValueError('Protected selected mask hash differs')
    if selected['logits'].shape != (len(selected['labels']),6):
        raise ValueError('Protected selected predictions must contain six class logits')
    np.testing.assert_array_equal(selected['predictions'],selected['logits'].argmax(-1))
    return selected


def parent_environment(accepted, actual, seed, output):
    """Seed66 predates full package provenance; never invent missing versions."""
    recorded = accepted['provenance']['environment']
    proof = dict(recorded=recorded,unrecorded_historical_versions=[])
    for key in ('torch','numpy','sklearn'):
        if key not in recorded:
            if seed != 66 or key != 'sklearn':
                raise ValueError('Protected parent environment version missing: '+key)
            proof['unrecorded_historical_versions'].append(key)
        elif actual[key] != recorded[key]:
            raise ValueError('Protected cmerc environment differs: '+key)
    executable = recorded.get('executable')
    if executable is None and seed == 66:
        manifest = Path(output).parents[3]/'MANIFEST.json'
        payload = json.loads(manifest.read_text())
        executable = payload['provenance']['environment']['executable']
        proof.update(launcher_manifest=str(manifest),launcher_manifest_sha256=h.sha(manifest))
    if executable != '/data2/yb/conda_envs/cmerc-repro/bin/python':
        raise ValueError('Protected parent cmerc executable differs or is unproved')
    proof['executable'] = executable
    return proof


def setup(seed, fold, uuid, output, rate):
    configure_environment(output)
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from gcnet_missing_m3 import train_gcnet as trainer
    case, rate = registered_parent(seed,fold), validate_rate(rate)
    path = Path(case['output'])
    accepted = json.loads((path/'SUMMARY.json').read_text())
    raw = json.loads((path/'config.json').read_text())
    if (accepted['status'] != 'complete' or accepted['method'] != 'Original_Flat'
            or accepted.get('smoke_only') is not False or accepted['effective_config'] != raw
            or any(raw.get(k) != v for k,v in accepted['config'].items())
            or accepted['identity']['source_git_sha'] != TRAINING_SHA
            or accepted['identity']['label_sha256'] != LABEL_SHA):
        raise ValueError('Protected completed original Flat identity/config differs')
    cfg, added = parent_config(raw,seed,fold)
    environment = require_replication_environment(torch)
    gpu = guard_gpu(uuid)
    environment_proof = parent_environment(accepted,environment,seed,path)
    label = DATA_ROOT/'IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    if h.sha(label) != LABEL_SHA:
        raise ValueError('Protected six-class label data changed')
    roots = [str(DATA_ROOT/'IEMOCAP/features'/name) for name in FEATURE_NAMES]
    if roots != accepted['provenance']['feature_roots'] or not all(Path(r).is_dir() for r in roots):
        raise ValueError('Protected original IEMOCAP feature roots differ')
    ref = dict(case['references'][rate])
    if accepted['testbest']['rates'][rate] != ref['expected_metrics']:
        raise ValueError('Parent metrics differ from accepted registry')
    if h.sha(ref['checkpoint']) != ref['checkpoint_sha256']:
        raise ValueError('Protected checkpoint hash differs')
    state = torch.load(ref['checkpoint'],map_location='cpu',weights_only=False)
    if (state['config'] != raw or state['epoch'] != ref['epoch'] or state['selection_rate'] != float(rate)
            or state['selection_metric'] != 'accuracy' or state['selection_split'] != 'test'):
        raise ValueError('Protected checkpoint epoch/config/ACC selection differs')
    ref['selection'] = str(h.resolve_selection(state,path,ref['selection_sha256']))
    if state['selection_artifact']['availability_sha256'] != ref['mask_sha256']:
        raise ValueError('Protected checkpoint selected mask differs')
    selected = selected_arrays(ref)
    changes = core_differences(accepted['identity']['code_hashes'],cfg)
    train, validation, test, *dimensions = trainer.get_loaders(*roots,dataset='IEMOCAPSix',
        num_folder=5,batch_size=32,num_workers=0,seed=seed,
        validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    if not all(len(loaders) == 5 for loaders in (train,validation,test)):
        raise ValueError('Require all five official LOSO loaders')
    training, valid, testing = train[fold-1],validation[fold-1],test[fold-1]
    if set(training.sampler.indices) & set(testing.sampler.indices):
        raise ValueError('LOSO training and held-out conversations overlap')
    if any(loader.protocol_metadata['fold'] != fold for loader in (training,valid,testing)):
        raise ValueError('LOSO loader fold identity differs')
    model = _build_model(cfg,tuple(dimensions),torch.device('cuda:0'))
    model.load_state_dict(state['model'],strict=True)
    proof = dict(dataset='IEMOCAPSix',seed=seed,fold=fold,rate=rate,parent_path=str(path),
        parent_id=ref['checkpoint_sha256'],checkpoint=ref['checkpoint'],
        checkpoint_sha256=ref['checkpoint_sha256'],epoch=ref['epoch'],
        summary_sha256=h.sha(path/'SUMMARY.json'),config_sha256=h.sha(path/'config.json'),
        aggregate_sha256=case['aggregate_sha256'],config=asdict(cfg),environment=environment,
        parent_environment_proof=environment_proof,
        dataset_sha256=LABEL_SHA,feature_roots=roots,parent_frozen=True,
        train_rates=list(map(float,RATES)),test_rates=[rate],saved_prediction_replay_required=True,
        selected_prediction=ref['selection'],selected_prediction_sha256=ref['selection_sha256'],
        test_mask_sha256=ref['mask_sha256'],source_git_sha=TRAINING_SHA,
        added_default_config_fields=added,core_source_differences=changes,
        source_bridge='Strict same-architecture load; reviewed inactive extensions; mandatory exact selected logits',
        original_selection='INTERNAL per-rate test-oracle ACC; same-epoch UA',
        official_validation_aliases_test=True)
    bundle = dict(model=model,cfg=cfg,trainer=trainer,dimensions=tuple(dimensions),
        train_loader=training,validation_loader=valid,test_loader=testing,
        environment=environment,gpu=gpu,parent_identity=proof,selected_arrays=selected,
        data_files={str(label):LABEL_SHA},split={name:loader.protocol_metadata for name,loader in
        [('train',training),('validation',valid),('test',testing)]})
    bundle['seed_protocol'] = dict(training_seed=seed,split_seed=seed,evaluation_seed=seed,
        train_loader=training.protocol_metadata,source='Original accepted LOSO seed and ordering')
    return bundle,FrozenFlatReader(model),proof


def test_cache(bundle, reader, rate):
    """Exactly one scan per saved-mask test batch; no newly sampled test mask."""
    rate = validate_rate(rate)
    proof = bundle['parent_identity']
    if proof['rate'] != rate:
        raise ValueError('Test rate must match frozen original parent')
    selected = bundle['selected_arrays']
    trainer, loader = bundle['trainer'], bundle['test_loader']
    device = next(reader.parent.parameters()).device
    rows, cursor = [], 0
    with h.isolated_rng(proof['seed']+99193),torch.no_grad():
        loader.sampler.set_epoch(0)
        for raw in loader:
            data = trainer._move_batch(raw,device)
            meta = h.metadata(data)
            n = len(meta['labels'])
            for key,value in meta.items():
                np.testing.assert_array_equal(value,selected[key][cursor:cursor+n])
            a = torch.zeros((*data[7].shape,3),device=device,dtype=torch.float32)
            a[data[7].bool()] = torch.as_tensor(selected['availability'][cursor:cursor+n],device=device,dtype=a.dtype)
            a = a.transpose(0,1)
            view = trainer._prepare_view_from_primary_masks(data,a,a,bundle['dimensions'])
            frame = reader(view['incomplete'],a,view['qmask'],view['umask'],view['lengths'])
            torch.testing.assert_close(reader.decode_delta(frame,torch.zeros_like(frame.anchor)),frame.prediction,rtol=0,atol=0)
            rows.append(dict(frame=frame.to('cpu'),labels=view['labels'].cpu(),umask=view['umask'].cpu(),
                conversation_ids=list(map(str,view['conversation_ids'])),metadata=meta,
                availability=a.transpose(0,1)[data[7].bool()].cpu().numpy()))
            cursor += n
    if cursor != len(selected['labels']) or not rows:
        raise ValueError('Protected test sample count differs')
    predictions = np.concatenate([r['frame'].prediction.transpose(0,1)[r['umask'].bool()].numpy() for r in rows])
    np.testing.assert_array_equal(predictions,selected['logits'])
    np.testing.assert_array_equal(predictions.argmax(-1),selected['predictions'])
    proof['saved_prediction_replay'] = dict(rate=rate,max_absolute_error=float(np.max(np.abs(predictions-selected['logits']))),
        logits_exact=True,classes_exact=True,prediction_sha256=proof['selected_prediction_sha256'],
        mask_bank_sha256=proof['test_mask_sha256'])
    return rows


test_cache.__test__ = False
