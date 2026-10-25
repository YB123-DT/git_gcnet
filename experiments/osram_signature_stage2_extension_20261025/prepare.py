"""Per-rate frozen Signature parents with explicit current-source replay gates."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from osram_four_block_wiring import FrozenFlatReader
from experiments.osram_neumiss_fixed_rate_stage2_20261025.prepare import RATES, validate_rate, train_schedule
from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
from experiments.osram_tasr_20261018 import runner_helpers as h

ROOT = Path(__file__).resolve().parents[2]
UNIFIED = Path('/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs')
MOSI66 = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
MOSEI66 = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none')
NEW_DEFAULTS = dict(osram_source_relation=False, osram_source_relation_layout='source-mean',
    osram_source_role_v3=False, osram_source_role_v3_dim=64, osram_source_role_v3_pair_hidden=128,
    osram_source_role_v3_mode='residual', osram_source_role_v3_variant='standard',
    osram_source_moe_mode='off', osram_conditional_reread=False, osram_reread_mode='free',
    osram_reread_rank_rtol=1e-5, osram_reread_rank_atol=1e-7)
REVIEWED_CHANGED_CORE = {'gcnet_missing_m3/model.py', 'gcnet_missing_m3/osram.py',
                         'gcnet_missing_m3/train_gcnet.py'}


def parent_path(dataset, seed):
    if dataset == 'CMUMOSI' and seed in (66, 67, 68):
        return MOSI66 if seed == 66 else UNIFIED / dataset / f'seed_{seed}' / 'flat'
    if dataset == 'CMUMOSEI' and seed == 66:
        return MOSEI66
    raise ValueError('Only MOSI66/67/68 and MOSEI66 parent replay is supported')


def mask_path(dataset, seed):
    parent_path(dataset, seed)
    return UNIFIED / dataset / 'seed_66' / 'flat' if dataset == 'CMUMOSEI' else parent_path(dataset, seed)


def _canonical(value):
    return json.loads(json.dumps(value))


def _config(raw, dataset, seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = TrainConfig(**raw)
    required = dict(dataset=dataset, seed=seed, epochs=150 if dataset == 'CMUMOSI' else 60,
        learning_rate=3e-4 if dataset == 'CMUMOSI' else 1e-4, batch_size=32,
        weight_decay=1e-5, optimizer='adam', missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluation_protocol='official', training_objective='emotion-only', train_rate_mode='cyclic',
        mosi_task_mode='regression', task_regression_loss='mse', latent_dim=256,
        dropout=.5, projector_dropout=.1, osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
        osram_readout_fusion='flat', osram_adapter_hidden_dim=0, osram_meaningful_block='none',
        osram_tasr=False, osram_source_role=False, paired_history_views=False,
        completion_path='none', classification_completion=False, osram_disable_local_skip=False,
        initial_backbone_checkpoint=None, fusion_type='mean', emotion_loss_mode='sample-mean',
        **NEW_DEFAULTS)
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError('Original frozen Flat protocol differs: ' + key)
    current = _canonical(asdict(cfg))
    added = {k: v for k, v in current.items() if k not in raw}
    if any(k not in NEW_DEFAULTS or v != NEW_DEFAULTS[k] for k, v in added.items()):
        raise ValueError('Unreviewed new config defaults')
    return cfg, added


def _build_bundle(dataset, seed, uuid, output, accepted):
    import sklearn
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as mosi
    from experiments.osram_mosei_gcnet_router_20261020 import worker as mosei
    from experiments.osram_mosi_recent_history_20261018 import run as mosi_data
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    environment = dict(python=sys.executable, python_version=sys.version, torch=torch.__version__,
                       numpy=np.__version__, sklearn=sklearn.__version__)
    if tuple(environment[k] for k in ('torch', 'numpy', 'sklearn')) != ('1.13.1+cu117', '1.26.4', '1.3.2'):
        raise ValueError('Use unchanged cmerc-repro environment')
    gpu = guard_gpu(uuid)
    if dataset == 'CMUMOSI':
        manifest, proof = mosi.load_manifest(mosi_data.DATA, mosi_data.DATA_SHA)
        for path, digest in manifest['files'].items():
            if h.sha(path) != digest:
                raise ValueError('MOSI feature changed: ' + path)
    else:
        manifest, proof = mosei.load_manifest()
    parent_identity = accepted['identity']
    expected_files = parent_identity.get('data_files', parent_identity.get('data', {}).get('files'))
    if expected_files is not None and manifest['files'] != expected_files:
        raise ValueError('Parent feature manifest differs')
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent / 'feature_cache' / dataset)
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg, added = _config(parent_identity['config'], dataset, seed)
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset=dataset, num_folder=1, batch_size=32, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), train_loader=train[0],
        validation_loader=validation[0], test_loader=test[0], environment=environment,
        gpu=gpu, data_manifest=proof, data_files=manifest['files'], added_default_config_fields=added,
        split={n: loader.protocol_metadata for n, loader in
               (('train', train[0]), ('validation', validation[0]), ('test', test[0]))})
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    for key in ('split', 'seed_protocol'):
        if _canonical(bundle[key]) != parent_identity[key]:
            raise ValueError('Parent data protocol differs: ' + key)
    for key in ('torch', 'numpy', 'sklearn'):
        if environment[key] != accepted['environment'][key]:
            raise ValueError('Parent environment differs: ' + key)
    factory = mosi.fresh_backbone if dataset == 'CMUMOSI' else mosei.fresh_backbone
    bundle['model'] = scalar.wrap_model(factory(cfg, tuple(dimensions), torch.device('cuda:0')), 'flat')
    if sum(p.numel() for p in bundle['model'].parameters()) != 13509793:
        raise ValueError('Original Large Flat architecture changed')
    return bundle


def setup(dataset, seed, uuid, output, rate):
    path, rate = parent_path(dataset, seed), validate_rate(rate)
    accepted = json.loads((path / 'SUMMARY.json').read_text())
    budget = 150 if dataset == 'CMUMOSI' else 60
    if accepted['status'] != 'complete' or accepted['epoch'] != budget:
        raise ValueError('Complete accepted parent required')
    selected = accepted['best_by_rate'][rate]
    checkpoint = path / selected['checkpoint']
    digest = h.sha(checkpoint)
    if digest != selected['checkpoint_sha256']:
        raise ValueError('Selected frozen checkpoint SHA256 differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != selected['epoch'] or _canonical(state['identity']) != accepted['identity']:
        raise ValueError('Selected frozen checkpoint epoch/identity differs')
    if 'selection' in state and validate_rate(state['selection']['rate']) != rate:
        raise ValueError('Selected checkpoint rate differs')
    bundle = _build_bundle(dataset, seed, uuid, output, accepted)
    bundle['model'].load_state_dict(state['model'], strict=True)
    reader = FrozenFlatReader(bundle['model'])
    core = {p: sha for p, sha in accepted['identity']['source_hashes'].items()
            if p.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/'))}
    changes = {p: dict(parent=sha, current=h.sha(ROOT / p)) for p, sha in core.items()
               if h.sha(ROOT / p) != sha}
    if set(changes) - REVIEWED_CHANGED_CORE:
        raise ValueError('Unreviewed changed core source: ' + str(set(changes) - REVIEWED_CHANGED_CORE))
    identity = dict(dataset=dataset, seed=seed, rate=rate, parent_path=str(path),
        parent_id=digest, checkpoint=str(checkpoint), checkpoint_sha256=digest, epoch=state['epoch'],
        summary_sha256=h.sha(path / 'SUMMARY.json'), config=asdict(bundle['cfg']),
        environment=bundle['environment'], parent_frozen=True, train_rate=rate, test_rates=[rate],
        masks_parent_path=str(mask_path(dataset, seed)),
        parent_rng_repair=accepted['identity'].get('rng_repair'),
        added_default_config_fields=bundle['added_default_config_fields'], core_source_differences=changes,
        core_difference_review='Only disabled SourceRelation/V3/MoE/reread extensions and optional Flat callback; original Flat branch unchanged.',
        saved_prediction_replay_required=True,
        parent_comparison='Own selected per-rate parent; repaired MOSEI parent is distinct from earlier unified control.')
    bundle['parent_identity'] = identity
    return bundle, reader, identity


def test_cache(bundle, reader, rate):
    rate = validate_rate(rate)
    proof = bundle['parent_identity']
    if proof['rate'] != rate:
        raise ValueError('Test rate must match frozen parent')
    path = Path(proof['parent_path'])
    accepted = json.loads((path / 'SUMMARY.json').read_text())
    mask_root = Path(proof['masks_parent_path'])
    mask_summary = json.loads((mask_root / 'SUMMARY.json').read_text())
    records = torch.load(mask_root / 'fixed_masks.pt', map_location='cpu', weights_only=False)['test'][rate]
    digest = hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in records)).hexdigest()
    expected = mask_summary['identity']['mask_hashes']['test'][rate]
    own = accepted['identity'].get('mask_hashes', {}).get('test')
    if own is None:
        own = accepted['identity']['mask_reference']['hashes']
    if digest != expected or digest != own[rate]:
        raise ValueError('Frozen parent exact test mask hash differs')
    selected = accepted['best_by_rate'][rate]
    pred_path = path / selected['predictions']
    if h.sha(pred_path) != selected['prediction_sha256']:
        raise ValueError('Frozen parent selected prediction hash differs')
    trainer, loader = bundle['trainer'], bundle['test_loader']
    device = next(reader.parent.parameters()).device
    rows = []
    with h.isolated_rng(66 + 99193), torch.no_grad():
        loader.sampler.set_epoch(0)
        for index, raw in enumerate(loader):
            if index >= len(records):
                raise ValueError('Exact test mask batch count differs')
            data = trainer._move_batch(raw, device)
            meta, record = scalar.metadata(data), records[index]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, record[key])
            a = record['availability'].to(device)
            view = trainer._prepare_view_from_primary_masks(data, a, a, bundle['dimensions'])
            frame = reader(view['incomplete'], a, view['qmask'], view['umask'], view['lengths'])
            torch.testing.assert_close(reader.decode_delta(frame, torch.zeros_like(frame.anchor)),
                                       frame.prediction, rtol=0, atol=0)
            rows.append(dict(frame=frame.to('cpu'), labels=view['labels'].cpu(), umask=view['umask'].cpu(),
                conversation_ids=list(map(str, view['conversation_ids'])), metadata=meta,
                availability=a.transpose(0, 1)[data[7].bool()].cpu().numpy()))
    if not rows or len(rows) != len(records):
        raise ValueError('Exact test mask batch count differs')
    predictions = np.concatenate([r['frame'].prediction.transpose(0, 1)[r['umask'].bool()].numpy() for r in rows])
    with np.load(pred_path, allow_pickle=False) as saved:
        for key in rows[0]['metadata']:
            np.testing.assert_array_equal(np.concatenate([r['metadata'][key] for r in rows]), saved[key])
        np.testing.assert_array_equal(np.concatenate([r['availability'] for r in rows]), saved['availability'])
        np.testing.assert_allclose(predictions, saved['prediction'], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(predictions > 0, saved['prediction'] > 0)
        proof['saved_prediction_replay'] = dict(rate=rate, rtol=1e-5, atol=1e-6,
            max_absolute_error=float(np.max(np.abs(predictions - saved['prediction']))),
            prediction_sha256=selected['prediction_sha256'], mask_bank_sha256=digest, classes_exact=True)
    return rows


test_cache.__test__ = False
