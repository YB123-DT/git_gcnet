"""Own-seed accepted Flat parents, exact saved masks, and CPU-only preflight.

Numerical imports remain lazy so dataset paths can be bound before config is
imported. No checkpoint, mask, historical metadata or old preparation is edited.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path

from .protocol import (DATASETS, SEEDS, RATES, REMOTE, budget, reference_dir,
                       source_commit, source_root, validate_scope)

ROOT = Path(__file__).resolve().parents[2]
UNIFIED = REMOTE/'osram_cmerc_sentiment_3seed_20261023/execution_v1/runs'
MELD_FEATURE = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
MELD_LABEL_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
IEMOCAP_LABEL_SHA = 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'
IEMOCAP_DATA = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
ENVIRONMENT = dict(torch='1.13.1+cu117', numpy='1.26.4', sklearn='1.3.2')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_rate(rate):
    return validate_scope('MELD', 67, 1, rate)


def registry_path(dataset):
    filename = ('osram_meld_wd_seed66_20261018/BASELINE.json' if dataset == 'MELD'
                else 'osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json')
    return source_root(dataset, 67)/'experiments'/filename


def meld_case(seed, path=None):
    validate_scope('MELD', seed, 1, '0.0')
    path = Path(path) if path is not None else registry_path('MELD')
    registry = read_json(path)
    matches = [r for r in registry['runs'] if r['seed'] == seed]
    if len(matches) != 1 or registry['baseline_id'] != 'meld-flat1600-lr1e-5-wd1e-5-seeds66-68-20261018':
        raise ValueError('Require accepted own-seed MELD baseline registry')
    registered = matches[0]
    config = dict(registry['config_without_seed'], seed=seed)
    if (config['learning_rate'] != 1e-5 or config['weight_decay'] != 1e-5
            or registered['identity']['dataset_sha256'] != MELD_LABEL_SHA
            or registered['identity']['numerical_commit'] != 'b4a313841f098487a6482d2712bdac94aa8e319e'):
        raise ValueError('Protected MELD baseline/data/numerical identity differs')
    refs = {rate: dict(checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
        selection=r['selected_predictions'], selection_sha256=r['prediction_sha256'],
        mask_sha256=r['availability_sha256'], epoch=r['epoch'])
        for rate, r in registered['artifacts'].items()}
    if set(refs) != set(RATES):
        raise ValueError('Require all eight own-seed MELD references')
    return dict(dataset='MELD', seed=seed, fold=1, output=registered['output'],
        expected_config=config, evaluation_references=refs, registered=registered,
        registry_sha256=sha(path), registry_path=str(path), dataset_sha256=MELD_LABEL_SHA,
        original_selection=registry['protocol'])


def iemocap_case(seed, fold, path=None):
    validate_scope('IEMOCAPSix', seed, fold, '0.0')
    path = Path(path) if path is not None else registry_path('IEMOCAPSix')
    registry = read_json(path)
    matches = [j for j in registry['jobs'] if j['id'] == f'iemocap6_Original_Flat_seed{seed}_fold{fold}']
    if (registry.get('status') != 'complete' or registry.get('completed_fold_runs') != 30
            or len(matches) != 1 or matches[0]['status'] != 'complete'
            or not matches[0]['summary_valid'] or matches[0]['failures']):
        raise ValueError('Require accepted complete own-seed/fold IEMOCAPSix registry')
    output = Path(matches[0]['output'])
    refs = {rate: dict(checkpoint=str(output/f'best_miss_{rate.replace(".", "p")}.pt'),
        checkpoint_sha256=r['checkpoint_sha256'], selection_sha256=r['selection_sha256'],
        mask_sha256=r['mask_sha256'], epoch=r['epoch'], expected_metrics=r)
        for rate, r in registry['datasets']['IEMOCAPSix']['folds'][str(seed)][str(fold)]['rates'].items()}
    if set(refs) != set(RATES):
        raise ValueError('Require all eight own-fold IEMOCAPSix references')
    return dict(dataset='IEMOCAPSix', seed=seed, fold=fold, output=str(output),
                evaluation_references=refs, registry_sha256=sha(path), registry_path=str(path))


def parent_path(dataset, seed, fold=1):
    validate_scope(dataset, seed, fold, '0.0')
    if dataset == 'MELD':
        return Path(meld_case(seed)['output'])
    if dataset == 'IEMOCAPSix':
        return Path(iemocap_case(seed, fold)['output'])
    return UNIFIED/dataset/f'seed_{seed}'/'flat'


def _preparation(dataset, seed):
    name = ('osram_signature_iemocap_cyclic_20261025' if dataset == 'IEMOCAPSix'
            else 'osram_signature_stage2_extension_20261025' if (dataset, seed) == ('CMUMOSI', 68)
            else 'osram_signature_cyclic_multi_20261025')
    module = importlib.import_module(f'experiments.{name}.prepare')
    if module.ROOT.resolve() != source_root(dataset, seed).resolve():
        raise ValueError('Original preparation escaped the dataset/seed pinned source')
    return module


def configure_environment(dataset, seed, fold, output):
    validate_scope(dataset, seed, fold, '0.0')
    if dataset == 'IEMOCAPSix':
        os.environ.update(GCNET_DATASET_ROOT=str(IEMOCAP_DATA),
            GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/dataset))
        os.environ.pop('GCNET_MELD_PICKLE', None)
    elif dataset == 'MELD':
        # Match the accepted parent feature source; e0 contains the reviewed bridge.
        os.environ.update(GCNET_MELD_PICKLE=str(MELD_FEATURE),
            GCNET_DATASET_ROOT=str(REMOTE/'osram_meld_write_retention_20261014/training_b4a3138/dataset'),
            GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/dataset))
    else:
        worker = importlib.import_module(f'experiments.osram_{"mosi_gcnet_router_3seed_20261019" if dataset == "CMUMOSI" else "mosei_gcnet_router_20261020"}.worker')
        if dataset == 'CMUMOSI':
            data = importlib.import_module('experiments.osram_mosi_recent_history_20261018.run')
            manifest, _ = worker.load_manifest(data.DATA, data.DATA_SHA)
        else:
            manifest, _ = worker.load_manifest()
        os.environ.update(GCNET_DATASET_ROOT=manifest['dataset_root'],
            GCNET_CACHE_ROOT=str(Path(output).parent/'feature_cache'/dataset))
        os.environ.pop('GCNET_MELD_PICKLE', None)


def _meld_setup(seed, uuid, output, rate):
    import torch
    from osram_four_block_wiring import FrozenFlatReader
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_tasr_20261018 import runner_helpers as h
    original = _preparation('MELD', seed)
    case = meld_case(seed)
    accepted, registered, raw = read_json(Path(case['output'])/'SUMMARY.json'), case['registered'], case['expected_config']
    _validate_meld_summary(case, accepted)
    environment = require_replication_environment(torch)
    _validate_environment(environment)
    if any(environment[k] != registered['environment'][k] for k in ENVIRONMENT):
        raise ValueError('MELD own-parent environment differs')
    gpu = guard_gpu(uuid)
    if sha(MELD_FEATURE) != case['dataset_sha256']:
        raise ValueError('Protected MELD feature changed')
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg = trainer.TrainConfig(**raw)
    added = {k: v for k, v in asdict(cfg).items() if k not in raw}
    if any(k not in original.MELD_ADDED_DEFAULTS or v != original.MELD_ADDED_DEFAULTS[k] for k, v in added.items()):
        raise ValueError('Unreviewed MELD added config default')
    ref = dict(case['evaluation_references'][rate])
    state, selected = _read_classification_artifacts(ref, raw, Path(case['output']), 'MELD', rate)
    train, validation, test, *dimensions = trainer.get_loaders(*('MELD-pickle',)*3,
        dataset='MELD', num_folder=1, batch_size=32, num_workers=0, seed=seed,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    model = _build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    model.load_state_dict(state['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != accepted['parameters']:
        raise ValueError('MELD original Flat parameter count differs')
    proof = dict(dataset='MELD', seed=seed, fold=1, rate=rate, parent_path=case['output'],
        parent_id=ref['checkpoint_sha256'], checkpoint=ref['checkpoint'],
        checkpoint_sha256=ref['checkpoint_sha256'], epoch=ref['epoch'],
        summary_sha256=registered['summary_sha256'], config=asdict(cfg), environment=environment,
        parent_frozen=True, train_rates=list(map(float, RATES)), test_rates=[rate],
        baseline_registry_sha256=case['registry_sha256'], dataset_sha256=case['dataset_sha256'],
        original_selection=case['original_selection'], selected_prediction=ref['selection'],
        selected_prediction_sha256=ref['selection_sha256'], test_mask_sha256=ref['mask_sha256'],
        added_default_config_fields=added, saved_prediction_replay_required=True,
        original_numerical_commit=registered['identity']['numerical_commit'],
        numerical_source=dict(source=str(source_root('MELD', seed)), commit=source_commit('MELD', seed)),
        source_bridge='Own-seed registry/config/environment/checkpoint plus strict load and mandatory exact selected logits')
    bundle = dict(model=model, cfg=cfg, trainer=trainer, dimensions=tuple(dimensions),
        train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
        environment=environment, gpu=gpu, parent_identity=proof, selected_arrays=selected,
        data_files={str(MELD_FEATURE):case['dataset_sha256']},
        split={n: loader.protocol_metadata for n, loader in [('train', train[0]), ('validation', validation[0]), ('test', test[0])]})
    bundle['seed_protocol'] = dict(training_seed=seed, split_seed=seed, evaluation_seed=seed,
        train_loader=train[0].protocol_metadata, source='Accepted original own-seed MELD loader; no seed66 substitution')
    return bundle, FrozenFlatReader(model), proof


def setup(dataset, seed, fold, uuid, output, rate):
    rate = validate_scope(dataset, seed, fold, rate)
    configure_environment(dataset, seed, fold, output)
    original = _preparation(dataset, seed)
    if dataset == 'MELD':
        return _meld_setup(seed, uuid, output, rate)
    if dataset == 'IEMOCAPSix':
        return original.setup(seed, fold, uuid, output, rate)
    return original.setup(dataset, seed, uuid, output, rate)


def test_cache(bundle, reader, rate):
    """Replay recorded masks only, and require bit-exact protected parent logits."""
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    dataset, seed = bundle['cfg'].dataset, bundle['cfg'].seed
    proof = bundle['parent_identity']
    rate = validate_scope(dataset, seed, bundle['cfg'].fold, rate)
    if dataset != 'MELD':
        rows = _preparation(dataset, seed).test_cache(bundle, reader, rate)
        if dataset in ('CMUMOSI', 'CMUMOSEI'):
            accepted = read_json(Path(proof['parent_path'])/'SUMMARY.json')
            selected = accepted['best_by_rate'][rate]
            path = Path(proof['parent_path'])/selected['predictions']
            if sha(path) != selected['prediction_sha256']:
                raise ValueError('Own-seed sentiment selected predictions changed')
            predicted = np.concatenate([r['frame'].prediction.transpose(0, 1)[r['umask'].bool()].numpy() for r in rows])
            with np.load(path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(predicted, stored['prediction'])
        return rows
    selected, rows, cursor = bundle['selected_arrays'], [], 0
    trainer, loader = bundle['trainer'], bundle['test_loader']
    device = next(reader.parent.parameters()).device
    with h.isolated_rng(seed + 99193), torch.no_grad():
        loader.sampler.set_epoch(0)
        for raw in loader:
            data = trainer._move_batch(raw, device)
            meta = h.metadata(data)
            n = len(meta['labels'])
            for key, value in meta.items():
                np.testing.assert_array_equal(value, selected[key][cursor:cursor+n])
            availability = torch.zeros((*data[7].shape, 3), device=device, dtype=torch.float32)
            availability[data[7].bool()] = torch.as_tensor(selected['availability'][cursor:cursor+n], device=device)
            availability = availability.transpose(0, 1)
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            frame = reader(view['incomplete'], availability, view['qmask'], view['umask'], view['lengths'])
            torch.testing.assert_close(reader.decode_delta(frame, torch.zeros_like(frame.anchor)), frame.prediction, rtol=0, atol=0)
            rows.append(dict(frame=frame.to('cpu'), labels=view['labels'].cpu(), umask=view['umask'].cpu(),
                conversation_ids=list(map(str, view['conversation_ids'])), metadata=meta,
                availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy()))
            cursor += n
    if not rows or cursor != len(selected['labels']):
        raise ValueError('Own-seed selected MELD sample count differs')
    logits = np.concatenate([r['frame'].prediction.transpose(0, 1)[r['umask'].bool()].numpy() for r in rows])
    np.testing.assert_array_equal(logits, selected['logits'])
    np.testing.assert_array_equal(logits.argmax(-1), selected['predictions'])
    proof['saved_prediction_replay'] = dict(rate=rate, logits_exact=True, classes_exact=True,
        max_absolute_error=0., prediction_sha256=proof['selected_prediction_sha256'],
        mask_bank_sha256=proof['test_mask_sha256'])
    return rows


test_cache.__test__ = False


def _validate_environment(actual):
    if any(actual.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError('Require unchanged cmerc-repro Torch/NumPy/sklearn versions')


def _validate_meld_summary(case, accepted):
    registered, path = case['registered'], Path(case['output'])
    if (sha(path/'SUMMARY.json') != registered['summary_sha256'] or accepted.get('status') != 'complete'
            or accepted.get('config') != case['expected_config'] or accepted.get('identity') != registered['identity']
            or accepted.get('testbest') != registered['testbest'] or accepted.get('parameters') != 13290795
            or accepted.get('paired_masks_equal_through_epoch') != 50):
        raise ValueError('Protected complete own-seed MELD summary/registry differs')
    _validate_environment(accepted['environment'])


def _read_classification_artifacts(ref, raw, root, dataset, rate):
    import numpy as np
    import torch
    if sha(ref['checkpoint']) != ref['checkpoint_sha256']:
        raise ValueError('Protected own-parent checkpoint hash differs')
    state = torch.load(ref['checkpoint'], map_location='cpu', weights_only=False)
    if state['config'] != raw or state['epoch'] != ref['epoch']:
        raise ValueError('Protected own-parent checkpoint config/epoch differs')
    artifact = state['selection_artifact']
    selection = (root/artifact['path']).resolve()
    if (not selection.is_relative_to(root.resolve()) or artifact['sha256'] != ref['selection_sha256']
            or artifact['availability_sha256'] != ref['mask_sha256']
            or ('selection' in ref and selection != Path(ref['selection']).resolve())
            or sha(selection) != ref['selection_sha256']):
        raise ValueError('Protected own-seed selected prediction/mask artifact differs')
    if (state['selection_rate'] != float(rate)
            or state['selection_metric'] != 'accuracy' or state['selection_split'] != 'test'):
        raise ValueError('Own-rate Test-ACC checkpoint selection differs')
    ref['selection'] = str(selection)
    with np.load(selection, allow_pickle=False) as stored:
        arrays = {k: stored[k].copy() for k in stored.files}
    classes = 7 if dataset == 'MELD' else 6
    if (arrays['logits'].shape != (len(arrays['labels']), classes)
            or not np.isfinite(arrays['logits']).all()
            or hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != ref['mask_sha256']):
        raise ValueError('Protected class logits/mask contents differ')
    np.testing.assert_array_equal(arrays['predictions'], arrays['logits'].argmax(-1))
    return state, arrays


def flat_record_availability(record):
    """Select saved valid rows using saved metadata, without a loader or RNG."""
    import numpy as np
    conversations = list(map(str, record['conversation_ids']))
    order = {name: index for index, name in enumerate(dict.fromkeys(conversations))}
    saved = record['availability'].numpy()
    indices = np.asarray(record['utterance_indices'])
    if (saved.ndim != 3 or saved.shape[-1] != 3 or len(order) != saved.shape[1]
            or len(indices) != len(conversations) or not np.equal(indices, indices.astype(np.int64)).all()
            or (indices < 0).any() or (indices >= saved.shape[0]).any()):
        raise ValueError('Saved batch mask/metadata shape or ordering differs')
    return saved[indices.astype(np.int64), [order[name] for name in conversations]]


def preflight_one(dataset, seed, fold, rate, *, cache=None):
    """CPU-only validation of saved records; never builds a model or draws masks."""
    import numpy as np
    import torch
    rate = validate_scope(dataset, seed, fold, rate)
    cache = {} if cache is None else cache
    def check(path, digest):
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha(path)
        if cache[key] != digest:
            raise ValueError('Protected artifact/source hash differs: ' + str(path))
    root = parent_path(dataset, seed, fold)
    accepted = read_json(root/'SUMMARY.json')
    if dataset in ('MELD', 'IEMOCAPSix'):
        case = meld_case(seed) if dataset == 'MELD' else iemocap_case(seed, fold)
        ref = dict(case['evaluation_references'][rate])
        if dataset == 'MELD':
            _validate_meld_summary(case, accepted)
            raw = case['expected_config']
            data_files = {str(MELD_FEATURE): MELD_LABEL_SHA}
        else:
            raw = read_json(root/'config.json')
            if (accepted.get('status') != 'complete' or accepted.get('method') != 'Original_Flat'
                    or accepted.get('smoke_only') is not False or accepted.get('effective_config') != raw
                    or any(raw.get(k) != v for k, v in accepted['config'].items())
                    or accepted['identity']['label_sha256'] != IEMOCAP_LABEL_SHA
                    or accepted['identity']['source_git_sha'] != '785a01242ef1a82b97d5337acbb33ba963df5537'
                    or accepted['testbest']['rates'][rate] != ref['expected_metrics']):
                raise ValueError('Protected IEMOCAP complete original Flat/config/registry differs')
            _validate_environment(accepted['provenance']['environment'])
            if any(not Path(p).is_dir() for p in accepted['provenance']['feature_roots']):
                raise ValueError('Protected IEMOCAP feature root missing')
            data_files = {str(IEMOCAP_DATA/'IEMOCAP/IEMOCAP_features_raw_6way.pkl'): IEMOCAP_LABEL_SHA}
        if any(raw.get(k) != v for k, v in dict(dataset=dataset, seed=seed, fold=fold,
                missing_mask_protocol='gcnet-random-mask-v1', osram_meaningful_block='none',
                osram_readout_fusion='flat', training_objective='emotion-only').items()):
            raise ValueError('Protected own-seed original Flat protocol differs')
        for path, digest in data_files.items():
            check(path, digest)
        state, arrays = _read_classification_artifacts(ref, raw, root, dataset, rate)
        del state
        n = len(arrays['labels'])
        checkpoint, prediction, mask_sha = ref['checkpoint'], ref['selection'], ref['mask_sha256']
        checkpoint_sha, prediction_sha = ref['checkpoint_sha256'], ref['selection_sha256']
    else:
        identity = accepted['identity']
        raw = identity['config']
        if (accepted.get('status') != 'complete' or accepted.get('epoch') != budget(dataset)
                or any(raw.get(k) != v for k, v in dict(dataset=dataset, seed=seed, fold=fold,
                    osram_meaningful_block='none', osram_readout_fusion='flat',
                    missing_mask_protocol='gcnet-random-mask-v1').items())):
            raise ValueError('Require completed same-seed original sentiment parent')
        _validate_environment(accepted['environment'])
        data_files = identity.get('data_files', identity.get('data', {}).get('files'))
        if not data_files:
            raise ValueError('Protected sentiment feature provenance missing')
        for path, digest in data_files.items():
            check(path, digest)
        selected = accepted['best_by_rate'][rate]
        checkpoint, prediction = root/selected['checkpoint'], root/selected['predictions']
        checkpoint_sha, prediction_sha = selected['checkpoint_sha256'], selected['prediction_sha256']
        check(checkpoint, checkpoint_sha)
        check(prediction, prediction_sha)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (json.loads(json.dumps(state['identity'])) != identity or state['epoch'] != selected['epoch']
                or ('selection' in state and validate_rate(state['selection']['rate']) != rate)):
            raise ValueError('Protected sentiment own-checkpoint identity/epoch/rate differs')
        del state
        mask_root = UNIFIED/dataset/'seed_66'/'flat' if dataset == 'CMUMOSEI' else root
        mask_summary = read_json(mask_root/'SUMMARY.json')
        mask_key = ('masks', str(mask_root))
        if mask_key not in cache:
            cache[mask_key] = torch.load(mask_root/'fixed_masks.pt', map_location='cpu', weights_only=False)
        records = cache[mask_key]['test'][rate]
        bank_sha = hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in records)).hexdigest()
        own_hashes = identity.get('mask_hashes', {}).get('test', identity.get('mask_reference', {}).get('hashes', {}))
        if bank_sha != mask_summary['identity']['mask_hashes']['test'][rate] or bank_sha != own_hashes[rate]:
            raise ValueError('Protected own-parent accepted mask bank differs')
        with np.load(prediction, allow_pickle=False) as stored:
            for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices'):
                np.testing.assert_array_equal(np.concatenate([r[key] for r in records]), stored[key])
            np.testing.assert_array_equal(np.concatenate([flat_record_availability(r) for r in records]), stored['availability'])
            n = len(stored['labels'])
            mask_sha = hashlib.sha256(stored['availability'].tobytes()).hexdigest()
    reference = reference_dir(dataset, seed, fold, rate)
    if reference is not None:
        comparison = read_json(reference/'SUMMARY.json')
        i = comparison['identity']
        if (comparison.get('status') != 'complete' or comparison.get('epoch') != budget(dataset)
                or i.get('dataset') != dataset or i.get('seed') != seed
                or i.get('fold', i['effective_stage2_config']['fold']) != fold
                or i.get('evaluation_rate') != rate or i.get('source_commit') != source_commit(dataset, seed)
                or comparison.get('parent_id') != checkpoint_sha or i.get('test_mask_sha256') != mask_sha):
            raise ValueError('Completed same-seed Signature checkpoint/mask/source differs')
        for name, digest in i['source_hashes'].items():
            check(source_root(dataset, seed)/name, digest)
    return dict(dataset=dataset, seed=seed, fold=fold, rate=rate, status='verified', N=n,
        parent=str(root), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        predictions=str(prediction), prediction_sha256=prediction_sha, test_mask_sha256=mask_sha,
        source_root=str(source_root(dataset, seed)), source_commit=source_commit(dataset, seed),
        signature_reference=str(reference) if reference is not None else None,
        checks=dict(checkpoint_and_predictions_sha256=True, saved_mask_sha256=True,
                    own_seed_config=True, environment_and_data_verified=True, no_masks_resampled=True))


def preflight():
    import numpy as np
    import sklearn
    import torch
    actual_environment = dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    _validate_environment(actual_environment)
    cache, rows, failures = {}, [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            for fold in (range(1, 6) if dataset == 'IEMOCAPSix' else (1,)):
                for rate in RATES:
                    try:
                        rows.append(preflight_one(dataset, seed, fold, rate, cache=cache))
                    except Exception as error:
                        failures.append(dict(dataset=dataset, seed=seed, fold=fold, rate=rate, error=repr(error)))
    return dict(status='verified' if len(rows) == 128 and not failures else 'failed',
                expected_branches=128, verified_branches=len(rows), failures=failures, branches=rows,
                environment=actual_environment,
                protocol='CPU read-only; existing accepted artifacts only; no numerical model or mask generation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', required=True)
    args = parser.parse_args(argv)
    result = preflight()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
