"""Pre-registered finite TASR protocol; never selects an adapter on test labels.

Backbones are protected historical Test-oracle checkpoints, one fixed miss=.3
checkpoint per dataset, not eight independently selected adapters. IEMOCAP's
adapter-only validation conversations were seen by historical backbone training;
this limitation must remain explicit even though adapter selection excludes test.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i/10:.1f}' for i in range(8))
PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV')
PATTERN_BITS = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1))
TRAINING_PROTOCOL = {
    'seed': 66, 'epochs': 30, 'optimizer': 'adam', 'learning_rate': 3e-4,
    'weight_decay': 1e-5, 'batch_size': 32,
    'random_weight': .8, 'persistent_weight': .2,
    'checkpoint_selection': 'validation-random8-ACC-then-CE',
    'persistent_sampling': 'uniform-six-patterns-per-conversation-independent-full-input',
    'adapter_validation_fraction': .1,
    'evaluation': 'one-validation-selected-adapter-checkpoint-for-all-rates-and-patterns',
}


def _read(relative):
    path = ROOT / relative
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _one(rows, **identity):
    matching = [row for row in rows if all(row.get(k) == v for k, v in identity.items())]
    if len(matching) != 1:
        raise ValueError(f'Expected one protected source: {identity}')
    return matching[0]


def _check_config(case):
    cfg = case['expected_config']
    required = dict(dataset=case['dataset'], seed=66, fold=case['fold'],
                    missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
                    osram_readout_fusion='flat', osram_write_step=.6, osram_num_heads=8,
                    osram_key_dim=64, osram_value_dim=64, osram_output_dim=1600,
                    beta_mode='embedded', osram_bidirectional=False, paired_history_views=False,
                    osram_meaningful_block='none')
    if any(cfg.get(name) != value for name, value in required.items()):
        raise ValueError('Protected original Flat configuration changed')
    for name, ref in case['evaluation_references'].items():
        for key in ('checkpoint_sha256', 'selection_sha256', 'mask_sha256'):
            value = ref[key]
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Invalid protected source hash: {name}:{key}')


def build_cases():
    """Read accepted registries without touching remote weights or numerical imports.

    Evaluation references pin historical saved masks, not separate backbones to be
    used at those rates. ``selection=None`` means resolve the actual selection path
    from that protected checkpoint's ``selection_artifact`` and verify its hash;
    no filename is guessed here.
    """
    registry, registry_sha = _read('experiments/osram_meld_wd_seed66_20261018/BASELINE.json')
    persistent, persistent_sha = _read('experiments/osram_persistent_patterns_20261018/SUMMARY.json')
    meld_run = _one(registry['runs'], seed=66)
    config = dict(registry['config_without_seed'], seed=66)
    if config['learning_rate'] != 1e-5 or config['weight_decay'] != 1e-5:
        raise ValueError('Require accepted MELD lr=1e-5 wd=1e-5 Flat baseline')
    meld_refs = {}
    for rate in RATES:
        ref = meld_run['artifacts'][rate]
        meld_refs[rate] = dict(checkpoint=ref['checkpoint'], checkpoint_sha256=ref['checkpoint_sha256'],
                               selection=ref['selected_predictions'], selection_sha256=ref['prediction_sha256'],
                               mask_sha256=ref['availability_sha256'], epoch=ref['epoch'])
    meld = dict(dataset='MELD', classes=7, seed=66, fold=1, anchor_rate='0.3',
                output=meld_run['output'], expected_config=config, evaluation_references=meld_refs,
                source_selection=registry['protocol'], baseline_id=registry['baseline_id'],
                dataset_sha256=meld_run['identity']['dataset_sha256'],
                numerical_source=persistent['datasets']['MELD']['numerical_source'],
                registry_sha256=registry_sha)

    aggregate, aggregate_sha = _read('experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json')
    previous = persistent['datasets']['IEMOCAPSix']
    source = _one(previous['sources'], seed=66, fold=5)
    metadata = source['metadata']
    iemo_config = dict(previous['common_training_config'], seed=66, fold=5)
    iemo_refs = {}
    for rate in RATES:
        ref = aggregate['datasets']['IEMOCAPSix']['folds']['66']['5']['rates'][rate]
        iemo_refs[rate] = dict(checkpoint=str(Path(source['output']) / f'best_miss_{rate.replace(".", "p")}.pt'),
                               checkpoint_sha256=ref['checkpoint_sha256'], selection=None,
                               selection_sha256=ref['selection_sha256'], mask_sha256=ref['mask_sha256'],
                               epoch=ref['epoch'])
    for rate, old in source['references'].items():
        for name in ('checkpoint', 'checkpoint_sha256', 'selection_sha256', 'mask_sha256', 'epoch'):
            if old[name] != iemo_refs[rate][name]:
                raise ValueError(f'IEMOCAP protected registries disagree: {rate}:{name}')
    iemo = dict(dataset='IEMOCAPSix', classes=6, seed=66, fold=5, anchor_rate='0.3',
                output=source['output'], expected_config=iemo_config, evaluation_references=iemo_refs,
                source_selection=metadata['protocol'], source_root=metadata['source_root'],
                source_git_sha=metadata['source_git_sha'], numerical_source=previous['numerical_source'],
                label_sha256=metadata['label_sha256'], feature_roots=metadata['feature_roots'],
                registry_sha256=persistent_sha, aggregate_sha256=aggregate_sha)
    for case in (meld, iemo):
        case.update(case['evaluation_references']['0.3'])
        _check_config(case)
    return {'MELD': meld, 'IEMOCAPSix': iemo}


def persistent_availability(umask, seed: int, epoch: int, batch_index: int):
    """Independent full-input View2 mask; returns availability[T,B,3], ids[B].

    Each conversation receives one of six nonempty patterns uniformly. A private
    CPU generator keyed by (seed,epoch,batch) leaves Torch/NumPy/Python RNG intact,
    including the existing GCNet View1 sampler's stochastic sequence.
    """
    import torch

    if umask.ndim != 2 or not torch.all((umask == 0) | (umask == 1)):
        raise ValueError('umask must be a binary [B,T] tensor')
    key = json.dumps([operator.index(seed), operator.index(epoch), operator.index(batch_index)], separators=(',', ':'))
    derived_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)
    generator = torch.Generator(device='cpu').manual_seed(derived_seed)
    ids_cpu = torch.randint(len(PATTERNS), (umask.shape[0],), generator=generator)
    ids = ids_cpu.to(device=umask.device)
    bits = torch.tensor(PATTERN_BITS, device=umask.device, dtype=torch.float32)[ids]
    availability = torch.where(umask.T.bool()[..., None], bits[None], torch.zeros_like(bits)[None])
    return availability, ids


def adapter_split_indices(train_indices, vids, seed=66, fraction=.1):
    """Reserve ceil(10%) whole training conversations without consulting labels.

    This is adapter validation only: the historical frozen backbone was already
    trained on these original training conversations. Test-fold indices are never
    accepted as candidates by callers; this function only partitions its input.
    """
    try:
        indices = tuple(sorted(operator.index(i) for i in train_indices))
    except TypeError as error:
        raise ValueError('Conversation indices must be integers') from error
    if (len(indices) < 2 or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(vids) for i in indices)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError('Need unique valid training indices and 0 < fraction < 1')
    names = [str(vids[i]) for i in indices]
    if len(set(names)) != len(names):
        raise ValueError('Conversation IDs must be unique')
    size = math.ceil(len(indices)*fraction)
    if size >= len(indices):
        raise ValueError('Adapter validation cannot consume all training conversations')
    ordered = sorted(indices, key=lambda i: (hashlib.sha256(
        json.dumps([operator.index(seed), str(vids[i])], separators=(',', ':')).encode()).hexdigest(), str(vids[i])))
    selected = set(ordered[:size])
    train = tuple(i for i in indices if i not in selected)
    validation = tuple(i for i in indices if i in selected)
    metadata = dict(method='sha256(seed,conversation_id)-sorted-ceil-fraction', seed=seed,
                    fraction=fraction, train_conversations=[str(vids[i]) for i in train],
                    validation_conversations=[str(vids[i]) for i in validation], uses_labels=False,
                    backbone_previously_saw_adapter_validation=True)
    return train, validation, metadata


def validation_key(metrics_by_rate):
    """Maximize validation random8 ACC; ties minimize validation random8 CE.

    Persistent test/validation metrics are deliberately not used for selection.
    Each method retains one adapter checkpoint, evaluated at every condition.
    """
    if set(metrics_by_rate) != set(RATES):
        raise ValueError('Validation requires exactly random missing rates 0.0 through 0.7')
    acc, ce = [], []
    for rate in RATES:
        metrics = metrics_by_rate[rate]
        a, c = float(metrics['ACC']), float(metrics['CE'])
        if not (math.isfinite(a) and math.isfinite(c)):
            raise ValueError('Validation metrics must be finite')
        acc.append(a); ce.append(c)
    return math.fsum(acc)/8, -math.fsum(ce)/8
