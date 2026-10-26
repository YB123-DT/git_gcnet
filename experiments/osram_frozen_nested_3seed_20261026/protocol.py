"""Finite addition of seed67/68; completed seed66 experiments stay immutable."""
import json
from pathlib import Path

DATASETS = ('CMUMOSI', 'CMUMOSEI', 'MELD', 'IEMOCAPSix')
SEEDS = (67, 68)
RATES = tuple(str(i / 10) for i in range(8))
PARAMETERS = 159235
METHOD = 'nested_gnn_rooted_evidence'
REMOTE = Path('/data2/yb/remote_experiments')
MULTI_SOURCE = 'e0ca94c4c33de0efda2c2f4786fd289067e1ab2c'
IEMOCAP_SOURCE = '21e46f2ecd092345818659694db8ff372672c35e'
MOSI68_SOURCE = '4d5aa09d999b49d7194f72f6bf75c8ab75edbef1'


def validate_scope(dataset, seed, fold, rate):
    folds = range(1, 6) if dataset == 'IEMOCAPSix' else (1,)
    if (dataset not in DATASETS or type(seed) is not int or seed not in SEEDS
            or type(fold) is not int or fold not in folds):
        raise ValueError('Only seeds67/68, four datasets and all five IEMOCAPSix folds are authorized')
    if isinstance(rate, bool):
        raise ValueError('Only rates0.0 through0.7 are authorized')
    try:
        rate = str(float(rate))
    except (TypeError, ValueError):
        raise ValueError('Only rates0.0 through0.7 are authorized') from None
    if rate not in RATES:
        raise ValueError('Only rates0.0 through0.7 are authorized')
    return rate


def budget(dataset):
    return {'CMUMOSI': 150, 'CMUMOSEI': 60, 'MELD': 50, 'IEMOCAPSix': 100}[dataset]


def learning_rate(dataset, epoch):
    if type(epoch) is not int or not 0 <= epoch < budget(dataset):
        raise ValueError('Epoch outside finite stage2 budget')
    return 3e-4 / 10 if dataset == 'CMUMOSI' and epoch >= 100 else 3e-4


def selection_metric(dataset):
    budget(dataset)
    return 'WF1' if dataset in ('CMUMOSI', 'CMUMOSEI') else 'ACC'


def improves(dataset, current, previous):
    metric = selection_metric(dataset)
    return previous is None or current[metric] > previous[metric]


def measured_metrics(dataset, labels, prediction, base):
    budget(dataset)
    if dataset == 'IEMOCAPSix':
        from experiments.osram_signature_iemocap_cyclic_20261025.run import measured_metrics as measured
        return measured(labels, prediction, base)
    from experiments.osram_signature_cyclic_multi_20261025.run import measured_metrics as measured
    return measured(dataset, labels, prediction, base)


def source_commit(dataset, seed):
    validate_scope(dataset, seed, 1, '0.0')
    if dataset == 'IEMOCAPSix':
        return IEMOCAP_SOURCE
    return MOSI68_SOURCE if (dataset, seed) == ('CMUMOSI', 68) else MULTI_SOURCE


def source_root(dataset, seed):
    sha = source_commit(dataset, seed)
    family = ('osram_signature_iemocap_cyclic_20261025' if dataset == 'IEMOCAPSix'
              else 'osram_signature_cyclic68_20261025' if (dataset, seed) == ('CMUMOSI', 68)
              else 'osram_signature_cyclic_multi_20261025')
    return REMOTE / family / ('source_' + sha[:7])


numerical_source = source_commit
numerical_root = source_root


def reference_dir(dataset, seed, fold, rate):
    rate = validate_scope(dataset, seed, fold, rate)
    if dataset == 'MELD':
        return None
    if dataset == 'IEMOCAPSix':
        return REMOTE/'osram_signature_iemocap_cyclic_20261025/execution_v1/train'/f'IEMOCAPSix_{seed}_fold{fold}_{rate}'
    if (dataset, seed) == ('CMUMOSI', 68):
        return REMOTE/'osram_signature_cyclic68_20261025/execution_v1/train'/rate
    return REMOTE/'osram_signature_cyclic_multi_20261025/execution_v2/train'/f'{dataset}_{seed}_{rate}'


def validate_epoch_trace(dataset, row, original=None):
    epoch, steps, count = row.get('epoch'), row.get('optimizer_steps'), row.get('N')
    rates, digest = row.get('training_rates'), row.get('mask_trace_sha256')
    if (type(epoch) is not int or type(steps) is not int or steps <= 0
            or type(count) is not int or count <= 0 or not isinstance(rates, list)
            or len(rates) != steps or any(isinstance(r, bool) or r not in tuple(map(float, RATES)) for r in rates)
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Incomplete/nonfinite original cyclic sample/mask/update trace')
    if row.get('learning_rate') != learning_rate(dataset, epoch - 1):
        raise ValueError('Learning rate differs from authorized exact expression')
    if original is not None:
        for key in ('epoch', 'optimizer_steps', 'training_rates', 'mask_trace_sha256', 'N'):
            if row.get(key) != original.get(key):
                raise ValueError('Original Signature training trace differs: ' + key)


def validate_reference(reference, parent, config, mask_sha, dataset, seed, fold, rate):
    rate = validate_scope(dataset, seed, fold, rate)
    if reference is None:
        if dataset != 'MELD':
            raise ValueError('Completed same-seed Signature reference is required')
        return
    expected = dict(mode='signature', dataset=dataset, seed=seed, evaluation_rate=rate,
        source_commit=source_commit(dataset, seed), smoke=False, stage2_epochs=budget(dataset),
        cyclic_training=True, training_rates=list(RATES), mask_protocol='gcnet-random-mask-v1',
        effective_stage2_config=config, test_mask_sha256=mask_sha)
    if dataset == 'IEMOCAPSix':
        expected['fold'] = fold
    actual = reference['identity']
    expected = json.loads(json.dumps(expected))
    if (reference.get('status') != 'complete' or reference.get('epoch') != budget(dataset)
            or reference.get('total_parameters') != 281856
            or reference.get('parent_id') != parent['parent_id']
            or config.get('dataset') != dataset or config.get('seed') != seed or config.get('fold') != fold
            or any(actual.get(k) != v for k, v in expected.items())):
        raise ValueError('Completed same-seed Signature protocol/config/mask differs')
    old = actual['parent']
    # Provenance descriptions may grow. The protected scientific identities may not.
    keys = ('parent_id', 'checkpoint_sha256', 'checkpoint', 'parent_path', 'config', 'seed', 'fold',
            'selected_prediction', 'selected_prediction_sha256', 'test_mask_sha256',
            'masks_parent_path', 'summary_sha256', 'config_sha256', 'aggregate_sha256',
            'dataset_sha256', 'environment', 'core_source_differences')
    if any(k in old and json.loads(json.dumps(parent.get(k))) != old[k] for k in keys):
        raise ValueError('Completed same-seed Signature own-parent identity differs')
