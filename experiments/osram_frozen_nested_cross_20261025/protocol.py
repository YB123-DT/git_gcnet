"""Finite seed66 cross-dataset protocol, pinned to completed cyclic Signature."""
import json
from pathlib import Path

DATASETS = ('CMUMOSEI', 'IEMOCAPSix', 'MELD')
RATES = tuple(str(i / 10) for i in range(8))
PARAMETERS = 159235
METHOD = 'nested_gnn_rooted_evidence'
MULTI_SOURCE = 'e0ca94c4c33de0efda2c2f4786fd289067e1ab2c'
IEMOCAP_SOURCE = '21e46f2ecd092345818659694db8ff372672c35e'
REMOTE = Path('/data2/yb/remote_experiments')


def validate_scope(dataset, seed, fold, rate):
    folds = range(1, 6) if dataset == 'IEMOCAPSix' else (1,)
    if dataset not in DATASETS or type(seed) is not int or seed != 66 or type(fold) is not int or fold not in folds:
        raise ValueError('Only MOSEI66/MELD66 fold1 and IEMOCAPSix66 folds1..5 are authorized')
    if isinstance(rate, bool):
        raise ValueError('Only eight rates 0.0 through 0.7 are authorized')
    try:
        rate = str(float(rate))
    except (TypeError, ValueError):
        raise ValueError('Only eight rates 0.0 through 0.7 are authorized') from None
    if rate not in RATES:
        raise ValueError('Only eight rates 0.0 through 0.7 are authorized')
    return rate


def budget(dataset):
    return {'CMUMOSEI': 60, 'MELD': 50, 'IEMOCAPSix': 100}[dataset]


def learning_rate(dataset, epoch):
    if type(epoch) is not int or not 0 <= epoch < budget(dataset):
        raise ValueError('Epoch outside finite stage2 budget')
    return 3e-4


def selection_metric(dataset):
    budget(dataset)
    return 'WF1' if dataset == 'CMUMOSEI' else 'ACC'


def improves(dataset, current, previous):
    metric = selection_metric(dataset)
    return previous is None or current[metric] > previous[metric]


def measured_metrics(dataset, labels, prediction, base):
    if dataset == 'IEMOCAPSix':
        from experiments.osram_signature_iemocap_cyclic_20261025.run import measured_metrics as measured
        return measured(labels, prediction, base)
    budget(dataset)
    from experiments.osram_signature_cyclic_multi_20261025.run import measured_metrics as measured
    return measured(dataset, labels, prediction, base)


def numerical_source(dataset):
    budget(dataset)
    return IEMOCAP_SOURCE if dataset == 'IEMOCAPSix' else MULTI_SOURCE


def numerical_root(dataset):
    family = 'osram_signature_iemocap_cyclic_20261025' if dataset == 'IEMOCAPSix' else 'osram_signature_cyclic_multi_20261025'
    return REMOTE / family / ('source_' + numerical_source(dataset)[:7])


# Scheduler-facing names; numerical_* also makes their meaning clear in audit records.
source_root = numerical_root
source_commit = numerical_source


def reference_dir(dataset, fold, rate):
    rate = validate_scope(dataset, 66, fold, rate)
    if dataset == 'IEMOCAPSix':
        return REMOTE / 'osram_signature_iemocap_cyclic_20261025/execution_v1/train' / f'IEMOCAPSix_66_fold{fold}_{rate}'
    return REMOTE / 'osram_signature_cyclic_multi_20261025/execution_v2/train' / f'{dataset}_66_{rate}'


def validate_epoch_trace(dataset, row, original):
    for key in ('epoch', 'optimizer_steps', 'training_rates', 'mask_trace_sha256', 'N'):
        if row.get(key) != original.get(key):
            raise ValueError('Training sample/mask/update trace differs from Signature: ' + key)
    if row.get('learning_rate') != learning_rate(dataset, row['epoch'] - 1):
        raise ValueError('Learning rate differs from constant 3e-4')


def validate_reference(reference, parent, config, mask_sha, dataset, fold, rate):
    rate = validate_scope(dataset, 66, fold, rate)
    expected = dict(mode='signature', dataset=dataset, seed=66, evaluation_rate=rate,
        source_commit=numerical_source(dataset), smoke=False, stage2_epochs=budget(dataset),
        cyclic_training=True, training_rates=list(RATES), mask_protocol='gcnet-random-mask-v1',
        effective_stage2_config=config, test_mask_sha256=mask_sha, parent=parent)
    if dataset == 'IEMOCAPSix':
        expected['fold'] = fold
    expected = json.loads(json.dumps(expected))
    actual = reference['identity']
    if (reference.get('status') != 'complete' or reference.get('epoch') != budget(dataset)
            or reference.get('total_parameters') != 281856
            or reference.get('parent_id') != parent['parent_id']
            or config.get('fold') != fold
            or any(actual.get(key) != value for key, value in expected.items())):
        raise ValueError('Completed Signature own-parent/config/mask/source/fold reference differs')
