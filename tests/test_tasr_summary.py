"""Fail-closed TASR paired summaries: complete runs and identical evidence only."""
import copy
import hashlib
import importlib.util
import json

import numpy as np
import pytest

from experiments.osram_tasr_20261018.protocol import PATTERNS, RATES, TRAINING_PROTOCOL


def api():
    assert importlib.util.find_spec('experiments.osram_tasr_20261018.summarize') is not None, 'Missing TASR summary implementation'
    from experiments.osram_tasr_20261018 import summarize
    return summarize


def write_json(path, obj):
    path.write_text(json.dumps(obj))


def make_runs(root):
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics
    from experiments.osram_tasr_20261018.runner_helpers import paired_counts
    for dataset, classes in (('MELD', 7), ('IEMOCAPSix', 6)):
        for mode in ('neutral', 'real'):
            path = root / f'{dataset}_{mode}'
            evaluation = path / 'evaluation'
            evaluation.mkdir(parents=True)
            labels = np.array([0, 1, 2, 0])
            logits0 = np.eye(classes)[[0, 0, 2, 0]].astype(np.float32)
            logits1 = np.eye(classes)[[0, 1, 2, 0] if mode == 'real' else [0, 0, 2, 1]].astype(np.float32)
            result = {}
            for group, names in (('random', RATES), ('persistent', (*PATTERNS, 'ATV'))):
                conditions = {}
                for name in names:
                    availability = np.tile(np.array([1., 0, 1], dtype=np.float32), (len(labels), 1))
                    if group == 'persistent':
                        bits = np.array([float(x in name) for x in 'ATV'], dtype=np.float32)
                        availability[:] = bits
                    common = dict(labels=labels, speakers=np.array([0, 1, 0, 1]),
                        conversation_ids=np.array(['a', 'a', 'b', 'b']), utterance_indices=np.array([0, 1, 0, 1]), availability=availability)
                    arrays = []
                    scores = []
                    for tag, logits in (('F0', logits0), ('adapter', logits1)):
                        array = dict(common, logits=logits, predictions=logits.argmax(1))
                        np.savez_compressed(evaluation/f'{group}_{name.replace(".", "p")}_{tag}.npz', **array)
                        arrays.append(array)
                        scores.append(dict(classification_metrics(labels, logits, classes),
                            mask_sha256=hashlib.sha256(availability.tobytes()).hexdigest()))
                    conditions[name] = dict(F0=scores[0], adapter=scores[1], paired=paired_counts(*arrays),
                        diagnostics=[dict(valid=4, source_tracking=dict(memory_sum_max_abs_error=1e-7,
                            memory_sum_max_relative_error=2e-7, gap_sum_max_abs_error=3e-7,
                            gap_sum_max_relative_error=4e-7), tasr=dict(gamma_mean=.01, gamma_min=-.1, gamma_max=.1,
                            target_source_read_norm=np.ones((3,3)).tolist(), target_source_gamma_mean=np.zeros((3,3)).tolist()))])
                result[group] = dict(conditions=conditions)
            history = [dict(epoch=e+1, optimizer_steps=2, batches=2, original_random_mask_trace_sha256=f'mask{e}',
                persistent_pattern_conversations={p:2 for p in PATTERNS}, random_actual_missing_fraction=.4,
                persistent_actual_missing_fraction=.5, loss=1/(e+1), random_CE=1., persistent_CE=1., gradient_norm=.1,
                validation={r:dict(ACC=60.+e, UA=40., WF1=55., CE=1/(e+1)) for r in RATES},
                validation_key=[60.+e, -1/(e+1)]) for e in range(30)]
            summary = dict(status='complete', phase='train', dataset=dataset, mode=mode, epoch=30,
                identity=dict(case=dict(dataset=dataset, classes=classes, seed=66, fold=1 if dataset=='MELD' else 5,
                    checkpoint='/protected/anchor.pt', checkpoint_sha256='a'*64, anchor_rate='0.3', expected_config=dict(foo=1)),
                    training=copy.deepcopy(TRAINING_PROTOCOL), mode=mode, source_hashes={'osram.py':'b'*64},
                    split=dict(train_indices=[0],validation_indices=[1],test_indices=[2],
                        backbone_previously_saw_adapter_validation=dataset!='MELD')),
                environment=dict(torch='1.13.1'), validation_mask_hashes={r:'c'*64 for r in RATES},
                checkpoint_references=dict(source='protected'), trainable_parameters=['osram.tasr.score.3.weight'],
                adapter_parameters=59423, total_parameters=13500000, backbone_unchanged=True,
                zero_adapter_logits_bit_exact=True, anchor_parity=dict(checks=dict(selected_logit_max_abs_error=0)),
                selected_epoch=30, validation_best_key=history[-1]['validation_key'],
                peak_memory_mib=123., seconds=10., evaluation=result)
            write_json(path/'SUMMARY.json', summary)
            write_json(path/'history.json', history)
            write_json(evaluation/'RESULTS.json', result)


def test_incomplete_runs_are_pending_without_invented_metrics(tmp_path):
    module = api()
    result = module.summarize(tmp_path)
    assert result['status'] == 'pending'
    assert 'datasets' not in result
    assert len(result['pending']) == 4


def test_complete_summary_recomputes_scores_and_pairs(tmp_path):
    module = api()
    make_runs(tmp_path)
    result = module.summarize(tmp_path)
    assert result['status'] == 'complete'
    for dataset in ('MELD', 'IEMOCAPSix'):
        row = result['datasets'][dataset]
        assert set(row['random']['conditions']) == set(RATES)
        assert set(row['persistent']['conditions']) == {*PATTERNS, 'ATV'}
        assert row['random']['conditions']['0.0']['paired']['F2_vs_F1']['corrections'] == 2
        assert row['random']['conditions']['0.0']['paired']['F2_vs_F0']['corrections'] == 1
        assert row['verdict']['random_protection_pass'] is True
        assert row['verdict']['persistent_F2_extra_value_over_F1'] is True
        assert row['methods']['F2']['diagnostics']['source_tracking_max']['gap_sum_max_abs_error'] == 3e-7
        assert row['methods']['F2']['selected_epoch'] == 30
    text = module.render_markdown(result)
    assert 'INTERNAL DIAGNOSTIC ONLY' in text
    assert 'adapter-only' in text
    assert '0.7' in text and 'ATV' in text
    module.write_outputs(tmp_path/'report', result)
    assert (tmp_path/'report'/'COMPARISON.csv').exists()


@pytest.mark.parametrize('field', ['backbone_unchanged', 'zero_adapter_logits_bit_exact'])
def test_rejects_failed_frozen_or_zero_init_contract(tmp_path, field):
    module = api()
    make_runs(tmp_path)
    path = tmp_path/'MELD_real'/'SUMMARY.json'
    summary = json.loads(path.read_text()); summary[field] = False; write_json(path, summary)
    with pytest.raises(ValueError, match=field):
        module.summarize(tmp_path)


@pytest.mark.parametrize('mutation', ['masktrace', 'valhash', 'anchor', 'budget', 'selected'])
def test_rejects_unpaired_training_or_selection(tmp_path, mutation):
    module = api()
    make_runs(tmp_path)
    path = tmp_path/'MELD_real'
    summary = json.loads((path/'SUMMARY.json').read_text())
    history = json.loads((path/'history.json').read_text())
    if mutation == 'masktrace': history[0]['original_random_mask_trace_sha256'] = 'wrong'
    if mutation == 'valhash': summary['validation_mask_hashes']['0.0'] = 'wrong'
    if mutation == 'anchor': summary['identity']['case']['checkpoint_sha256'] = 'wrong'
    if mutation == 'budget': history[1]['optimizer_steps'] += 1
    if mutation == 'selected': summary['selected_epoch'] = 29
    write_json(path/'SUMMARY.json', summary); write_json(path/'history.json', history)
    with pytest.raises(ValueError): module.summarize(tmp_path)


def test_rejects_wrong_same_shape_sample_identity(tmp_path):
    module = api()
    make_runs(tmp_path)
    path = tmp_path/'MELD_real'/'evaluation'/'random_0p0_adapter.npz'
    with np.load(path) as source: arrays = {key:source[key] for key in source.files}
    arrays['conversation_ids'] = np.array(['different'] * 4)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='conversation_ids'): module.summarize(tmp_path)


def test_incomplete_epoch_cannot_be_reported_complete(tmp_path):
    module = api()
    make_runs(tmp_path)
    path = tmp_path/'MELD_real'/'SUMMARY.json'
    summary = json.loads(path.read_text()); summary['epoch'] = 29; write_json(path, summary)
    with pytest.raises(ValueError, match='30'): module.summarize(tmp_path)


def test_missing_prediction_is_pending_even_when_worker_says_complete(tmp_path):
    module = api()
    make_runs(tmp_path)
    path = tmp_path/'MELD_real'/'evaluation'/'random_0p0_adapter.npz'
    path.unlink()
    result = module.summarize(tmp_path)
    assert result['status']=='pending'
    assert 'datasets' not in result
    assert result['pending'][0]['artifact']=='evaluation/random_0p0_adapter.npz'


def test_reported_metric_tampering_is_not_trusted(tmp_path):
    module = api()
    make_runs(tmp_path)
    root = tmp_path/'MELD_real'
    summary = json.loads((root/'SUMMARY.json').read_text())
    evaluation = json.loads((root/'evaluation'/'RESULTS.json').read_text())
    evaluation['random']['conditions']['0.0']['adapter']['UA'] += .1
    summary['evaluation'] = evaluation
    write_json(root/'SUMMARY.json',summary)
    write_json(root/'evaluation'/'RESULTS.json',evaluation)
    with pytest.raises(ValueError,match='Reported metric'): module.summarize(tmp_path)


def test_metadata_aligned_but_different_F0_logits_are_rejected(tmp_path):
    module = api()
    make_runs(tmp_path)
    root = tmp_path/'MELD_real'
    path = root/'evaluation'/'random_0p0_F0.npz'
    with np.load(path) as source: arrays = {key:source[key] for key in source.files}
    arrays['logits'] += 1  # same probabilities/metrics, but not same backbone readout
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='F0 across runs:logits'): module.summarize(tmp_path)
