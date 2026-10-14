import copy
from types import SimpleNamespace

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_meld_write_retention_20261014.run import WriteCapture


def tiny_model():
    osram = OSRAMBackbone(latent_dim=4, output_dim=4, num_heads=2,
                          key_dim=2, value_dim=2, bidirectional=False,
                          alpha_init=.8, write_step=.6, dropout=0).eval()
    # A trained, nonzero adapter is necessary to verify task sensitivity.
    torch.nn.init.normal_(osram.emotion_adapter[-1].weight, std=.2)
    model = SimpleNamespace(osram=osram, smax_fc=torch.nn.Linear(4, 7).eval())
    node = torch.randn(10, 2, 4)
    latent = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
    availability = torch.tensor([[[1., 0., 1.], [1., 1., 0.]]] * 10)
    umask = torch.ones(2, 10)
    umask[1, 7:] = 0
    availability[7:, 1] = 0
    args = (node, latent, availability, torch.zeros(2, 10, dtype=torch.long), umask, [10, 7])
    return model, args


def test_capture_and_original_read_match_true_original_normal_predictions():
    from experiments.osram_meld_write_retention_20261014.task_replay import (
        TaskCapture, original_read, replay_task_logits,
    )
    model, args = tiny_model()
    before = copy.deepcopy(model.osram.state_dict())
    rng = torch.get_rng_state().clone()
    with torch.no_grad(), TaskCapture(model.osram) as target, WriteCapture(model.osram, args[-2]) as writer:
        hidden, _ = model.osram(*args)
        logits = model.smax_fc(hidden)
    for t, memory, keys, *_ in writer.snapshots:
        base, gap = original_read(model.osram, memory, target.queries[t], keys,
                                  args[2][t], args[-2][:, t].bool())
        torch.testing.assert_close(base, target.base[t])
        torch.testing.assert_close(gap, target.gap[t])
        actual = replay_task_logits(model, target.local[t], memory, target.queries[t],
                                    keys, args[2][t], args[-2][:, t].bool())
        torch.testing.assert_close(actual[args[-2][:, t].bool()], logits[t][args[-2][:, t].bool()])
    assert torch.equal(rng, torch.get_rng_state())
    assert '_scan' not in model.osram.__dict__
    for name, value in before.items():
        assert torch.equal(value, model.osram.state_dict()[name])


def test_four_branches_are_paired_and_distance_one_causal_boundary_matches():
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, task_rows
    model, args = tiny_model()
    with torch.no_grad(), TaskCapture(model.osram) as target, WriteCapture(model.osram, args[-2]) as writer:
        hidden, _ = model.osram(*args)
        logits = model.smax_fc(hidden)
    labels = torch.zeros(2, 10, dtype=torch.long)
    rows = list(task_rows(model, writer.snapshots, target, logits, labels, ['c0', 'c1'], .7))
    assert rows
    groups = {}
    for row in rows:
        key = (row['conversation_id'], row['source_t'], row['target_t'], row['modality'])
        groups.setdefault(key, {})[row['branch']] = row
        assert row['target_t'] < (10 if row['conversation_id'] == 'c0' else 7)
        assert row['distance'] in (1, 2, 4, 8)
    for group in groups.values():
        assert set(group) == {'normal', 'decay_only', 'write_only', 'hold'}
        assert group['normal']['delta_ce'] == pytest.approx(0, abs=1e-6)
        if group['normal']['distance'] == 1:
            assert group['normal']['logits'] == pytest.approx(group['decay_only']['logits'])
            assert group['write_only']['logits'] == pytest.approx(group['hold']['logits'])


def test_inactive_gap_and_padding_are_zero_and_capture_restores_on_error():
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, original_read
    model, args = tiny_model()
    memory = torch.randn(2, 2, 2, 2)
    query = torch.randn(2, 4, 2, 2)
    keys = torch.randn(2, 2, 2, 3)
    base, gap = original_read(model.osram, memory, query, keys,
                              torch.tensor([[1., 0., 1.], [0., 0., 0.]]), torch.tensor([True, False]))
    assert torch.count_nonzero(gap[0, [0, 2]]) == 0
    assert torch.count_nonzero(base[1]) == torch.count_nonzero(gap[1]) == 0
    with pytest.raises(RuntimeError):
        with TaskCapture(model.osram):
            raise RuntimeError('deliberate')
    assert '_scan' not in model.osram.__dict__


def test_source_guard_rejects_legacy_no_skip_and_small_flat():
    from experiments.osram_meld_write_retention_20261014.task_replay import validate_source
    good = dict(dataset='MELD', seed=66, missing_mask_protocol='gcnet-random-mask-v1',
                backbone_type='osram', osram_bidirectional=False, osram_readout_fusion='flat',
                osram_meaningful_block='none', training_objective='emotion-only',
                osram_disable_local_skip=False, osram_output_dim=1600,
                osram_adapter_hidden_dim=0)
    validate_source(good)
    historical = dict(good)
    historical.pop('osram_disable_local_skip')
    validate_source(historical)
    for name, value in [('missing_mask_protocol', 'gcnet-legacy-fixed-v1'),
                         ('osram_disable_local_skip', True), ('osram_adapter_hidden_dim', 512)]:
        with pytest.raises(ValueError):
            validate_source({**good, name: value})


def test_aggregation_deduplicates_source_modalities_and_keeps_distances_separate(tmp_path):
    import gzip
    import json
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, task_rows
    from experiments.osram_meld_write_retention_20261014.task_analyze import aggregate
    model, args = tiny_model()
    with torch.no_grad(), TaskCapture(model.osram) as target, WriteCapture(model.osram, args[-2]) as writer:
        hidden, _ = model.osram(*args)
        logits = model.smax_fc(hidden)
    rows = list(task_rows(model, writer.snapshots, target, logits,
                          torch.zeros(2, 10, dtype=torch.long), ['c0', 'c1'], .7))
    filename = tmp_path / 'rows.jsonl.gz'
    with gzip.open(filename, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    result = aggregate([filename])
    for d in (1, 2, 4, 8):
        summary = next(g for g in result['groups'] if g['population'] == 'unique_target'
                       and g['distance'] == d and g['branch'] == 'normal' and g['availability'] == 'ALL')
        expected = len({(r['conversation_id'], r['target_t']) for r in rows
                        if r['distance'] == d and r['branch'] == 'normal'})
        assert summary['N'] == expected
        assert summary['mean_delta_ce'] == pytest.approx(0., abs=1e-6)
        assert 'mean_source_relative_error' not in summary
        assert sum(summary['class_support']) == expected
        assert len(summary['class_recall_percent']) == 7


def test_padding_negative_labels_are_not_ce_targets():
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, task_rows
    model, args = tiny_model()
    labels = torch.zeros(2, 10, dtype=torch.long)
    labels[1, 7:] = -1
    with torch.no_grad(), TaskCapture(model.osram) as target, WriteCapture(model.osram, args[-2]) as writer:
        hidden, _ = model.osram(*args)
        logits = model.smax_fc(hidden)
    assert list(task_rows(model, writer.snapshots, target, logits, labels, ['c0', 'c1'], .7))
