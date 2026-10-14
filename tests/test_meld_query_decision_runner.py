"""Ownership, paired scoring, and fixed-decoder diagnostic fit contracts."""
import importlib
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.osram_meld_query_decision_20261014.learning import LightReadout, QueryCorrector


def runner():
    return importlib.import_module('experiments.osram_meld_query_decision_20261014.run')


def data():
    torch.manual_seed(15)
    n = 12
    return dict(local=torch.randn(n, 4), memory=torch.randn(n, 2, 3, 3),
                queries=F.normalize(torch.randn(n, 4, 2, 3), dim=-1),
                keys=F.normalize(torch.randn(n, 2, 3, 3), dim=-2),
                availability=torch.tensor([[1., 0., 1.]] * n),
                labels=torch.arange(n) % 3, donors=torch.arange(n).roll(6))


def test_paired_counts_and_fixed_class_macro_recall():
    r = runner()
    labels = np.array([0, 0, 1, 2])
    a = np.eye(3)[[0, 1, 0, 2]]
    b = np.eye(3)[[0, 0, 1, 0]]
    counts = r.paired_counts(labels, a, b)
    assert counts['corrections'] == 2 and counts['harms'] == 1
    assert counts['N'] == 4
    score = r.metrics(labels, np.pad(b, ((0, 0), (0, 1))), num_classes=4)
    assert score['UA'] == pytest.approx(50.)  # absent class has zero recall


def test_donor_reads_use_recipient_queries_and_keys():
    r, d = runner(), data()
    from experiments.osram_meld_query_decision_20261014.learning import forward_reads
    ids = torch.tensor([0, 1, 2])
    base, gap = r.read_inputs(d, ids, 'donor', None, .001, 1.)
    expected = forward_reads(d['memory'][d['donors'][ids]], d['queries'][ids],
                             d['keys'][ids], d['availability'][ids])
    torch.testing.assert_close(base, expected[0], atol=0, rtol=0)
    torch.testing.assert_close(gap, expected[1], atol=0, rtol=0)
    with pytest.raises(ValueError, match='unmatched'):
        d['donors'][ids[0]] = -1
        r.read_inputs(d, ids, 'donor', None, .001, 1.)


def test_query_fit_freezes_decoder_but_updates_query(tmp_path):
    r, d = runner(), data()
    head = LightReadout(local_dim=4, context_dim=6, hidden_dim=8, num_classes=3)
    base, gap = r.read_inputs(d, torch.arange(12), 'real', None, .001, 1.)
    head.fit_normalizer(d['local'], base, gap, d['availability'])
    # Deliberate stale gradients must be cleared on entry to C.
    head(d['local'], base, gap, d['availability']).sum().backward()
    before = r.module_digest(head)
    query = QueryCorrector(local_dim=4, key_dim=3, num_heads=2, hidden_dim=8,
                           type_dim=2, head_dim=2)
    ids = torch.arange(12)
    report, best_logits = r.fit_stage(head, query, d, d, ids, ids, 'real', tmp_path,
        epochs=2, batch_size=6, seed=66, lr=.001, weight_decay=0., clip=1.,
        read_ridge=.001, residual_strength=1., num_classes=3)
    assert r.module_digest(head) == before
    assert all(p.grad is None and not p.requires_grad for p in head.parameters())
    assert report['gradient']['nonzero_steps'] > 0
    assert best_logits.shape == (12, 3)
    saved = torch.load(tmp_path / 'last.pt', weights_only=False)
    assert torch.count_nonzero(saved['corrector']['delta_mlp.2.weight']) > 0
    assert {'optimizer', 'torch_rng', 'numpy_rng', 'python_rng', 'epoch',
            'readout', 'corrector', 'settings', 'shuffle_rng'} <= saved.keys()
    assert report['frozen_decoder_unchanged']


def test_empty_population_and_bad_budget_fail(tmp_path):
    r, d = runner(), data()
    head = LightReadout(4, 6, 8, 3)
    with pytest.raises(ValueError):
        r.fit_stage(head, None, d, d, torch.tensor([], dtype=torch.long), torch.arange(12),
            'local', tmp_path, epochs=1, num_classes=3)


def test_weight_binding_rejects_changed_hash_or_epoch(tmp_path):
    r = runner()
    source = tmp_path / 'source'
    source.mkdir()
    weights = source / 'best_miss_0p7.pt'
    weights.write_bytes(b'exact weights')
    manifest = tmp_path / 'manifest.json'
    record = dict(checkpoint_sha256=r._sha256(weights), epoch=5, ACC=50., WF1=40.)
    manifest.write_text(json.dumps(dict(sources={str(source): dict(rates={'0.7': record})})))
    checkpoint = dict(config={'seed': 66}, epoch=5, selection_rate=.7)
    assert r.bind_source(source, weights, manifest, .7, checkpoint, {'seed': 66}) == record
    weights.write_bytes(b'changed weights same classes')
    with pytest.raises(ValueError, match='identity'):
        r.bind_source(source, weights, manifest, .7, checkpoint, {'seed': 66})
