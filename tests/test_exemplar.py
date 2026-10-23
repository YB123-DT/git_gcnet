import copy
import unittest

import torch


class RetrievalTests(unittest.TestCase):
    def reader(self, use_labels=True):
        from gcnet_missing_m3.exemplar import TrainExemplarReadout
        torch.manual_seed(12)
        return TrainExemplarReadout(8, 10, key_dim=4, neighbors=2,
                                   use_labels=use_labels)

    def test_same_conversation_is_completely_excluded(self):
        r = self.reader()
        q, support = torch.randn(2, 8), torch.randn(4, 8)
        delta, indices = r(q, support, torch.arange(4.).float(),
                           ['a', 'b'], ['a', 'a', 'b', 'c'])
        self.assertEqual(delta.shape, (2, 10))
        for row, cid in zip(indices.tolist(), ['a', 'b']):
            self.assertTrue(all(['a', 'a', 'b', 'c'][i] != cid for i in row if i >= 0))

    def test_empty_legal_bank_has_strict_zero_not_output_bias(self):
        r = self.reader()
        delta, indices = r(torch.randn(1, 8), torch.randn(2, 8), torch.ones(2),
                           ['a'], ['a', 'a'])
        self.assertEqual(delta.count_nonzero().item(), 0)
        self.assertTrue(indices.eq(-1).all())

    def test_feature_only_is_label_invariant_but_label_aware_is_not(self):
        q, support, labels = torch.randn(2, 8), torch.randn(4, 8), torch.arange(4.).float()
        for use_labels in (False, True):
            r = self.reader(use_labels)
            a, _ = r(q, support, labels, ['x', 'x'], ['a', 'b', 'c', 'd'])
            b, _ = r(q, support, labels + 10, ['x', 'x'], ['a', 'b', 'c', 'd'])
            self.assertEqual(torch.equal(a, b), not use_labels)

    def test_gradients_reach_query_support_and_actual_label_value(self):
        r = self.reader()
        q, support = torch.randn(2, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
        delta, _ = r(q, support, torch.tensor([-2., -1., 1., 2.]),
                           ['x', 'y'], ['a', 'b', 'c', 'd'])
        delta.square().mean().backward()
        for grad in (q.grad, support.grad, r.label.weight.grad, r.key.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.count_nonzero().item(), 0)


def tiny_backbone():
    from gcnet_missing_m3.model import MissingM3GraphModel
    return MissingM3GraphModel(base_model='LSTM', adim=2, tdim=3, vdim=4,
        D_e=4, graph_hidden_size=2, n_speakers=2, window_past=1, window_future=1,
        n_classes=1, dropout=0., no_cuda=True, latent_dim=8, num_experts=2,
        top_k=1, projector_dropout=0., predictor_dropout=0., backbone_type='osram',
        osram_output_dim=10, osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
        disable_unused_aux_modules=True, training_objective='emotion-only')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        from gcnet_missing_m3.exemplar import TrainSupportBank, ExemplarAugmentedModel
        torch.manual_seed(23)
        backbone = tiny_backbone()
        features = torch.randn(4, 9)
        self.bank = TrainSupportBank(features, torch.tensor([-2., -1., 1., 2.]),
                                     ['train0', 'train0', 'train1', 'train1'], [0, 1, 0, 1])
        before = torch.get_rng_state().clone()
        self.model = ExemplarAugmentedModel(backbone, self.bank, use_labels=True, seed=66)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.view = dict(incomplete=torch.randn(3, 2, 9),
            availability=torch.tensor([[[1,0,1],[0,1,1]], [[1,1,1],[1,0,0]], [[0,0,0],[0,0,0]]]).float(),
            umask=torch.tensor([[1.,1.,0.],[1.,1.,0.]]), qmask=torch.zeros(2,3),
            lengths=[2,2], conversation_ids=['test0','test1'], labels=torch.randn(2,3))

    def test_disabled_exact_original_output_rng_and_no_extra_state(self):
        m = self.model.eval()
        m.enabled = False
        v = self.view
        state = torch.get_rng_state().clone()
        actual = m(v)
        expected = m.backbone([v['incomplete']], v['availability'], v['qmask'],
                              v['umask'], v['lengths'])[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(any('support' in key for key in m.state_dict()))

    def test_eval_labels_never_read_and_missing_features_cannot_leak(self):
        m = self.model.eval()
        v = copy.deepcopy(self.view)
        with torch.no_grad():
            original = m(v)
            v['labels'].fill_(float('nan'))
            blocks = torch.split(v['incomplete'], (2, 3, 4), -1)
            for i, block in enumerate(blocks):
                block.masked_fill_(~v['availability'][..., i:i+1].bool(), float('nan'))
            changed = m(v)
        self.assertTrue(torch.equal(original, changed))
        self.assertTrue(torch.isfinite(changed).all())
        self.assertEqual(changed[2].count_nonzero().item(), 0)

    def test_forward_scans_once_and_reloads_exactly(self):
        m = self.model.eval()
        scans = []
        hook = m.backbone.osram.register_forward_hook(lambda *a: scans.append(1))
        with torch.no_grad():
            p = m(self.view)
        hook.remove()
        self.assertEqual(len(scans), 1)
        other = copy.deepcopy(m)
        other.load_state_dict(m.state_dict(), strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(p, other(self.view)))
        self.assertEqual(m.bank.manifest(), other.bank.manifest())

    def test_eval_cache_invalidated_on_weight_reload(self):
        m = self.model.eval()
        with torch.no_grad():
            m(self.view)
        self.assertTrue(m._support_cache)
        state = copy.deepcopy(m.state_dict())
        m.load_state_dict(state)
        self.assertFalse(m._support_cache)

    def test_support_missing_modalities_and_query_future_do_not_leak(self):
        m = self.model.eval()
        v = copy.deepcopy(self.view)
        v['availability'][:2] = v['availability'].new_tensor([1, 0, 0])
        with torch.no_grad():
            before = m(v)
            # All queries are A-only: support Text/Visual must be unreachable.
            m.bank.features[:, 2:] += 10000
            m.eval()  # invalidate cache because this test changes support data
            after = m(v)
            self.assertTrue(torch.equal(before, after))
            v['incomplete'][1] += 500
            future_changed = m(v)
            self.assertTrue(torch.equal(after[0], future_changed[0]))

    def test_training_preserves_rng_and_updates_all_expected_paths(self):
        m = self.model.train()
        before = torch.get_rng_state().clone()
        p = m(self.view)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        p[:2].square().mean().backward()
        for parameter in (m.retrieval.output.weight, m.retrieval.key.weight,
                          m.retrieval.label.weight,
                          m.backbone.observed_set.projectors['audio'][1].weight
                          if isinstance(m.backbone.observed_set.projectors['audio'], torch.nn.Sequential)
                          else next(m.backbone.observed_set.projectors['audio'].parameters())):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.count_nonzero().item(), 0)

    def test_support_dropout_does_not_advance_primary_rng_stream(self):
        m = self.model.train()
        for module in m.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .3
        state = torch.get_rng_state().clone()
        m.enabled = False
        m(self.view)
        expected = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        m.enabled = True
        m(self.view)
        self.assertTrue(torch.equal(expected, torch.get_rng_state()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
