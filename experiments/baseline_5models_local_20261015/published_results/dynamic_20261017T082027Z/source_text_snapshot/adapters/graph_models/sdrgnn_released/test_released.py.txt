"""CPU regressions for the released-code-compatible SDR-GNN path."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adapters.graph_models import build_model


def make(version='released_compatible', dims=(3, 4, 2), speakers=9, **extra):
    config = dict(model='SDR-GNN', hidden=4, graph_hidden=2, dropout=0,
                  model_version=version, **extra)
    return build_model(config, dims, 7, speakers)


class ReleasedRegression(unittest.TestCase):
    def test_version_route_and_reject_typo(self):
        released = make()
        self.assertIn('sdrgnn_released', type(released.model).__module__)
        self.assertEqual(released.version, 'SDR-GNN-Released-compatible-MELD-adapted')
        with self.assertRaisesRegex(ValueError, 'model_version'):
            make('released_compatibile')

    def test_default_still_fixed_and_explicit_fixed_same_initialization(self):
        torch.manual_seed(66)
        default = build_model(dict(model='SDR-GNN', hidden=4, dropout=0), (3, 4, 2), 7, 9)
        torch.manual_seed(66)
        explicit = build_model(dict(model='SDR-GNN', model_version='implementation_fixed',
                                    hidden=4, dropout=0), (3, 4, 2), 7, 9)
        self.assertIn('.sdrgnn.', type(default.model).__module__)
        self.assertEqual(default.version, explicit.version)
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)

    def test_official_tuple_and_speaker_graph_computed_without_classifier_gradient(self):
        model = make()
        seen = {}
        handles = [graph.register_forward_hook(
            lambda module, args, output, name=name: seen.update({name: output}))
            for name, graph in [('temporal', model.model.graph_net_temporal),
                                ('speaker', model.model.graph_net_speaker)]]
        output = model(torch.randn(4, 2, 9), torch.tensor([[0, 8, 3, 4], [4, 1, 0, 0]]),
                       torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool), [4, 2])
        for handle in handles:
            handle.remove()
        self.assertEqual(set(seen), {'temporal', 'speaker'})
        for output_tuple in seen.values():
            self.assertIsInstance(output_tuple, tuple)
            self.assertEqual(len(output_tuple), 1)
        torch.testing.assert_close(output['logits'], model.model.smax_fc(seen['temporal'][0]))
        output['logits'].square().mean().backward()
        self.assertGreater(model.model.graph_net_temporal.conv1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.model.graph_net_speaker.parameters()))
        self.assertTrue(torch.isfinite(output['reconstruction']).all())

    def test_attention_keeps_released_batch_time_axis(self):
        model = make()
        shapes = []
        handle = model.model.multihead_attn.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape)))
        model(torch.randn(4, 2, 9), torch.zeros(2, 4, dtype=torch.long),
              torch.ones(2, 4, dtype=torch.bool), [4, 4])
        handle.remove()
        self.assertFalse(model.model.multihead_attn.batch_first)
        self.assertEqual(shapes, [(2, 4, 9)])

    def test_meld_width_and_stable_nine_speaker_relations(self):
        model = make(dims=(300, 1024, 342))
        self.assertEqual(model.model.multihead_attn.embed_dim, 1666)
        self.assertEqual(model.model.multihead_attn.num_heads, 238)
        self.assertEqual(model.model.graph_net_speaker.conv1.num_relations, 81)
        from adapters.graph_models.sdrgnn_released.graph import batch_graphify
        args = (torch.randn(3, 2, 1, 8), torch.tensor([[0, 8, 4], [3, 1, 2]]),
                [3, 3], 9, 2, 2, 'speaker', True)
        _, _, _, mapping = batch_graphify(*args)
        self.assertEqual(mapping['88'], 80)
        self.assertEqual(mapping, {f'{i}{j}': i * 9 + j for i in range(9) for j in range(9)})
        output = model(torch.randn(3, 2, 1666), args[1], torch.ones(2, 3), [3, 3])
        (output['logits'].square().mean() + output['reconstruction'].square().mean()).backward()
        self.assertTrue(torch.isfinite(model.model.multihead_attn.in_proj_weight.grad).all())

    def test_iemocap_original_width_and_heads(self):
        model = make(dims=(1024, 1024, 512), speakers=2)
        self.assertEqual(model.model.multihead_attn.embed_dim, 2560)
        self.assertEqual(model.model.multihead_attn.num_heads, 256)
        output = model(torch.randn(3, 2, 2560), torch.tensor([[0, 1, 0], [1, 0, 1]]),
                       torch.ones(2, 3), [3, 3])
        output['reconstruction'].square().mean().backward()
        self.assertTrue(torch.isfinite(model.model.multihead_attn.in_proj_weight.grad).all())

    def test_iemocap_matches_local_official_state_forward_loss_and_gradients(self):
        """Independent upstream oracle, when the audited local checkout is available."""
        import json
        source = Path(json.loads((Path(__file__).parent / 'PROVENANCE.json').read_text())['upstream_directory'])
        if not (source / 'model_SDRGNN.py').is_file():
            self.skipTest('Audited upstream checkout is unavailable')

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        previous = sys.modules.get('graph')
        try:
            sys.modules['graph'] = load('released_reference_graph', source / 'graph.py')
            official = load('released_reference_model', source / 'model_SDRGNN.py')
            losses = load('released_reference_loss', source / 'loss.py')
        finally:
            if previous is None:
                sys.modules.pop('graph', None)
            else:
                sys.modules['graph'] = previous
        torch.manual_seed(66)
        reference = official.GraphModel('GRU', 1024, 1024, 512, 4, 2, 2, 3, 3, 7,
                                        dropout=0, time_attn=False, no_cuda=True)
        torch.manual_seed(66)
        model = make(dims=(1024, 1024, 512), speakers=2)
        self.assertEqual(set(reference.state_dict()), set(model.model.state_dict()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.model.state_dict()[key], rtol=0, atol=0)
        target = torch.randn(3, 2, 2560)
        speakers = torch.tensor([[0, 1, 0], [1, 0, 0]])
        valid = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        labels = torch.tensor([[0, 1, 2], [3, 0, 0]])
        availability = torch.randint(0, 2, (3, 2, 3)).float()
        logits, recon, hidden = reference([target], speakers, valid, [3, 1])
        output = model(target, speakers, valid, [3, 1])
        for actual, expected in [(output['logits'], logits), (output['reconstruction'], recon[0]),
                                 (output['hidden'], hidden)]:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ce = losses.MaskedCELoss()(logits.transpose(0, 1).contiguous().view(-1, 7),
                                   labels.view(-1), valid)
        rec = losses.MaskedReconLoss()(recon, [target], [availability], valid, 1024, 1024, 512)
        expected_loss = .5 * ce + .5 * rec
        actual_loss, _ = model.loss(output, labels, target, availability, valid)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        expected_loss.backward()
        actual_loss.backward()
        parameters = dict(model.model.named_parameters())
        for key, value in reference.named_parameters():
            if value.grad is None:
                self.assertIsNone(parameters[key].grad)
            else:
                torch.testing.assert_close(parameters[key].grad, value.grad, rtol=0, atol=0)

    def test_variable_length_loss_exactly_official_including_recon_flatten(self):
        from adapters.graph_models.sdrgnn_released.loss import MaskedCELoss, MaskedReconLoss
        torch.manual_seed(66)
        model = make(recon_loss_weight=.3)
        logits = torch.randn(4, 2, 7, requires_grad=True)
        reconstruction = torch.randn(4, 2, 9, requires_grad=True)
        target = torch.randn(4, 2, 9)
        availability = torch.randint(0, 2, (4, 2, 3)).float()
        valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        labels = torch.randint(0, 7, (2, 4))
        ce = MaskedCELoss()(logits.transpose(0, 1).contiguous().view(-1, 7), labels.view(-1), valid)
        recon = MaskedReconLoss()([reconstruction], [target], [availability], valid, 3, 4, 2)
        result, parts = model.loss(dict(logits=logits, reconstruction=reconstruction),
                                   labels, target, availability, valid)
        torch.testing.assert_close(parts['classification'], ce, rtol=0, atol=0)
        torch.testing.assert_close(parts['reconstruction'], recon, rtol=0, atol=0)
        torch.testing.assert_close(result, .7 * ce + .3 * recon, rtol=0, atol=0)
        expected_grads = torch.autograd.grad(.7 * ce + .3 * recon, (logits, reconstruction), retain_graph=True)
        actual_grads = torch.autograd.grad(result, (logits, reconstruction))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A correctly aligned reconstruction mask is deliberately a different version.
        fixed = make('implementation_fixed', recon_loss_weight=.3)
        _, fixed_parts = fixed.loss(dict(logits=logits, reconstruction=reconstruction),
                                    labels, target, availability, valid)
        self.assertNotEqual(float(recon), float(fixed_parts['reconstruction']))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
