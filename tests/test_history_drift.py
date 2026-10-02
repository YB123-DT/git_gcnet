import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_history_drift_20261002/diagnostic.py'


class DriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PATH.exists():
            spec = importlib.util.spec_from_file_location('drift_diagnostic', PATH)
            cls.d = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.d)

    def setUp(self):
        self.assertTrue(PATH.exists(), 'Diagnostic helpers not implemented')

    def test_targeted_subset_and_prefix(self):
        a = torch.ones(5, 2, 3)
        a[0, 0] = torch.tensor([1., 0., 0.])
        valid = torch.ones(2, 5)
        valid[1, 4] = 0
        out, meta = self.d.nested_view(a, valid, 'A', 4, prob=1.)
        self.assertTrue((out <= a).all())
        self.assertTrue((out.sum(-1)[valid.T.bool()] >= 1).all())
        self.assertEqual(out[4, 1].sum().item(), 0)
        self.assertTrue(torch.equal(out[0, 0], a[0, 0]))
        self.assertFalse(meta['contrast_mask'][0].any())
        self.assertEqual(meta['prior_deleted_bit_count'][0].sum().item(), 0)
        self.assertEqual(meta['nearest_prior_change_distance'][2, 0].item(), 1)
        self.assertEqual(meta['prior_deleted_modalities'][2, 0, 0].item(), 1)

    def test_identity_and_zero_cosine(self):
        a = torch.tensor([[1., 2.], [0., 0.]])
        out = self.d.vector_drift(a, a)
        self.assertEqual(out['cos'][0].item(), 0)
        self.assertTrue(torch.isnan(out['cos'][1]))
        self.assertFalse(out['valid'][1])
        self.assertEqual(out['rel'].sum().item(), 0)

    def test_anchor_strictly_prior_and_current_equal(self):
        a = torch.ones(4, 1, 3)
        b = a.clone()
        b[1, 0, 0] = 0
        meta = self.d.history_metadata(a, b, torch.ones(1, 4))
        self.assertEqual(meta['contrast_mask'][:, 0].tolist(), [False, False, True, True])
        self.assertEqual(meta['nearest_prior_change_distance'][:, 0].tolist(), [-1, -1, 1, 2])

    def test_mixed_repair_reproducible_without_global_rng(self):
        a = torch.ones(8, 2, 3)
        state = torch.random.get_rng_state()
        b, _ = self.d.nested_view(a, torch.ones(2, 8), 'mixed', 5, prob=1.)
        c, _ = self.d.nested_view(a, torch.ones(2, 8), 'mixed', 5, prob=1.)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(b, c))
        self.assertTrue((b.sum(-1) == 1).all())

    def test_capture_cleanup_on_exception_and_slices(self):
        model = torch.nn.Module()
        model.osram = torch.nn.Module()
        b = model.osram
        b.osram_readout_fusion = 'flat'
        b.bidirectional = False
        b.latent_dim = 256
        b.emotion_adapter = torch.nn.Identity()
        model.eval()
        x = torch.zeros(2, 1, 4352)
        x[..., :256] = 1
        x[..., 256:768] = 2
        def run():
            b.emotion_adapter(x)
            return torch.zeros(2, 1, 1), torch.zeros(2, 1, 1600)
        out = self.d.capture(model, run)
        self.assertEqual(out['base'].shape, (2, 1, 512))
        self.assertEqual(out['gap'].shape, (2, 1, 3, 512))
        self.assertTrue((out['base'] == 2).all())
        def fail():
            b.emotion_adapter(x)
            raise RuntimeError('forward failure')
        with self.assertRaisesRegex(RuntimeError, 'forward failure'):
            self.d.capture(model, fail)
        self.assertEqual(len(b.emotion_adapter._forward_pre_hooks), 0)
        x[..., 768] = 1
        with self.assertRaisesRegex(AssertionError, 'backward context'):
            self.d.capture(model, run)

    def test_metrics_mask_inactive_gap_and_threshold_zero(self):
        snapshot = dict(local=torch.ones(2, 1, 256), base=torch.ones(2, 1, 512),
                        gap=torch.ones(2, 1, 3, 512), hidden=torch.ones(2, 1, 1600),
                        prediction=torch.tensor([[[0.]], [[1.]]]))
        other = {k: v.clone() for k, v in snapshot.items()}
        other['gap'][:] = float('nan')
        other['prediction'][0] = -1
        result = self.d.anchor_metrics(snapshot, other, torch.ones(2, 1, 3))
        self.assertEqual(result['delta_local'].sum().item(), 0)
        self.assertTrue(torch.isnan(result['gap_cos']).all())
        self.assertEqual(result['gap_abs'].sum().item(), 0)
        self.assertFalse(result['sign_flip'][0].item())
        self.assertTrue(result['mathematical_sign_flip'][0].item())


if __name__ == '__main__':
    unittest.main()
