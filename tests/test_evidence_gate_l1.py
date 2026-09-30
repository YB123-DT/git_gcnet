import unittest
from dataclasses import replace
import torch
from gcnet_missing_m3.osram import LocalConditionedEvidenceGate
from gcnet_missing_m3.train_gcnet import TrainConfig


class GateL1Tests(unittest.TestCase):
    def test_launcher_only_changes_penalty_from_joint_l2(self):
        import tempfile
        from pathlib import Path
        from tests.test_memory_shift_runner import RunnerTest
        from experiments.osram_local_evidence_gate_20260930 import run as l2
        from experiments.osram_evidence_gate_l1_20260930 import run as l1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            RunnerTest().reference(root)
            self.assertEqual(l1.configuration_dict(66,root),dict(l2.configuration_dict(66,root),osram_local_evidence_gate_reg_type='l1'))
        args=l1.parser().parse_args(['--preflight'])
        args.gpus=['4']
        with self.assertRaises(ValueError): l1.validate_args(args)

    def test_active_mean_and_gradient(self):
        gate = LocalConditionedEvidenceGate(4, 3)
        args = (torch.randn(2,1,4), torch.randn(2,1,3), torch.randn(2,1,3,3),
                torch.tensor([[[1.,0,1]],[[0.,0,0]]]), torch.tensor([[1.,0.]]))
        out = gate(*args)
        self.assertTrue(hasattr(gate, 'regularization_l1'))
        self.assertEqual(gate.regularization_l1.item(),0.)
        with torch.no_grad(): gate.output.bias.fill_(.3)
        out = gate(*args)
        expected = (out[0,0,[0,2]]-1).abs().mean()
        torch.testing.assert_close(gate.regularization_l1,expected)
        gate.regularization_l1.backward()
        self.assertGreater(gate.output.bias.grad.item(),0.)
        self.assertTrue(torch.isfinite(gate.output.bias.grad).all())

    def test_default_preserved_and_selected_penalty(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from types import SimpleNamespace
        c=TrainConfig()
        self.assertTrue(hasattr(c,'osram_local_evidence_gate_reg_type'))
        self.assertEqual(c.osram_local_evidence_gate_reg_type,'l2')
        with self.assertRaises(ValueError): replace(c,osram_local_evidence_gate_reg_type='invalid')
        gate=SimpleNamespace(regularization=torch.tensor(.01),regularization_l1=torch.tensor(.1))
        model=SimpleNamespace(osram=SimpleNamespace(local_evidence_gate=gate))
        self.assertEqual(tr._evidence_gate_penalty(model,c),gate.regularization)
        self.assertEqual(tr._evidence_gate_penalty(model,replace(c,osram_local_evidence_gate_reg_type='l1')),gate.regularization_l1)

    def test_l1_train_epoch_once_and_joint_updates(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        c=TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
            latent_dim=8,osram_output_dim=19,osram_num_heads=2,osram_key_dim=3,osram_value_dim=4,
            osram_bidirectional=False,osram_local_evidence_gate=True,osram_local_evidence_gate_reg_type='l1')
        model=_build_model(c,(4,4,4))
        self.assertTrue(all(p.requires_grad for p in model.osram.parameters()))
        with torch.no_grad(): model.osram.local_evidence_gate.output.bias.fill_(.3)
        penalties=[]
        hook=model.osram.local_evidence_gate.register_forward_hook(lambda m,a,o:penalties.append(float(m.regularization_l1.detach())))
        batch=[torch.randn(3,2,4) for _ in range(6)]
        batch += [torch.zeros(2,3),torch.ones(2,3),torch.tensor([[1.,-1.,1.],[-1.,1.,-1.]]),['a','b']]
        old=model.osram.emotion_adapter[-1].weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        result=tr.train_epoch(model,[batch,batch],optimizer,c,tr._schedules(c,'train'),0,(4,4,4),torch.device('cpu'))
        hook.remove()
        self.assertAlmostEqual(result['loss']-result['classification_loss'],.001*sum(penalties)/2,places=6)
        self.assertEqual(result['local_evidence_gate_regularization_type'],'l1')
        self.assertFalse(torch.equal(old,model.osram.emotion_adapter[-1].weight))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__': unittest.main()
