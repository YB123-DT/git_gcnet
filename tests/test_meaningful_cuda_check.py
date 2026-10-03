import importlib
import unittest
import ast
import inspect


class CUDAProfileTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.osram_meaningful20_20261003.cuda_check')

    def test_cli_requires_explicit_card_identity_and_real_data(self):
        parser = self.module().parser()
        args = parser.parse_args(['--candidate','otke','--reference','/ref','--dataset','/data.json',
                                  '--output','/out','--gpu-index','1','--gpu-uuid','GPU-good'])
        self.assertEqual(args.gpu_index, '1')
        self.assertEqual(str(args.dataset), '/data.json')

    def test_bad_gpu_rejected_before_torch(self):
        m = self.module()
        mapping = {'1':'GPU-good','4':'GPU-bad'}
        m.check_gpu('1','GPU-good',mapping)
        for index, uuid in [('4','GPU-bad'),('1','GPU-bad'),('5','GPU-good')]:
            with self.assertRaises(ValueError): m.check_gpu(index,uuid,mapping)

    def test_disk_budget_retains_all_versions_and_optimizer(self):
        m = self.module()
        self.assertGreaterEqual(m.artifact_budget(1024**3), 112)
        with self.assertRaises(ValueError): m.artifact_budget(0)

    def test_data_binding_precedes_trainer_import(self):
        tree=ast.parse(inspect.getsource(self.module().profile))
        binding=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Name) and n.func.id=='bind_data_environment']
        imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)
                 and n.module=='gcnet_missing_m3.train_gcnet']
        self.assertEqual(len(binding),1)
        self.assertLess(binding[0],imports[0])

    def test_node_requires_three_initialized_finite_layers(self):
        import torch
        model=torch.nn.Module()
        model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
        for layer in model.layers:
            layer.register_buffer('initialized',torch.tensor(False))
            layer.register_parameter('threshold',torch.nn.Parameter(torch.zeros(2)))
            layer.register_parameter('log_temperature',torch.nn.Parameter(torch.zeros(2)))
        check=self.module().check_finite_state
        with self.assertRaises(RuntimeError): check(model,'node')
        for layer in model.layers: layer.initialized.fill_(True)
        self.assertEqual(check(model,'node')['node_initialized_layers'],3)
        model.layers[1].threshold.data.fill_(float('nan'))
        with self.assertRaises(RuntimeError): check(model,'node')
        with self.assertRaises(RuntimeError): check(model,'otke')

    def test_scan_mocks_deleted_before_peak_reset(self):
        source=inspect.getsource(self.module().profile)
        self.assertIn('del bscan, scan',source)
        self.assertLess(source.index('del bscan, scan'),source.index('reset_peak_memory_stats'))


if __name__ == '__main__': unittest.main()
