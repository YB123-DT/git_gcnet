import unittest


class RunnerContract(unittest.TestCase):
    def test_only_architecture_changes(self):
        from experiments.osram_base_first_gap_20261011.run import candidate_config
        cfg = dict(dataset='CMUMOSI', seed=66, epochs=100, learning_rate=.001,
                   lr_schedule='constant', backbone_type='osram',
                   training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
                   osram_readout_fusion='flat', osram_output_dim=1600,
                   osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
                   osram_write_step=.6, osram_bidirectional=False, train_rate_mode='cyclic')
        result = candidate_config(cfg)
        self.assertEqual({k:v for k,v in result.items() if cfg.get(k)!=v},
                         {'osram_base_first_gap':True})
        for key,value in [('seed',67),('checkpoint_selection','validation'),
                          ('osram_meaningful_block','nested_gnn_rooted_evidence'),
                          ('paired_history_views',True)]:
            with self.assertRaises(ValueError): candidate_config(dict(cfg,**{key:value}))


if __name__ == '__main__': unittest.main()
