from gcnet_missing_m3.train_gcnet import TrainConfig


def test_stage_configs_separate_selection_and_no_student_transfer():
    from experiments.osram_supervised_teacher_20260914.run import teacher_config, student_config
    base=TrainConfig(dataset='CMUMOSI',backbone_type='osram',osram_bidirectional=False,
                     osram_write_step=.6,checkpoint_selection='test-oracle-per-rate')
    first=teacher_config(base)
    assert first.train_rate_mode=='fixed' and first.fixed_missing_rate==0
    assert first.training_objective=='emotion-only'
    assert first.checkpoint_selection=='validation' and not first.evaluate_test
    assert first.teacher_mode=='ema' and first.teacher_checkpoint is None
    second=student_config(base,'projectors.pt')
    assert second.train_rate_mode=='cyclic' and second.training_objective=='joint'
    assert second.checkpoint_selection=='test-oracle-per-rate'
    assert second.teacher_mode=='pretrained-frozen' and second.teacher_checkpoint=='projectors.pt'
    assert second.initial_backbone_checkpoint is None
    for k in ('learning_rate','jepa_weight','temperature','jepa_regression_aggregation',
              'jepa_contrastive_source','num_experts','top_k','epochs'):
        assert getattr(first,k)==getattr(second,k)==getattr(base,k)
