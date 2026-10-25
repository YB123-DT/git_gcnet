"""Finite cyclic Signature extension preserves dataset task and selection rules."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_scope_and_predeclared_budgets():
    from experiments.osram_signature_cyclic_multi_20261025.run import budget, validate_scope, learning_rate
    for dataset, seeds, epochs in [('CMUMOSI',(66,67),150),('CMUMOSEI',(66,67,68),60),('MELD',(66,),50)]:
        for seed in seeds:
            validate_scope(dataset,seed)
            assert budget(dataset) == epochs
    for dataset,seed in [('CMUMOSI',68),('MELD',67),('IEMOCAP',66)]:
        with pytest.raises(ValueError):
            validate_scope(dataset,seed)
    assert learning_rate('CMUMOSI',99) == 3e-4
    assert learning_rate('CMUMOSI',100) == 3e-5
    assert learning_rate('CMUMOSEI',59) == 3e-4
    assert learning_rate('MELD',49) == 3e-4


def test_selection_is_dataset_specific_and_strict():
    from experiments.osram_signature_cyclic_multi_20261025.run import selection_metric, improves
    assert selection_metric('MELD') == 'ACC'
    assert selection_metric('CMUMOSI') == 'WF1'
    assert not improves('MELD',{'ACC':61.,'WF1':59.},{'ACC':61.,'WF1':58.})
    assert improves('MELD',{'ACC':61.1,'WF1':57.},{'ACC':61.,'WF1':58.})
    assert improves('CMUMOSEI',{'ACC':81.,'WF1':82.},{'ACC':83.,'WF1':81.})


def test_classification_metrics_use_same_logits_and_seven_class_macro_recall():
    from experiments.osram_signature_cyclic_multi_20261025.run import measured_metrics
    labels=np.arange(7)
    logits=np.full((7,7),-2.,dtype=np.float32)
    logits[np.arange(7),labels]=2.
    baseline=logits.copy()
    baseline[0]=logits[1]
    result=measured_metrics('MELD',labels,logits,baseline)
    assert result['ACC'] == result['UA'] == result['WF1'] == 100.
    assert result['N'] == 7 and result['corrections'] == 1 and result['harms'] == 0
    assert 0 < result['CE'] < .2
    assert result['support'] == [1]*7


def test_regression_mse_keeps_gold_zero_but_polarity_metrics_exclude_it():
    from experiments.osram_signature_cyclic_multi_20261025.run import measured_metrics
    result=measured_metrics('CMUMOSI',np.array([-1.,0.,1.]),np.array([-1.,3.,1.]),np.array([1.,0.,1.]))
    assert result['ACC'] == result['WF1'] == 100.
    assert result['MSE'] == 3.
    assert result['N'] == 2 and result['corrections'] == 1


def test_evaluate_preserves_class_axis():
    from experiments.osram_signature_cyclic_multi_20261025.run import evaluate
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p=torch.nn.Parameter(torch.zeros(()))
        def forward(self,frame):
            return self.p
    class Frame:
        valid=torch.ones(2,1,dtype=torch.bool)
        prediction=torch.tensor([[[3.,0.,0.,0.,0.,0.,0.]],[[0.,3.,0.,0.,0.,0.,0.]]])
        def to(self,device):
            return self
    frame=Frame()
    rows=[dict(frame=frame,metadata=dict(labels=np.array([0,1])),availability=np.ones((2,3),dtype=bool))]
    reader=SimpleNamespace(decode_delta=lambda frame,delta:frame.prediction+delta)
    score,arrays=evaluate({'cfg':SimpleNamespace(dataset='MELD')},reader,Block(),rows)
    assert arrays['prediction'].shape == (2,7)
    assert score['ACC'] == 100.
