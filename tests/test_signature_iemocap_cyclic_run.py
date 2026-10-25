"""IEMOCAP six-class cyclic adapter extension never changes sentiment/MELD runs."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace


def test_only_authorized_six_class_seed_folds_and_budget():
    from experiments.osram_signature_iemocap_cyclic_20261025.run import validate_scope, budget, learning_rate
    for seed in (66,67,68):
        for fold in range(1,6):
            validate_scope(seed,fold)
    for seed,fold in [(65,5),(66,0),(68,6),(True,1)]:
        with pytest.raises(ValueError):
            validate_scope(seed,fold)
    assert budget() == 100
    assert learning_rate(0) == learning_rate(99) == 3e-4
    with pytest.raises(ValueError):
        learning_rate(100)


def test_metrics_preserve_all_six_labels_including_class_zero():
    from experiments.osram_signature_iemocap_cyclic_20261025.run import measured_metrics
    labels=np.arange(6)
    logits=np.eye(6,dtype=np.float32)*3.
    original=logits.copy()
    original[0]=logits[1]
    score=measured_metrics(labels,logits,original)
    assert score['N'] == 6 and score['ACC'] == score['UA'] == score['WF1'] == 100.
    assert score['corrections'] == 1 and score['harms'] == 0
    assert score['support'] == [1]*6
    assert 0 < score['CE'] < .3
    with pytest.raises(ValueError):
        measured_metrics(labels,np.ones((6,7)),np.ones((6,7)))


def test_acc_selects_same_epoch_ua_and_ties_keep_earliest():
    from experiments.osram_signature_iemocap_cyclic_20261025.run import improves
    previous={'ACC':65.,'UA':64.}
    assert improves({'ACC':65.1,'UA':60.},previous)
    assert not improves({'ACC':65.,'UA':70.},previous)
    assert not improves({'ACC':64.,'UA':70.},previous)


def test_evaluation_preserves_six_logit_axis_and_no_grad():
    from experiments.osram_signature_iemocap_cyclic_20261025.run import evaluate
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(()))
        def forward(self,frame):
            assert not torch.is_grad_enabled()
            return self.weight
    class Frame:
        valid=torch.ones(2,1,dtype=torch.bool)
        prediction=torch.eye(6)[:2,None,:]*3
        def to(self,device):return self
    frame=Frame()
    rows=[dict(frame=frame,metadata={'labels':np.array([0,1])},availability=np.ones((2,3)))]
    reader=SimpleNamespace(decode_delta=lambda frame,delta:frame.prediction+delta)
    score,arrays=evaluate(reader,Block(),rows)
    assert arrays['prediction'].shape == (2,6)
    assert score['ACC'] == 100. and score['UA'] == pytest.approx(100/3)
