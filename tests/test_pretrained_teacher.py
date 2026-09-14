"""Supervised online projectors, not stale EMA keys, supply frozen targets."""
from unittest.mock import patch
import copy
import pytest
import torch
from test_future_state import batch,call
from gcnet_missing_m3.model import MissingM3GraphModel


def make(**overrides):
    torch.manual_seed(34)
    kw=dict(dropout=0,latent_dim=8,projector_dropout=0,predictor_dropout=0,
            backbone_type='osram',osram_num_heads=2,osram_key_dim=3,osram_value_dim=3,
            osram_output_dim=12,osram_bidirectional=False,osram_write_step=.6,fusion_type='mean')
    kw.update(overrides)
    return MissingM3GraphModel('LSTM',3,4,5,4,4,2,2,2,1,**kw)


def source_checkpoint(tmp_path):
    m=make()
    with torch.no_grad():
        for p in m.observed_set.projectors.parameters():p.add_(.2)
        for p in m.teacher.parameters():p.fill_(9.)
    config=dict(training_objective='emotion-only',train_rate_mode='fixed',fixed_missing_rate=0.,
                checkpoint_selection='validation',evaluate_test=False,backbone_type='osram',
                osram_bidirectional=False,osram_write_step=.6,osram_readout_fusion='flat',
                fusion_type='mean',latent_dim=8,seed=66,dataset='CMUMOSI',fold=1)
    path=tmp_path/'teacher.pt'
    torch.save(dict(model=m.state_dict(),config=config,epoch=3,selection_split='validation',
                    validation_mean_weighted_f1=.7),path)
    return path,m


def test_load_online_prefix_preserves_student_rng_and_default_forward(tmp_path):
    path,source=source_checkpoint(tmp_path)
    old=make();rng=torch.get_rng_state()
    m=make(teacher_mode='pretrained-frozen',teacher_checkpoint=str(path))
    assert torch.equal(rng,torch.get_rng_state())
    assert set(old.state_dict())==set(m.state_dict())
    for k,v in old.state_dict().items():
        if not k.startswith('teacher.'):assert torch.equal(v,m.state_dict()[k]),k
    for k,v in m.teacher.state_dict().items():
        assert torch.equal(v,source.observed_set.projectors.state_dict()[k])
        assert not torch.equal(v,source.teacher.state_dict()[k])
    x,a,u,q,mask=batch();old.eval();m.eval()
    assert torch.equal(call(old,x,a,u,q,mask)[0],call(m,x,a,u,q,mask)[0])
    assert m.teacher_provenance['source_prefix']=='observed_set.projectors.'


def test_frozen_teacher_optimizer_ema_and_target_gradient(tmp_path):
    from gcnet_missing_m3.loss import missing_m3_loss
    path,_=source_checkpoint(tmp_path)
    m=make(teacher_mode='pretrained-frozen',teacher_checkpoint=str(path)).train()
    before=m.teacher_integrity();x,a,u,q,mask=batch();x.requires_grad_()
    optimizer=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=.001)
    _,_,_,pred=m([x*mask],a,q,u,[3,2],predict_missing=True)
    targets=m.encode_teacher_targets([x])
    loss=missing_m3_loss(pred,targets).total;loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in m.teacher.parameters())
    assert not m.teacher.training
    assert sum(p.grad.abs().sum() for p in m.missing_predictor.parameters() if p.grad is not None)>0
    optimizer.step()
    with patch.object(m.teacher,'update_from',side_effect=AssertionError('EMA')):
        m.update_teacher(.5)
    assert m.ema_step==0 and before==m.teacher_integrity()
    assert torch.count_nonzero(x.grad[mask==0])==0


def test_default_ema_still_updates_and_old_keys_strict_restore():
    m=make();old=copy.deepcopy(m.teacher.state_dict())
    with torch.no_grad():
        for p in m.observed_set.projectors.parameters():p.add_(.2)
    m.update_teacher(.5)
    for k,v in m.teacher.state_dict().items():
        torch.testing.assert_close(v,.5*old[k]+.5*m.observed_set.projectors.state_dict()[k])
    make().load_state_dict(m.state_dict(),strict=True)


@pytest.mark.parametrize('corrupt',['missing','extra','shape','nonfinite','test_selected','not_complete'])
def test_reject_invalid_source(tmp_path,corrupt):
    path,_=source_checkpoint(tmp_path);d=torch.load(path,weights_only=False)
    key='observed_set.projectors.audio.fc1.weight'
    if corrupt=='missing':del d['model'][key]
    elif corrupt=='extra':d['model']['observed_set.projectors.fake.weight']=torch.ones(1)
    elif corrupt=='shape':d['model'][key]=torch.ones(2,2)
    elif corrupt=='nonfinite':d['model'][key].fill_(float('nan'))
    elif corrupt=='test_selected':d['selection_split']='test'
    else:d['config']['fixed_missing_rate']=.5
    torch.save(d,path)
    with pytest.raises((ValueError,RuntimeError)):
        make(teacher_mode='pretrained-frozen',teacher_checkpoint=str(path))


def test_export_load_hash_and_no_stale_teacher(tmp_path):
    from gcnet_missing_m3.pretrained_teacher import export_teacher_projectors
    path,_=source_checkpoint(tmp_path);export=tmp_path/'projectors.pt'
    export_teacher_projectors(path,export)
    contents=torch.load(export,weights_only=False)
    assert all(k.startswith('observed_set.projectors.') for k in contents['model'])
    a=make(teacher_mode='pretrained-frozen',teacher_checkpoint=str(path))
    b=make(teacher_mode='pretrained-frozen',teacher_checkpoint=str(export))
    assert a.teacher_integrity()==b.teacher_integrity()


@pytest.mark.parametrize('kwargs',[dict(teacher_mode='bad'),dict(teacher_mode='pretrained-frozen'),
    dict(teacher_checkpoint='x'),dict(teacher_mode='pretrained-frozen',teacher_checkpoint='x',future_state_jepa=True),
    dict(teacher_mode='pretrained-frozen',teacher_checkpoint='x',classification_completion=True)])
def test_mode_conflicts(kwargs):
    with pytest.raises(ValueError):make(**kwargs)
