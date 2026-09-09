import copy
import importlib
import inspect

import pytest
import torch

from gcnet_missing_m3.model import MissingM3GraphModel, MODALITIES
from gcnet_missing_m3.loss import missing_m3_loss


def components():
    spec = importlib.util.find_spec("gcnet_missing_m3.b2")
    assert spec is not None, "B2 components not implemented"
    return importlib.import_module("gcnet_missing_m3.b2")


def inputs():
    a = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],
                      [0,1,1],[1,1,1],[0,0,0]], dtype=torch.float32)[:,None]
    u = torch.tensor([[1,1,1,1,1,1,1,0]], dtype=torch.float32)
    z = {m: torch.randn(8,1,8) * a[...,i:i+1] for i,m in enumerate(MODALITIES)}
    return z, a, u


def model_kwargs():
    return dict(base_model="LSTM",adim=3,tdim=4,vdim=5,D_e=4,
        graph_hidden_size=2,n_speakers=1,window_past=2,window_future=2,
        n_classes=1,latent_dim=8,num_experts=2,top_k=1,dropout=0.,
        projector_dropout=0.,predictor_dropout=0.,time_attn=False,no_cuda=True,
        backbone_type="osram",osram_output_dim=10,osram_num_heads=2,
        osram_key_dim=3,osram_value_dim=4,osram_bidirectional=False,osram_write_step=.6)


def test_source_only_six_directions_masks_and_order():
    b2 = components()
    predictor = b2.SourceOnlyM3Predictor(8,2,1,0.).eval()
    assert set(inspect.signature(predictor.forward).parameters) == {"latents","availability","umask"}
    z,a,u = inputs()
    pred = predictor(z,a,u)
    expected = (~a.bool()) & u.T.bool().unsqueeze(-1)
    assert torch.equal(pred.target_mask, expected)
    assert torch.count_nonzero(pred.reg_predictions[~expected]) == 0
    reverse = predictor(dict(reversed(list(z.items()))),a,u)
    assert torch.equal(pred.reg_predictions, reverse.reg_predictions)
    for target in range(3):
        for row in range(7):
            if a[row,0,target]:
                continue
            outputs = [predictor.mmoe(predictor.input_norm(z[name][row]), src, target)[0]
                       for src,name in enumerate(MODALITIES) if a[row,0,src]]
            torch.testing.assert_close(pred.reg_predictions[row,:,target], torch.stack(outputs).mean(0))
    changed = {m: v.clone() for m,v in z.items()}
    for i,m in enumerate(MODALITIES):
        changed[m][~a[...,i].bool()] = 1000 * torch.randn_like(changed[m][~a[...,i].bool()])
    assert torch.equal(predictor(changed,a,u).reg_predictions, pred.reg_predictions)


def test_completed_fusion_zero_init_complete_padding_and_status():
    b2 = components()
    z,a,u = inputs()
    node = torch.randn(8,1,8) * u.T.unsqueeze(-1)
    fusion = b2.CompletedReadFusion(8,0.).eval()
    prediction = torch.randn(8,1,3,8)
    assert torch.equal(fusion(node,z,prediction,a,u), node)
    torch.nn.init.normal_(fusion.residual[-1].weight, std=.1)
    value = fusion(node,z,prediction,a,u)
    assert torch.equal(value[6:],node[6:])
    assert not torch.equal(value[:6],node[:6])
    filled = fusion.fill_slots(z,prediction,a,u)
    for i,m in enumerate(MODALITIES):
        assert torch.equal(filled[...,i,:][a[...,i].bool()],z[m][a[...,i].bool()])
    assert torch.count_nonzero(filled[7]) == 0


def test_default_compatibility_and_b2_exclusion():
    torch.manual_seed(44)
    baseline = MissingM3GraphModel(**model_kwargs()).eval()
    state = torch.get_rng_state().clone()
    torch.manual_seed(44)
    explicit = MissingM3GraphModel(**model_kwargs(),completion_path="none").eval()
    assert torch.equal(state,torch.get_rng_state())
    explicit.load_state_dict(baseline.state_dict(),strict=True)
    assert all(torch.equal(v,explicit.state_dict()[k]) for k,v in baseline.state_dict().items())
    with pytest.raises(ValueError):
        MissingM3GraphModel(**model_kwargs(),completion_path="pre_osram_b2",classification_completion=True)
    for change in ({"osram_bidirectional":True},{"osram_write_step":1.0}):
        with pytest.raises(ValueError):
            MissingM3GraphModel(**(model_kwargs() | change),completion_path="pre_osram_b2")


def test_b2_one_osram_no_teacher_and_gradients():
    components()
    baseline = MissingM3GraphModel(**model_kwargs()).eval()
    model = MissingM3GraphModel(**model_kwargs(),completion_path="pre_osram_b2").eval()
    missing,unexpected = model.load_state_dict(baseline.state_dict(),strict=False)
    assert not unexpected and all(k.startswith(("source_only_predictor.","completed_read_fusion.")) for k in missing)
    z,a,u = inputs()
    x = torch.randn(8,1,12)
    q = torch.zeros(1,8,dtype=torch.long)
    args=([x],a,q,u,[7])
    calls=[]
    hook=model.osram.register_forward_hook(lambda *args: calls.append(1))
    model.teacher.register_forward_pre_hook(lambda *args: pytest.fail("teacher called at inference"))
    out=model(*args,predict_missing=True)
    assert len(calls)==1
    assert torch.equal(out[0],baseline(*args)[0])
    loss=out[0].square().sum()
    loss.backward()
    assert model.completed_read_fusion.residual[-1].weight.grad.norm()>0
    # Expected zero-init exception: first emotion step cannot reach predictor.
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.source_only_predictor.parameters())
    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(model.completed_read_fusion.residual[-1].weight,std=.1)
    out=model(*args,predict_missing=True)
    out[0].square().sum().backward()
    assert sum(float(p.grad.norm()) for p in model.source_only_predictor.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in model.teacher.parameters())
    model.zero_grad(set_to_none=True)
    pred=model(*args,predict_missing=True)[3]
    targets={m:torch.randn(8,1,8) for m in MODALITIES}
    missing_m3_loss(pred,targets).total.backward()
    assert model.source_only_predictor.mmoe.reg_heads[1].weight.grad.norm()>0
    hook.remove()


@pytest.mark.parametrize("replacement",["zero","random","shuffle"])
def test_b2_prediction_override_memory_exact_each_step(replacement):
    components()
    model=MissingM3GraphModel(**model_kwargs(),completion_path="pre_osram_b2").eval()
    torch.nn.init.normal_(model.completed_read_fusion.residual[-1].weight,std=.2)
    torch.nn.init.normal_(model.osram.emotion_adapter[-1].weight,std=.1)
    _,a,u=inputs()
    args=([torch.randn(8,1,12)],a,torch.zeros(1,8,dtype=torch.long),u,[7])
    original_write=model.osram.block_write
    trace=[]
    def traced(*args,**kwargs):
        result=original_write(*args,**kwargs)
        trace.append(result.detach().clone())
        return result
    model.osram.block_write=traced
    first=model(*args,predict_missing=True)
    history=trace[:]
    trace.clear()
    pred=first[3].reg_predictions.detach()
    changed=torch.zeros_like(pred) if replacement=="zero" else torch.randn_like(pred)
    if replacement=="shuffle":
        # Repeat each pattern to permit a nontrivial within-pattern shuffle.
        args=( [args[0][0].repeat(1,2,1)+torch.randn(8,2,12)],a.repeat(1,2,1),
                torch.zeros(2,8,dtype=torch.long),u.repeat(2,1),[7,7])
        trace.clear()
        first=model(*args,predict_missing=True)
        history=trace[:]
        trace.clear()
        changed=first[3].reg_predictions.detach().flip(1)
    second=model(*args,completion_predictions_override=changed)
    assert len(history)==len(trace)>0
    assert all(torch.equal(a,b) for a,b in zip(history,trace))
    assert not torch.equal(first[0],second[0])
    assert second[3] is None
