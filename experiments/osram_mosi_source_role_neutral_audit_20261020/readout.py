"""Frozen calibration neutralization: no encoder/query/Memory calls."""
import torch


@torch.no_grad()
def replay_neutral(osram,head,local,emotion_input,sources,gamma,trajectory,
                   availability,umask,full_prediction):
    if osram.training or head.training:raise ValueError('Eval-only readout required')
    t,b,ld=local.shape;fd=osram.num_heads*osram.value_dim
    if (emotion_input.shape!=(t,b,ld+8*fd) or sources.shape!=(t,b,4,3,fd)
            or gamma.shape!=(t,b,4,3) or trajectory.shape!=(t,b,3,4)
            or availability.shape!=(t,b,3) or umask.shape!=(b,t)):
        raise ValueError('Unexpected source-role shapes')
    contexts=emotion_input[...,ld:].reshape(t,b,4,2*fd)
    if contexts[...,fd:].ne(0).any():raise ValueError('backward evidence must remain zero')
    valid=umask.T.bool();seen=trajectory[...,3]>0;history=seen.any(-1)
    role_active=torch.cat((history[...,None],~availability.bool()),-1)&valid[...,None]&history[...,None]
    active=role_active[...,None]&seen[:,:,None,:]
    if gamma[~active].ne(0).any():raise ValueError('Inactive gamma leaked')
    safe=torch.where(active[...,None],sources,torch.zeros_like(sources))
    if not torch.isfinite(safe).all() or not torch.isfinite(gamma).all():raise ValueError('Nonfinite active inputs')
    weighted=(gamma[...,None]*safe).sum(-2)
    torch.testing.assert_close(weighted,contexts[...,:fd],atol=0,rtol=0)
    torch.testing.assert_close(local,emotion_input[...,:ld],atol=0,rtol=0)
    skip=osram.local_skip(local)

    def classify(value):
        hidden=osram.emotion_norm(skip+osram.emotion_adapter(value))
        return head(hidden*valid[...,None].to(hidden.dtype))

    torch.testing.assert_close(classify(emotion_input),full_prediction,atol=0,rtol=0)
    def predict(weights):
        x=emotion_input.clone()
        x[...,ld:].reshape(t,b,4,2*fd)[...,:fd]=(weights[...,None]*safe).sum(-2)
        return classify(x)

    neutral=predict(active.to(gamma.dtype));reset=[]
    for r in range(4):
        row=[]
        for s in range(3):
            weights=gamma.clone();weights[...,r,s]=active[...,r,s].to(gamma.dtype)
            row.append(predict(weights))
        reset.append(torch.stack(row,-2))
    return dict(prediction_neutral=neutral,prediction_reset=torch.stack(reset,-3),active=active)
