"""Classification-input interventions; preserve learned Source-Role calibration."""
import torch


@torch.no_grad()
def replay_roles(osram, head, local, emotion_input, availability, umask, full_prediction):
    if osram.training or head.training:
        raise ValueError('Eval-only diagnostic')
    t, b, ld = local.shape
    fd = osram.num_heads * osram.value_dim
    if emotion_input.shape != (t, b, ld + 8 * fd) or availability.shape != (t, b, 3) or umask.shape != (b, t):
        raise ValueError('Unexpected Flat shapes')
    torch.testing.assert_close(local, emotion_input[..., :ld], atol=0, rtol=0)
    contexts = emotion_input[..., ld:].reshape(t, b, 4, 2 * fd)
    valid = umask.T.bool()
    active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
    if contexts[~active].ne(0).any():
        raise ValueError('Inactive evidence leaked')
    if contexts[..., fd:].ne(0).any() or not torch.isfinite(emotion_input).all():
        raise ValueError('Nonfinite or nonzero backward evidence')
    skip = osram.local_skip(local)

    def classify(x):
        hidden = osram.emotion_norm(skip + osram.emotion_adapter(x))
        return head(hidden * valid[..., None].to(hidden.dtype))

    torch.testing.assert_close(classify(emotion_input), full_prediction, atol=0, rtol=0)
    result = {'Full': full_prediction}
    for name, keep in [('L', (0, 0, 0, 0)), ('LB', (1, 0, 0, 0)), ('LG', (0, 1, 1, 1))]:
        value = emotion_input.clone()
        selected = torch.tensor(keep, dtype=torch.bool, device=value.device)[None, None, :, None]
        value[..., ld:] = torch.where(selected, contexts, torch.zeros_like(contexts)).flatten(-2)
        result[name] = classify(value)
    return result
