"""Read-only, per-head Local-to-role source attention with exact return parity."""
import numpy as np
import torch


def matched_donors(role, availability_code, seen_code, ids):
    donors = np.full(len(role), -1, dtype=np.int64)
    groups = {}
    for i, key in enumerate(zip(role, availability_code, seen_code)):
        groups.setdefault(tuple(key), []).append(i)
    for group in groups.values():
        for position, i in enumerate(group):
            for offset in range(1, len(group)):
                j = group[(position+offset) % len(group)]
                if ids[i] != ids[j]:
                    donors[i] = j
                    break
    return donors


class AttentionCapture:
    def __init__(self, module):
        self.module = module
        self.conversation_ids = None
        self.arrays = None

    def __enter__(self):
        self.original = self.module.output_attention.forward
        self.pre_hook = self.module.register_forward_pre_hook(self.prepare)
        self.post_hook = self.module.register_forward_hook(self.finish)
        self.module.output_attention.forward = self.attend
        return self

    def __exit__(self, *unused):
        self.module.output_attention.forward = self.original
        self.pre_hook.remove(); self.post_hook.remove()

    def prepare(self, module, args):
        local, _, _, trajectory, availability, umask = args
        valid = umask.T.bool()
        seen = trajectory[..., 3].bool() & valid[..., None]
        enabled = valid & ~availability.bool().all(-1) & seen.any(-1)
        roles = torch.cat((enabled[..., None], enabled[..., None] & ~availability.bool()), -1)
        t, b = local.shape[:2]
        self.valid, self.roles = valid, roles
        self.local, self.trajectory, self.availability, self.seen = local, trajectory, availability, seen
        self.weights = local.new_zeros(t, b, 4, module.output_attention.num_heads, 3)
        self.shuffled = torch.zeros_like(self.weights)
        self.shuffle_valid = torch.zeros_like(roles)

    def attend(self, query, key, value, **kwargs):
        opts = dict(kwargs, average_attn_weights=False)
        output, weights = self.original(query, key, value, **opts)
        # This averaging is exactly what PyTorch's default return would do;
        # the actual attention output, memory trajectory and prediction stay.
        dense = self.weights.reshape(-1, weights.shape[1], 3)
        selected = self.roles.reshape(-1)
        dense[selected] = weights.detach().squeeze(-2)
        t, b = self.roles.shape[:2]
        indices = np.argwhere(self.roles.detach().cpu().numpy())
        ids = np.asarray(self.conversation_ids if self.conversation_ids is not None
                         else [str(i) for i in range(b)])
        avail = (self.availability.long() * self.availability.new_tensor([1, 2, 4])).sum(-1)
        seen = (self.seen.long() * self.availability.new_tensor([1, 2, 4])).sum(-1)
        donors = matched_donors(indices[:, 2],
            avail.detach().cpu().numpy()[indices[:, 0], indices[:, 1]],
            seen.detach().cpu().numpy()[indices[:, 0], indices[:, 1]], ids[indices[:, 1]])
        eligible = donors >= 0
        if eligible.any():
            row = torch.as_tensor(np.flatnonzero(eligible), device=query.device)
            donor = torch.as_tensor(donors[eligible], device=query.device)
            sub = dict(opts)
            if sub.get('key_padding_mask') is not None:
                sub['key_padding_mask'] = sub['key_padding_mask'][row]
            with torch.no_grad():
                _, changed = self.original(query[donor], key[row], value[row], **sub)
            changed_dense = self.shuffled.reshape_as(dense)
            chosen = selected.nonzero().squeeze(-1)[row]
            changed_dense[chosen] = changed.squeeze(-2)
            self.shuffle_valid.reshape(-1)[chosen] = True
        return output, weights.mean(dim=1)

    def finish(self, module, args, output):
        take = lambda v: v.detach().transpose(0, 1)[self.valid.T].cpu().numpy()
        self.arrays = dict(attention=take(self.weights), role_active=take(self.roles),
            seen=take(self.seen), availability=take(self.availability),
            trajectory=take(self.trajectory), local=take(self.local),
            query_shuffle_attention=take(self.shuffled),
            query_shuffle_eligible=take(self.shuffle_valid))
