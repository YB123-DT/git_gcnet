"""Value-slot source tracking for the *existing joint* OSRAM block update.

Sources denote additive Value-column contributions, not independent unimodal
memories or semantic causal attributions. No write/query formula lives here.
"""

from __future__ import annotations

import torch
from torch import nn


def strict_past_trajectory(availability: torch.Tensor, umask: torch.Tensor) -> torch.Tensor:
    """Return count/coverage/age/ever_seen, measured in valid past utterances."""
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError('availability must be [T,B,3]')
    length, batch = availability.shape[:2]
    if umask.shape != (batch, length):
        raise ValueError('umask must be [B,T]')
    valid = umask.transpose(0, 1).bool()
    if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
        raise ValueError('valid availability must be binary')
    observed = torch.where(valid[..., None], availability, torch.zeros_like(availability)).float()
    steps = availability.new_zeros(batch, dtype=observed.dtype)
    counts = observed.new_zeros(batch, 3)
    last = observed.new_zeros(batch, 3)
    output = observed.new_zeros(length, batch, 3, 4)
    for t in range(length):
        ever = counts > 0
        age = torch.where(ever, steps[:, None] - last, torch.zeros_like(counts))
        current = torch.stack((counts, counts / steps[:, None].clamp_min(1), age, ever.float()), -1)
        output[t] = torch.where(valid[t, :, None, None], current, torch.zeros_like(current))
        last = torch.where(observed[t].bool(), steps[:, None], last)
        counts = counts + observed[t]
        steps = steps + valid[t].to(steps.dtype)
    return output.to(dtype=availability.dtype if availability.is_floating_point() else torch.float32)


class JointSourceTracker:
    """Reuse Kbar/Vbar/P from one real joint solve to track additive sources."""

    def __init__(self, availability, valid, memory_template, *, audit=False):
        if memory_template.ndim != 4 or valid.shape != availability.shape[:2]:
            raise ValueError('invalid memory template or valid mask')
        length, batch = availability.shape[:2]
        if memory_template.shape[0] != batch:
            raise ValueError('memory batch differs from availability')
        self.valid = valid.bool()
        self.states = memory_template.new_zeros(batch, 3, *memory_template.shape[1:])
        self.source_gap = memory_template.new_zeros(
            length, batch, 3, 3, memory_template.shape[1] * memory_template.shape[2])
        self.trajectory = strict_past_trajectory(availability, valid.transpose(0, 1))
        self._errors = {'memory': [], 'gap': []}
        self.audit = bool(audit)
        if self.audit:
            # Audit-only summaries: never retain a time sequence of full
            # memory matrices, and never feed these values back into CHM.
            self.source_base = memory_template.new_zeros(
                length, batch, 3, memory_template.shape[1] * memory_template.shape[2])
            self.state_source_norms = memory_template.new_zeros(length, batch, 3)
            self.state_joint_norm = memory_template.new_zeros(length, batch)
            self._errors.update(base=[], prewrite_memory=[])

    def decay(self, alpha, active):
        decayed = self.states * alpha.view(1, 1, -1, 1, 1)
        self.states = torch.where(active[:, None, None, None, None], decayed, self.states)

    def _record_error(self, kind, reconstructed, original, active):
        # Keep detached scalar tensors on device; synchronize only at finish().
        error = torch.where(active.reshape(-1, *([1] * (original.ndim - 1))),
                            reconstructed - original, torch.zeros_like(original)).detach()
        reference = torch.where(active.reshape(-1, *([1] * (original.ndim - 1))),
                                original, torch.zeros_like(original)).detach()
        self._errors[kind].append(torch.stack((error.abs().amax(),
                                               error.norm() / reference.norm().clamp_min(1e-12))))

    def read(self, time_index, modality_index, residual_query, original_gap, active_missing):
        reads = (self.states @ residual_query[:, None, :, :, None]).squeeze(-1)
        reads = reads.reshape(reads.shape[0], 3, -1)
        reads = torch.where(active_missing[:, None, None], reads, torch.zeros_like(reads))
        self.source_gap[time_index, :, modality_index] = reads
        self._record_error('gap', reads.sum(1), original_gap, active_missing)
        return reads

    def read_base(self, time_index, query, original_base, active, memory):
        """Record Base and state norms after decay, strictly before writing.

        ``query`` is the existing Base task query; ``memory`` is the original
        shared state, not a reconstructed replacement. Sources are additive
        joint-Value-slot contributions, not isolated modality memories.
        """
        if not self.audit:
            raise ValueError('Base source capture requires audit=True')
        with torch.no_grad():
            reads = (self.states.detach() @ query.detach()[:, None, :, :, None]).squeeze(-1)
            reads = reads.reshape(reads.shape[0], 3, -1)
            reads = torch.where(active[:, None, None], reads, torch.zeros_like(reads))
            self.source_base[time_index] = reads
            norms = self.states.detach().flatten(2).norm(dim=-1)
            self.state_source_norms[time_index] = torch.where(
                active[:, None], norms, torch.zeros_like(norms))
            joint_norm = memory.detach().flatten(1).norm(dim=-1)
            self.state_joint_norm[time_index] = torch.where(
                active, joint_norm, torch.zeros_like(joint_norm))
            self._record_error('base', reads.sum(1), original_base, active)
            self._record_error('prewrite_memory', self.states.detach().sum(1), memory, active)
        return reads

    def update(self, time_index, k_bar, v_bar, solved, write_step, active, memory):
        # v_s p_s is the source's injection; ALL sources share the same joint
        # erasure operator Kbar P, including when that source is not observed.
        injection = v_bar.permute(0, 3, 1, 2).unsqueeze(-1) * solved.permute(0, 2, 1, 3).unsqueeze(-2)
        erasure = (self.states @ k_bar[:, None]) @ solved[:, None]
        correction = injection - erasure
        candidate = self.states + (correction if write_step == 1.0 else write_step * correction)
        self.states = torch.where(active[:, None, None, None, None], candidate, self.states)
        self._record_error('memory', self.states.sum(1), memory, active)

    def finish(self):
        diagnostics = {}
        for kind, values in self._errors.items():
            maximum = torch.stack(values).amax(0).cpu().tolist() if values else [0., 0.]
            diagnostics[kind + '_sum_max_abs_error'] = maximum[0]
            diagnostics[kind + '_sum_max_relative_error'] = maximum[1]
        trace = dict(source_gap=self.source_gap, trajectory=self.trajectory,
                     final_sources=self.states, diagnostics=diagnostics)
        if self.audit:
            trace.update(source_base=self.source_base,
                         state_source_norms=self.state_source_norms,
                         state_joint_norm=self.state_joint_norm)
        return trace


class TASRAdapter(nn.Module):
    """Bounded target/source correction of existing Gap reads only.

    ``neutral`` retains exactly the real adapter's parameterization and
    trajectory inputs but presents G_original/3 for every source slot.
    """

    def __init__(self, latent_dim, forward_dim, source_mode='real'):
        super().__init__()
        if source_mode not in ('real', 'neutral'):
            raise ValueError('source_mode must be real or neutral')
        self.latent_dim = int(latent_dim)
        self.forward_dim = int(forward_dim)
        self.source_mode = source_mode
        self.local_projection = nn.Linear(self.latent_dim, 64)
        self.memory_projection = nn.Linear(self.forward_dim, 64)
        self.target_embedding = nn.Embedding(3, 8)
        self.source_embedding = nn.Embedding(3, 8)
        self.score = nn.Sequential(nn.LayerNorm(151), nn.Linear(151, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.last_diagnostics = {}
        self.last_gamma = None

    def forward(self, local, original_gap, source_gap, availability, umask, trajectory):
        length, batch = local.shape[:2]
        fd = self.forward_dim
        if (local.shape != (length, batch, self.latent_dim)
                or original_gap.shape != (length, batch, 3, 2 * fd)
                or source_gap.shape != (length, batch, 3, 3, fd)
                or availability.shape != (length, batch, 3)
                or umask.shape != (batch, length)
                or trajectory.shape != (length, batch, 3, 4)):
            raise ValueError('TASR input shapes do not match Local/forward dimensions')
        valid = umask.transpose(0, 1).bool()
        has_history = (trajectory[..., 0].sum(-1) > 0) & valid
        active_target = valid[..., None] & has_history[..., None] & ~availability.bool()
        active_pair = active_target[..., None].expand(-1, -1, -1, 3)
        if self.source_mode == 'real':
            active_pair = active_pair & trajectory[..., 3].bool().unsqueeze(-2)
            evidence = source_gap
        else:
            evidence = original_gap[..., :fd].unsqueeze(-2).expand(-1, -1, -1, 3, -1) / 3
        evidence = torch.where(active_pair[..., None], evidence, torch.zeros_like(evidence))
        safe_local = torch.where(has_history[..., None], local, torch.zeros_like(local))
        q = self.local_projection(safe_local)[:, :, None, None].expand(-1, -1, 3, 3, -1)
        k = self.memory_projection(evidence)
        target = self.target_embedding.weight[None, None, :, None].expand(length, batch, -1, 3, -1)
        source = self.source_embedding.weight[None, None, None].expand(length, batch, 3, -1, -1)
        meta = trajectory.to(local.dtype)[:, :, None].expand(-1, -1, 3, -1, -1)
        av = availability.to(local.dtype)[:, :, None, None].expand(-1, -1, 3, 3, -1)
        features = torch.cat((q, k, av, meta, target, source), -1)
        features = torch.where(active_pair[..., None], features, torch.zeros_like(features))
        gamma = .25 * torch.tanh(self.score(features).squeeze(-1))
        gamma = torch.where(active_pair, gamma, torch.zeros_like(gamma))
        forward_delta = (gamma[..., None] * evidence).sum(-2)
        forward_delta = torch.where(active_target[..., None], forward_delta, torch.zeros_like(forward_delta))
        delta = torch.cat((forward_delta, torch.zeros_like(forward_delta)), -1)
        self.last_gamma = gamma.detach()
        with torch.no_grad():
            selected_gamma = gamma[active_pair]
            safe_gap = torch.where(active_target[..., None], original_gap[..., :fd], torch.zeros_like(forward_delta))
            safe_trajectory = torch.where(valid[..., None, None], trajectory, torch.zeros_like(trajectory))
            values = torch.stack((
                selected_gamma.mean() if selected_gamma.numel() else gamma.new_zeros(()),
                selected_gamma.amin() if selected_gamma.numel() else gamma.new_zeros(()),
                selected_gamma.amax() if selected_gamma.numel() else gamma.new_zeros(()),
                evidence.norm(dim=-1).mean(), forward_delta.norm(),
                forward_delta.norm() / safe_gap.norm().clamp_min(1e-12),
                safe_trajectory[..., 1].sum() / (valid.sum() * 3).clamp_min(1),
                safe_trajectory[..., 3].sum() / (valid.sum() * 3).clamp_min(1),
                (safe_gap + forward_delta).norm(),
            )).detach().cpu().tolist()
            names = ('gamma_mean', 'gamma_min', 'gamma_max', 'source_read_norm',
                     'delta_gap_norm', 'delta_original_gap_norm_ratio',
                     'trajectory_coverage_mean', 'trajectory_ever_seen_fraction', 'corrected_gap_norm')
            self.last_diagnostics = dict(zip(names, values))
            # Preserve the target/source structure rather than interpreting a
            # single overall norm as source-specific utility or reliability.
            counts = active_pair.sum((0, 1)).clamp_min(1)
            slot_read = evidence.norm(dim=-1).sum((0, 1)) / counts
            slot_gamma = gamma.sum((0, 1)) / counts
            self.last_diagnostics['target_source_read_norm'] = slot_read.cpu().tolist()
            self.last_diagnostics['target_source_gamma_mean'] = slot_gamma.cpu().tolist()
            quantiles = (torch.quantile(selected_gamma.float(), selected_gamma.new_tensor([.1, .5, .9]).float())
                         if selected_gamma.numel() else gamma.new_zeros(3))
            self.last_diagnostics['gamma_q10_q50_q90'] = quantiles.cpu().tolist()
        return delta
