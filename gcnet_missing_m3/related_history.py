"""Strictly causal B1 retrieval / B2 emotion-consistency residual for original Flat.

This optional branch reads previous *Local* vectors, not another CHM trajectory.
It preserves all Base / Gap forward coordinates in its final concatenation.
Emotion labels are accepted only by the separate training-only auxiliary method.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RelatedHistoryResidual(nn.Module):
    """B1/B2 pre-LayerNorm residual with a shared, eight-utterance past window.

    All tensors are time-major. ``base`` and ``gap`` must already contain only
    the effective forward half. ``speakers`` is [T,B] IDs or [T,B,P] one-hot.
    The caller isolates parameter initialization with ``torch.random.fork_rng``.

    Retrieval biases encode same/different speaker, exact lag 1..window, and
    historical availability pattern (A/T/V bits 1/2/4). They are zero-initialized
    scalar lookup tables, not emotion labels or manually designed decay gates.
    """

    def __init__(
        self,
        kind: str = "B1",
        local_dim: int = 256,
        context_dim: int = 512,
        output_dim: int = 1600,
        window: int = 8,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        if kind not in {"B1", "B2"}:
            raise ValueError("kind must be B1 or B2")
        if min(local_dim, context_dim, output_dim, window, attention_dim) < 1:
            raise ValueError("all dimensions and window must be positive")
        self.kind = kind
        self.local_dim = local_dim
        self.context_dim = context_dim
        self.window = window
        self.attention_dim = attention_dim
        self.query = nn.Linear(local_dim + context_dim + 3, attention_dim)
        self.key = nn.Linear(local_dim, attention_dim)
        self.value = nn.Linear(local_dim, attention_dim)
        self.speaker_bias = nn.Embedding(2, 1)
        self.distance_bias = nn.Embedding(window, 1)
        self.availability_bias = nn.Embedding(8, 1)
        for layer in (self.speaker_bias, self.distance_bias, self.availability_bias):
            nn.init.zeros_(layer.weight)
        input_dim = local_dim + 4 * context_dim + attention_dim + 3
        self.relation: Optional[nn.Sequential] = None
        if kind == "B2":
            # Local pair + same-speaker bit + normalized distance + 2*availability.
            self.relation = nn.Sequential(
                nn.Linear(2 * local_dim + 8, 128), nn.GELU(), nn.Linear(128, 1))
            input_dim += attention_dim + 1
        self.fusion = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU())
        self.output = nn.Linear(128, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_attention: Optional[Tensor] = None
        self.last_diagnostics: Dict[str, Tensor] = {}
        self._cache: Optional[Dict[str, Tensor]] = None

    def clear_cache(self) -> None:
        """Release detached training-pair inputs after the loss or evaluation."""
        self._cache = None

    @staticmethod
    def _past(tensor: Tensor, indices: Tensor) -> Tensor:
        # [T,W,B,...] -> [T,B,W,...]; one batch entry is one conversation.
        selected = tensor[indices]
        return selected.transpose(1, 2)

    def forward(
        self,
        local: Tensor,
        base: Tensor,
        gap: Tensor,
        availability: Tensor,
        umask: Tensor,
        speakers: Tensor,
    ) -> Tensor:
        length, batch = local.shape[:2]
        if local.shape != (length, batch, self.local_dim) or length < 1:
            raise ValueError("local must be nonempty [T,B,local_dim]")
        if base.shape != (length, batch, self.context_dim):
            raise ValueError("base must contain exactly the forward context dimensions")
        if gap.shape != (length, batch, 3, self.context_dim):
            raise ValueError("gap must retain three fixed forward context slots")
        if availability.shape != (length, batch, 3) or umask.shape != (length, batch):
            raise ValueError("availability / umask must be time-major [T,B,3] / [T,B]")
        if speakers.ndim == 3:
            speakers = speakers.argmax(-1)
        if speakers.shape != (length, batch):
            raise ValueError("speakers must be [T,B] IDs or [T,B,P] one-hot")
        valid = umask.bool()
        safe_local = torch.where(valid[..., None], local, torch.zeros_like(local))
        safe_base = torch.where(valid[..., None], base, torch.zeros_like(base))
        safe_availability = torch.where(
            valid[..., None], availability, torch.zeros_like(availability)).to(local.dtype)
        active_gap = valid[..., None] & ~safe_availability.bool()
        safe_gap = torch.where(active_gap[..., None], gap, torch.zeros_like(gap))
        offsets = torch.arange(1, self.window + 1, device=local.device)
        raw_indices = torch.arange(length, device=local.device)[:, None] - offsets
        indices = raw_indices.clamp_min(0)
        pair_mask = (raw_indices >= 0)[:, None, :] & self._past(valid, indices)
        pair_mask = pair_mask & valid[..., None]
        history_local = self._past(safe_local, indices)
        history_local = torch.where(pair_mask[..., None], history_local,
                                    torch.zeros_like(history_local))
        history_availability = self._past(safe_availability, indices)
        history_availability = torch.where(pair_mask[..., None], history_availability,
                                           torch.zeros_like(history_availability))
        historical_speaker = self._past(speakers, indices)
        same_speaker = (historical_speaker == speakers[..., None]).to(local.dtype)
        normalized_distance = (offsets.to(local.dtype) / self.window).view(1, 1, -1)
        normalized_distance = normalized_distance.expand(length, batch, -1)
        current_local = safe_local[..., None, :].expand_as(history_local)
        current_availability = safe_availability[..., None, :].expand_as(history_availability)
        relation_input = torch.cat((
            current_local, history_local, same_speaker[..., None],
            normalized_distance[..., None], current_availability, history_availability), -1)
        relation_input = torch.where(pair_mask[..., None], relation_input,
                                     torch.zeros_like(relation_input))

        query = self.query(torch.cat((safe_local, safe_base, safe_availability), -1))
        keys = self.key(history_local)
        values = self.value(history_local)
        scores = (query[..., None, :] * keys).sum(-1) * self.attention_dim ** -0.5
        pattern = (history_availability.long() * availability.new_tensor([1, 2, 4]).long()).sum(-1)
        scores = scores + self.speaker_bias(same_speaker.long()).squeeze(-1)
        scores = scores + self.distance_bias(offsets - 1).squeeze(-1).view(1, 1, -1)
        scores = scores + self.availability_bias(pattern).squeeze(-1)
        scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, -1)
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))
        evidence = (weights[..., None] * values).sum(-2)
        fusion_inputs = [safe_local, safe_base, safe_gap.flatten(-2), evidence,
                         safe_availability]
        self.clear_cache()
        if self.kind == "B2":
            assert self.relation is not None
            consistency = torch.sigmoid(self.relation(relation_input).squeeze(-1))
            consistency = torch.where(pair_mask, consistency, torch.zeros_like(consistency))
            same_evidence = (weights[..., None] * consistency[..., None] * values).sum(-2)
            change_evidence = (weights[..., None] * (1 - consistency[..., None]) * values).sum(-2)
            difference = same_evidence - change_evidence
            rho = (weights * consistency).sum(-1, keepdim=True)
            fusion_inputs += [difference, rho]
            # Auxiliary BCE recomputes S from detached features: no encoder,
            # CHM, retrieval, Flat, or classifier receives its direct gradient.
            self._cache = {"relation_input": relation_input.detach(),
                           "pair_mask": pair_mask.detach(), "indices": indices.detach()}
        residual = self.output(self.fusion(torch.cat(fusion_inputs, -1)))
        active = valid & pair_mask.any(-1)
        residual = torch.where(active[..., None], residual, torch.zeros_like(residual))
        with torch.no_grad():
            denominator = valid.sum().clamp_min(1)
            self.last_attention = weights.detach()
            self.last_diagnostics = {
                "history_residual_norm": residual.norm(dim=-1).sum() / denominator,
                "history_candidate_count_mean": pair_mask.sum().to(local.dtype) / denominator,
                "history_active_fraction": active.sum().to(local.dtype) / denominator,
            }
            if self.kind == "B2":
                self.last_diagnostics["history_consistency_mean"] = (
                    consistency.sum() / pair_mask.sum().clamp_min(1))
                self.last_diagnostics["history_same_weight_mean"] = rho.sum() / denominator
        return residual

    def auxiliary_relation_loss(self, labels: Tensor, dataset: str = "MELD") -> Tensor:
        """Training-only BCE, averaged within each conversation then across them.

        B2 forward never receives labels. Main-task gradients use the original
        differentiable S computation; only this fresh S evaluation is detached
        from its feature inputs. The returned scalar is unscaled (caller *.05).
        """
        if self.kind != "B2" or self.relation is None:
            raise RuntimeError("auxiliary relation supervision is only defined for B2")
        if self._cache is None:
            raise RuntimeError("run forward before auxiliary_relation_loss")
        cache = self._cache
        pair_mask = cache["pair_mask"]
        if labels.shape != pair_mask.shape[:2]:
            raise ValueError("labels must have time-major [T,B] shape")
        past_labels = self._past(labels.detach(), cache["indices"])
        current_labels = labels.detach()[..., None].expand_as(past_labels)
        dataset = dataset.upper()
        if dataset in {"MELD", "IEMOCAP", "IEMOCAP4", "IEMOCAP6", "IEMOCAP-4", "IEMOCAP-6"}:
            targets = current_labels == past_labels
        elif dataset in {"MOSI", "MOSEI", "CMU-MOSI", "CMU-MOSEI"}:
            pair_mask = pair_mask & (current_labels.abs() > 0.5) & (past_labels.abs() > 0.5)
            targets = (current_labels > 0) == (past_labels > 0)
        else:
            raise ValueError("unsupported relation-label dataset: " + dataset)
        logits = self.relation(cache["relation_input"]).squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
        losses = torch.where(pair_mask, losses, torch.zeros_like(losses))
        counts = pair_mask.sum(dim=(0, 2))
        conversation_loss = losses.sum(dim=(0, 2)) / counts.clamp_min(1)
        active = counts > 0
        return conversation_loss.sum() / active.sum().clamp_min(1)
