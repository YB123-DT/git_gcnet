"""Target-conditioned predictive associative memory for Text targets (PAM-T).

The memory is not a parameter.  It is rebuilt from zero for every forward call
and every conversation, and only real observed Text latents are written.
"""

from typing import Dict, Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("audio", "text", "visual")


class TextConditionedPAMMemory(nn.Module):
    """Causal A/V-condition -> Text-latent associative memory.

    Parameters
    ----------
    latent_dim:
        Width of the Student latent space.  MOSI/MOSEI use 256.
    key_dim:
        Width of the normalized source-condition key/query.
    mask_embed_dim:
        Width of the A/V availability embedding.
    dropout:
        Dropout used inside the source encoder.

    The module owns only the source-condition encoder, the A/V mask embedding,
    and the scalar write strength ``beta``.  The memory itself is a local
    tensor created in :meth:`forward`.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        key_dim: int = 64,
        mask_embed_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0 or int(key_dim) <= 0 or int(mask_embed_dim) <= 0:
            raise ValueError("latent_dim, key_dim and mask_embed_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.key_dim = int(key_dim)
        self.mask_embed_dim = int(mask_embed_dim)
        self.mask_embedding = nn.Embedding(4, self.mask_embed_dim)
        input_dim = 2 * self.latent_dim + self.mask_embed_dim
        self.source_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.key_dim),
        )
        # sigmoid(0) = 0.5, the requested first-version write strength.
        self.beta = nn.Parameter(torch.zeros(()))

    def _validate(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        reference = latents["audio"]
        if reference.ndim != 3 or reference.shape[-1] != self.latent_dim:
            raise ValueError("latents must have shape [L,B,latent_dim]")
        if any(value.shape != reference.shape for value in latents.values()):
            raise ValueError("all modality latents must share the same shape")
        if availability.shape != (*reference.shape[:2], 3):
            raise ValueError("availability must have shape [L,B,3]")
        if umask.shape != reference.shape[:2][::-1]:
            raise ValueError("umask must have shape [B,L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        if not bool(((umask == 0) | (umask == 1)).all()):
            raise ValueError("umask must be binary")
        valid = umask.transpose(0, 1).bool()
        if bool(availability[~valid].any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require at least one observed modality")
        return valid, availability, umask

    def _source_query(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized key/query and the valid A/V source mask."""

        audio_mask = availability[..., 0]
        visual_mask = availability[..., 2]
        has_source = valid & ((audio_mask > 0) | (visual_mask > 0))
        pattern_id = (
            audio_mask.long() * 2 + visual_mask.long()
        )  # A only=1, V only=2, AV=3, no source=0
        pattern = self.mask_embedding(pattern_id)
        source_input = torch.cat(
            (
                latents["audio"] * audio_mask.unsqueeze(-1),
                latents["visual"] * visual_mask.unsqueeze(-1),
                pattern,
            ),
            dim=-1,
        )
        query = F.normalize(self.source_encoder(source_input), dim=-1)
        query = torch.where(
            has_source.unsqueeze(-1), query, torch.zeros_like(query)
        )
        return query, has_source

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid, availability, _ = self._validate(latents, availability, umask)
        length, batch = availability.shape[:2]
        query, has_source = self._source_query(latents, availability, valid)
        text_present = availability[..., 1].bool() & valid
        write_mask = has_source & text_present
        text_missing = valid & (~availability[..., 1].bool()) & has_source

        memory = query.new_zeros(batch, self.latent_dim, self.key_dim)
        z_hat_steps = []
        write_strength = torch.sigmoid(self.beta)

        for t in range(length):
            q_t = query[t]  # [B,K]
            read_t = torch.einsum("bdk,bk->bd", memory, q_t)
            read_t = torch.where(
                has_source[t].unsqueeze(-1), read_t, torch.zeros_like(read_t)
            )
            z_hat_steps.append(read_t)

            mask_t = write_mask[t]  # [B]
            if bool(mask_t.any()):
                # Only a real observed Student Text latent can write.  Keep
                # this value differentiable so future missing-Text reads can
                # train the source-conditioned memory trajectory.
                value_t = latents["text"][t]
                delta = (value_t - read_t).unsqueeze(-1) * q_t.unsqueeze(1)
                updated = memory + write_strength * delta
                memory = torch.where(
                    mask_t.unsqueeze(-1).unsqueeze(-1), updated, memory
                )

        z_hat_text = torch.stack(z_hat_steps, dim=0)
        return {
            "z_hat_text": z_hat_text,
            "pam_mask": has_source,
            "text_missing_mask": text_missing,
            "write_mask": write_mask,
            "query": query,
        }
