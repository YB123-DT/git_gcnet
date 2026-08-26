"""Full-state target selection for GCNet's existing reconstruction head."""

import torch
import torch.nn as nn


class FullFusedReconLoss(nn.Module):
    """Reconstruct all modalities when an utterance has any missing modality."""

    def forward(
        self,
        recon_input,
        target_input,
        input_mask,
        umask,
        adim,
        tdim,
        vdim,
    ):
        if len(recon_input) != 1 or len(target_input) != 1 or len(input_mask) != 1:
            raise ValueError("expected one fused reconstruction, target, and mask")

        predicted = recon_input[0]
        expected = target_input[0].detach()
        availability = input_mask[0]
        if predicted.shape != expected.shape:
            raise ValueError("reconstruction and target shapes must match")

        real = umask.transpose(0, 1).bool()
        selected = real & (availability < 1).any(dim=-1)
        if not selected.any():
            return predicted.sum() * 0.0

        dimensions = (adim, tdim, vdim)
        predicted_parts = torch.split(predicted, dimensions, dim=-1)
        expected_parts = torch.split(expected, dimensions, dim=-1)
        modality_losses = [
            (predicted_part[selected] - expected_part[selected]).square().mean()
            for predicted_part, expected_part in zip(
                predicted_parts, expected_parts
            )
        ]
        return torch.stack(modality_losses).mean()
