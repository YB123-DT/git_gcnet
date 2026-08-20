import os
import time
import glob
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv

from .model import ModalityPredictions
from .targets import ModalityMeans


class AllModalReconLoss(nn.Module):
    """Reconstruct every observed modality at real, non-padding utterances."""

    def forward(
        self,
        reconstruction,
        target,
        umask,
        adim: int,
        tdim: int,
        vdim: int,
    ):
        if len(reconstruction) != 1 or len(target) != 1:
            raise ValueError("expected one fused reconstruction and target tensor")
        predicted = reconstruction[0]
        expected = target[0].detach()
        if predicted.shape != expected.shape:
            raise ValueError("reconstruction and target shapes must match")
        valid = umask.transpose(0, 1).to(predicted.dtype).unsqueeze(-1)
        valid_count = valid.sum()
        if valid_count.item() == 0:
            return predicted.sum() * 0.0

        dimensions = (adim, tdim, vdim)
        predicted_parts = torch.split(predicted, dimensions, dim=-1)
        expected_parts = torch.split(expected, dimensions, dim=-1)
        modality_losses = []
        for predicted_part, expected_part, dimension in zip(
            predicted_parts, expected_parts, dimensions
        ):
            squared_error = (predicted_part - expected_part).square() * valid
            modality_losses.append(squared_error.sum() / (valid_count * dimension))
        return torch.stack(modality_losses).mean()


def masked_centered_cosine_loss(
    predictions: ModalityPredictions,
    full_features: torch.Tensor,
    availability_mask: torch.Tensor,
    umask: torch.Tensor,
    means: ModalityMeans,
):
    """Average per-modality cosine distance over truly missing utterances."""
    dimensions = (means.audio.numel(), means.text.numel(), means.visual.numel())
    targets = torch.split(full_features, dimensions, dim=-1)
    prediction_tensors = (predictions.audio, predictions.text, predictions.visual)
    mean_tensors = (means.audio, means.text, means.visual)
    names = ("audio", "text", "visual")
    valid_utterance = umask.transpose(0, 1).bool()
    losses = []
    counts = {}
    for index, name in enumerate(names):
        selected = valid_utterance & (availability_mask[..., index] == 0)
        count = int(selected.sum().item())
        counts[name] = count
        if count == 0:
            continue
        predicted = prediction_tensors[index][selected]
        target = (targets[index][selected] - mean_tensors[index]).detach()
        losses.append((1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean())
    if not losses:
        zero = predictions.audio.sum() * 0.0
        return zero, counts
    return torch.stack(losses).mean(), counts


## for reconstruction [only recon loss on miss part]
class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        assert len(recon_input) == 1
        recon = recon_input[0] # [seqlen, batch, dim]
        target = target_input[0].detach() # [seqlen, batch, dim]
        mask = input_mask[0] # [seqlen, batch, 3]
        real = umask.transpose(0, 1).bool()

        dimensions = (adim, tdim, vdim)
        recon_parts = torch.split(recon, dimensions, dim=-1)
        target_parts = torch.split(target, dimensions, dim=-1)
        modality_losses = []
        for index, (recon_part, target_part) in enumerate(
            zip(recon_parts, target_parts)
        ):
            selected = real & (mask[..., index] == 0)
            if not selected.any():
                continue
            squared_error = self.loss(recon_part[selected], target_part[selected])
            modality_losses.append(squared_error.sum() / squared_error.numel())

        if not modality_losses:
            return recon.sum() * 0.0
        return torch.stack(modality_losses).mean()


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1,1) # [batch*seq_len, 1]
        target = target.view(-1,1) # [batch*seq_len, 1]
        pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
        loss = self.loss(pred*umask, (target*umask).squeeze().long()) / torch.sum(umask) 
        return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch*seq_len]
        """
        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]
        umask = umask.view(-1, 1) # [batch*seq_len, 1]
        loss = self.loss(pred*umask, target*umask) / torch.sum(umask)
        return loss
