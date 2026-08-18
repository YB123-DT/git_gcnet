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
        target = target_input[0] # [seqlen, batch, dim]
        mask = input_mask[0] # [seqlen, batch, 3]

        recon  = torch.reshape(recon, (-1, recon.size(2)))   # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2))) # [seqlen*batch, dim]
        mask   = torch.reshape(mask, (-1, mask.size(2)))     # [seqlen*batch, 3] 1(exist); 0(mask)
        umask = torch.reshape(umask, (-1, 1)) # [seqlen*batch, 1]

        A_rec = recon[:, :adim]
        L_rec = recon[:, adim:adim+tdim]
        V_rec = recon[:, adim+tdim:]
        A_full = target[:, :adim]
        L_full = target[:, adim:adim+tdim]
        V_full = target[:, adim+tdim:]
        A_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 2], (-1, 1))

        loss_recon1 = self.loss(A_rec*umask, A_full*umask) * -1 * (A_miss_index - 1)
        loss_recon2 = self.loss(L_rec*umask, L_full*umask) * -1 * (L_miss_index - 1)
        loss_recon3 = self.loss(V_rec*umask, V_full*umask) * -1 * (V_miss_index - 1)
        loss_recon1 = torch.sum(loss_recon1) / adim
        loss_recon2 = torch.sum(loss_recon2) / tdim
        loss_recon3 = torch.sum(loss_recon3) / vdim
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon


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
