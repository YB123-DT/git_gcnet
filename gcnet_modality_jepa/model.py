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
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch_geometric.nn import RGCNConv, GraphConv

from .module import *
from .graph import batch_graphify


def _run_recurrent(
    recurrent,
    values,
    seq_lengths,
    padding_mode,
    umask=None,
):
    """Run a sequence encoder with explicit conversation-length semantics."""
    if padding_mode == "legacy":
        return recurrent(values)[0]
    if padding_mode != "packed":
        raise ValueError("padding_mode must be 'legacy' or 'packed'")
    lengths = [int(length) for length in seq_lengths]
    if len(lengths) != values.shape[1]:
        raise ValueError("seq_lengths must contain one length per batch item")
    if any(length <= 0 or length > values.shape[0] for length in lengths):
        raise ValueError("seq_lengths must be within [1, sequence length]")
    if umask is None or umask.shape != (values.shape[1], values.shape[0]):
        raise ValueError("packed recurrent requires umask with shape [B, L]")
    if not bool(((umask == 0) | (umask == 1)).all()):
        raise ValueError("umask must be binary")
    length_tensor = torch.tensor(lengths, device=umask.device)
    expected_mask = (
        torch.arange(values.shape[0], device=umask.device).unsqueeze(0)
        < length_tensor.unsqueeze(1)
    )
    if not torch.equal(umask.bool(), expected_mask):
        raise ValueError(
            "seq_lengths and umask must define the same contiguous prefix"
        )
    packed = pack_padded_sequence(
        values,
        lengths,
        enforce_sorted=False,
    )
    packed_output = recurrent(packed)[0]
    output, _ = pad_packed_sequence(
        packed_output,
        total_length=values.shape[0],
    )
    return output


class ModalityPredictions:
    """Typed container for [seq, batch, modality_dim] predictions."""

    def __init__(self, audio, text, visual):
        self.audio = audio
        self.text = text
        self.visual = visual


def _prediction_head(input_dim, hidden_dim, output_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class ModalityPredictor(nn.Module):
    """Independent A/T/V heads conditioned on the GCNet conversation state."""

    def __init__(self, hidden_dim, audio_dim, text_dim, visual_dim, dropout=0.1):
        super().__init__()
        self.audio_head = _prediction_head(hidden_dim, 256, audio_dim, dropout)
        self.text_head = _prediction_head(hidden_dim, 512, text_dim, dropout)
        self.visual_head = _prediction_head(hidden_dim, 512, visual_dim, dropout)

    def forward(self, hidden):
        return ModalityPredictions(
            audio=self.audio_head(hidden),
            text=self.text_head(hidden),
            visual=self.visual_head(hidden),
        )


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64,
                 dropout=0.5, no_cuda=False,
                 recurrent_padding_mode="legacy",
                 graph_message_calibration="none",
                 graph_second_layer="graphconv"):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda 
        self.time_attn = time_attn
        self.hidden_size = hidden_size
        if recurrent_padding_mode not in {"legacy", "packed"}:
            raise ValueError(
                "recurrent_padding_mode must be 'legacy' or 'packed'"
            )
        self.recurrent_padding_mode = recurrent_padding_mode
        if graph_message_calibration not in {
            "none",
            "branch-layernorm-residual",
        }:
            raise ValueError("unsupported graph_message_calibration")
        self.graph_message_calibration = graph_message_calibration
        if graph_second_layer not in {"graphconv", "identity"}:
            raise ValueError("unsupported graph_second_layer")
        self.graph_second_layer = graph_second_layer

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        if self.graph_message_calibration == "branch-layernorm-residual":
            self.message_calibration_alpha = nn.Parameter(
                torch.zeros(hidden_size)
            )

        ## nodal attention
        D_h = num_features+hidden_size
        self.grufusion = nn.LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2*D_h, 2*D_h, att_type='general2')
        self.linear = nn.Linear(2*D_h, D_h)
        self.record_activation_diagnostics = False
        self.last_pre_activation = None
        self.last_hidden = None

    def _calibrate_graph_message(self, message):
        if self.graph_message_calibration == "none":
            return message
        normalized = F.layer_norm(message, (message.shape[-1],))
        blend = torch.tanh(self.message_calibration_alpha)
        return message + blend * (normalized - message)


    def forward(
        self,
        features,
        edge_index,
        edge_type,
        seq_lengths,
        umask,
        postgraph_recurrent=None,
    ):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''

        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type) # [num_features -> hidden_size]
        if self.graph_second_layer == "graphconv":
            out = self.conv2(out, edge_index) # [hidden_size -> hidden_size]
        out = self._calibrate_graph_message(out)
        outputs = torch.cat([features, out], dim=-1) # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1)) # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda) # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1) # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1)) # [seqlen, batch, dim]
        recurrent = (
            self.grufusion
            if postgraph_recurrent is None
            else postgraph_recurrent
        )
        outputs = _run_recurrent(
            recurrent,
            outputs,
            seq_lengths,
            self.recurrent_padding_mode,
            umask,
        ) # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs: # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0)) # [1, batch, mem_dim]
                alpha.append(alpha_[:,0,:]) # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0) # [seqlen, batch, mem_dim]
            pre_activation = self.linear(att_emotions)
        else:
            alpha = []
            pre_activation = self.linear(outputs)
        hidden = F.relu(pre_activation) # [seqlen, batch, D_h]
        if self.recurrent_padding_mode == "packed":
            valid = umask[:, : hidden.shape[0]].T.unsqueeze(-1)
            hidden = hidden * valid.to(hidden.dtype)
        if self.record_activation_diagnostics:
            self.last_pre_activation = pre_activation.detach()
            self.last_hidden = hidden.detach()
        else:
            self.last_pre_activation = None
            self.last_hidden = None

        return hidden # [seqlen, batch, D_h]

        
'''
base_model: LSTM or GRU
adim, tdim, vdim: input feature dim
D_e: hidder feature dimensions of base_model is 2*D_e
D_g, D_p, D_h, D_a, graph_hidden_size
'''
class GraphModel(nn.Module):

    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes ,dropout=0.5, time_attn=True, no_cuda=False,
                 enable_reconstruction=True,
                 enable_stability_reconstruction=False,
                 graph_branch_mode="both",
                 recurrent_padding_mode="legacy",
                 postgraph_sequence_mode="independent",
                 graph_message_calibration="none",
                 graph_second_layer="graphconv"):
        
        super(GraphModel, self).__init__()

        self.no_cuda = no_cuda
        self.base_model = base_model
        if graph_branch_mode not in {"both", "temporal-only", "speaker-only"}:
            raise ValueError(
                "graph_branch_mode must be 'both', 'temporal-only', or "
                "'speaker-only'"
            )
        self.graph_branch_mode = graph_branch_mode
        if postgraph_sequence_mode not in {"independent", "shared-bilstm"}:
            raise ValueError(
                "postgraph_sequence_mode must be 'independent' or "
                "'shared-bilstm'"
            )
        if (
            postgraph_sequence_mode == "shared-bilstm"
            and graph_branch_mode != "both"
        ):
            raise ValueError(
                "shared-bilstm requires graph_branch_mode='both'"
            )
        self.postgraph_sequence_mode = postgraph_sequence_mode
        if graph_message_calibration not in {
            "none",
            "branch-layernorm-residual",
        }:
            raise ValueError("unsupported graph_message_calibration")
        self.graph_message_calibration = graph_message_calibration
        if graph_second_layer not in {"graphconv", "identity"}:
            raise ValueError("unsupported graph_second_layer")
        self.graph_second_layer = graph_second_layer
        if recurrent_padding_mode not in {"legacy", "packed"}:
            raise ValueError(
                "recurrent_padding_mode must be 'legacy' or 'packed'"
            )
        self.recurrent_padding_mode = recurrent_padding_mode

        # The base model is the sequential context encoder.
        # Change input features => 2*D_e
        self.lstm = nn.LSTM(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru = nn.GRU(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
       
        ## Defination for graph model
        ## [modality_type=3(AVT); time_order=3(past, now, future)]
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(
            2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout,
            self.no_cuda, self.recurrent_padding_mode,
            self.graph_message_calibration, self.graph_second_layer
        )
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(
            2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout,
            self.no_cuda, self.recurrent_padding_mode,
            self.graph_message_calibration, self.graph_second_layer
        )
        if self.postgraph_sequence_mode == "shared-bilstm":
            for parameter in self.graph_net_speaker.grufusion.parameters():
                parameter.requires_grad_(False)

        ## classification and reconstruction
        D_h = 2*D_e + graph_hidden_size
        self.smax_fc  = nn.Linear(D_h, n_classes)
        stability_rng_state = torch.get_rng_state()
        self.enable_stability_reconstruction = enable_stability_reconstruction
        self.enable_reconstruction = enable_reconstruction
        if enable_reconstruction:
            self.linear_rec = nn.Linear(D_h, adim+tdim+vdim)
        if enable_stability_reconstruction:
            rng_state_after_existing_heads = torch.get_rng_state()
            try:
                torch.set_rng_state(stability_rng_state)
                self.stability_rec_head = nn.Linear(D_h, adim+tdim+vdim)
            finally:
                torch.set_rng_state(rng_state_after_existing_heads)

    def reconstruct_stability(self, hidden):
        if not self.enable_stability_reconstruction:
            raise RuntimeError("stability reconstruction is disabled")
        return self.stability_rec_head(hidden)

    def encode_hidden(
        self,
        inputfeats,
        qmask,
        umask,
        seq_lengths,
        pre_graph_residual=None,
    ):
        """Run the Original recurrent and graph path."""
        if self.base_model == 'LSTM':
            outputs = _run_recurrent(
                self.lstm,
                inputfeats[0],
                seq_lengths,
                self.recurrent_padding_mode,
                umask,
            )
        elif self.base_model == 'GRU':
            outputs = _run_recurrent(
                self.gru,
                inputfeats[0],
                seq_lengths,
                self.recurrent_padding_mode,
                umask,
            )
        else:
            raise ValueError("base_model must be LSTM or GRU")

        if pre_graph_residual is not None:
            if pre_graph_residual.shape != outputs.shape:
                raise ValueError(
                    "pre_graph_residual must match recurrent outputs [L, B, 2*D_e]"
                )
            outputs = outputs + pre_graph_residual
        outputs = outputs.unsqueeze(2)

        hidden1 = None
        if self.graph_branch_mode in {"both", "temporal-only"}:
            features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers,
                                                                 self.window_past, self.window_future, 'temporal', self.no_cuda)
            assert len(edge_type_mapping) == 3
            hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        hidden2 = None
        if self.graph_branch_mode in {"both", "speaker-only"}:
            features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers,
                                                                 self.window_past, self.window_future, 'speaker', self.no_cuda)
            assert len(edge_type_mapping) == self.n_speakers ** 2
            if self.postgraph_sequence_mode == "shared-bilstm":
                hidden2 = self.graph_net_speaker(
                    features,
                    edge_index,
                    edge_type,
                    seq_lengths,
                    umask,
                    postgraph_recurrent=self.graph_net_temporal.grufusion,
                )
            else:
                hidden2 = self.graph_net_speaker(
                    features, edge_index, edge_type, seq_lengths, umask
                )
        if self.graph_branch_mode == "both":
            return hidden1 + hidden2
        if self.graph_branch_mode == "temporal-only":
            return hidden1
        return hidden2

    def forward(self, inputfeats, qmask, umask, seq_lengths):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        """

        hidden = self.encode_hidden(inputfeats, qmask, umask, seq_lengths)

        ## for classification
        log_prob = self.smax_fc(hidden) # [seqlen, batch, n_classes]

        ## for reconstruction
        rec_outputs = [self.linear_rec(hidden)] if self.enable_reconstruction else []

        return log_prob, rec_outputs, hidden


class ModalityJEPAGraphModel(GraphModel):
    """Original GCNet plus predictors used only by the training objective."""

    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size,
                 n_speakers, window_past, window_future, n_classes,
                 dropout=0.5, time_attn=True, no_cuda=False,
                 predictor_dropout=0.1,
                 enable_stability_reconstruction=False):
        super().__init__(
            base_model, adim, tdim, vdim, D_e, graph_hidden_size,
            n_speakers, window_past, window_future, n_classes,
            dropout, time_attn, no_cuda,
            enable_stability_reconstruction=enable_stability_reconstruction,
        )
        hidden_dim = 2 * D_e + graph_hidden_size
        rng_state = torch.get_rng_state()
        self.modality_predictor = ModalityPredictor(
            hidden_dim, adim, tdim, vdim, predictor_dropout
        )
        torch.set_rng_state(rng_state)

    def predict_modalities(self, hidden, enabled=True, detach_input=False):
        if not enabled:
            return None
        predictor_input = hidden.detach() if detach_input else hidden
        return self.modality_predictor(predictor_input)

    def forward(
        self,
        inputfeats,
        qmask,
        umask,
        seq_lengths,
        predict_modalities=True,
        detach_predictor_input=False,
    ):
        log_prob, rec_outputs, hidden = super().forward(
            inputfeats, qmask, umask, seq_lengths
        )
        predictions = self.predict_modalities(
            hidden,
            enabled=predict_modalities,
            detach_input=detach_predictor_input,
        )
        return log_prob, rec_outputs, hidden, predictions
