"""SDR-GNN conversation backbones for Missing-M3 representations."""

import inspect

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcnet_missing_m3.model import MissingM3GraphModel

from .layers import (
    SDRRelationBranch,
    _run_packed_bigru,
    _validated_dropout,
    _validated_umask,
)


class SDRConversationBackbone(nn.Module):
    """Encode padded conversations with public-effective or paper SDR paths."""

    VARIANTS = ("sdr-public", "sdr-paper")

    def __init__(
        self,
        variant="sdr-public",
        input_dim=256,
        recurrent_hidden=200,
        graph_hidden=100,
        n_speakers=1,
        window_past=2,
        window_future=2,
        dropout=0.5,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "recurrent_hidden": recurrent_hidden,
            "graph_hidden": graph_hidden,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if variant not in self.VARIANTS:
            raise ValueError("variant must be 'sdr-public' or 'sdr-paper'")
        if (
            isinstance(n_speakers, bool)
            or not isinstance(n_speakers, int)
            or n_speakers not in (1, 2)
        ):
            raise ValueError("n_speakers must be 1 or 2")
        dropout = _validated_dropout(dropout)

        self.variant = variant
        self.input_dim = input_dim
        self.recurrent_hidden = recurrent_hidden
        self.recurrent_dim = 2 * recurrent_hidden
        self.graph_hidden = graph_hidden
        self.output_dim = self.recurrent_dim + graph_hidden
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future

        self.pre_graph_bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=recurrent_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        branch_arguments = {
            "recurrent_dim": self.recurrent_dim,
            "graph_hidden": graph_hidden,
            "n_speakers": n_speakers,
            "window_past": window_past,
            "window_future": window_future,
            "dropout": dropout,
        }
        self.temporal_branch = SDRRelationBranch(
            num_relations=3,
            relation="temporal",
            **branch_arguments
        )
        if variant == "sdr-paper":
            self.speaker_branch = SDRRelationBranch(
                num_relations=n_speakers ** 2,
                relation="speaker",
                **branch_arguments
            )
            self.fusion = nn.Linear(2 * self.output_dim, self.output_dim)

    def _validate_inputs(self, values, qmask, umask, lengths):
        if not isinstance(values, Tensor) or values.dim() != 3:
            raise ValueError("values must be a tensor with shape [L, B, D]")
        sequence_length, batch_size, feature_dim = values.shape
        if sequence_length <= 0 or batch_size <= 0:
            raise ValueError("values must contain a non-empty sequence and batch")
        if feature_dim != self.input_dim:
            raise ValueError(
                "values feature dimension must be {}, got {}".format(
                    self.input_dim,
                    feature_dim,
                )
            )
        if not values.is_floating_point():
            raise ValueError("values must use a floating-point dtype")

        expected_mask_shape = (batch_size, sequence_length)
        if not isinstance(qmask, Tensor) or tuple(qmask.shape) != expected_mask_shape:
            raise ValueError("qmask must have shape [B, L]")
        if not isinstance(umask, Tensor):
            raise ValueError("umask must have shape [B, L]")
        if qmask.device != values.device or umask.device != values.device:
            raise ValueError("values, qmask, and umask must share a device")

        normalized_lengths, valid = _validated_umask(
            umask,
            lengths,
            batch_size,
            sequence_length,
        )

        valid_speakers = qmask[valid]
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if valid_speakers.numel():
            if valid_speakers.dtype == torch.bool:
                raise ValueError("valid speaker ids must be integers, not booleans")
            if valid_speakers.is_floating_point():
                if not torch.isfinite(valid_speakers).all().item():
                    raise ValueError("valid speaker ids must be finite integers")
                if not torch.equal(valid_speakers, valid_speakers.round()):
                    raise ValueError("valid speaker ids must be integers")
            elif valid_speakers.dtype not in integer_dtypes:
                raise ValueError("valid speaker ids must be numeric integers")
            if (
                torch.any(valid_speakers < 0).item()
                or torch.any(valid_speakers >= self.n_speakers).item()
            ):
                raise ValueError(
                    "valid speaker ids must be in [0, n_speakers - 1]"
                )
        return normalized_lengths, valid

    def forward(self, values, qmask, umask, lengths):
        normalized_lengths, valid = self._validate_inputs(
            values,
            qmask,
            umask,
            lengths,
        )
        value_padding = ~valid.transpose(0, 1).unsqueeze(-1)
        values = values.masked_fill(value_padding, 0.0)
        recurrent = _run_packed_bigru(
            self.pre_graph_bigru,
            values,
            normalized_lengths,
        )
        temporal = self.temporal_branch(
            recurrent,
            qmask,
            umask,
            normalized_lengths,
        )
        if self.variant == "sdr-public":
            hidden = temporal
        else:
            speaker = self.speaker_branch(
                recurrent,
                qmask,
                umask,
                normalized_lengths,
            )
            hidden = F.relu(self.fusion(torch.cat((temporal, speaker), dim=-1)))
        return hidden.masked_fill(value_padding, 0.0)


class MissingM3SDRModel(MissingM3GraphModel):
    """Missing-M3 with its complete conversation path replaced by SDR."""

    def __init__(
        self,
        *args,
        sdr_variant="sdr-public",
        sdr_input_type="slot",
        **kwargs,
    ):
        parent_arguments = inspect.signature(
            MissingM3GraphModel.__init__
        ).bind(self, *args, **kwargs)
        parent_arguments.apply_defaults()
        configuration = parent_arguments.arguments
        dropout = configuration["dropout"]

        if isinstance(dropout, bool):
            raise TypeError("dropout must be a real probability, not bool")
        if sdr_input_type not in {"slot", "raw-residual"}:
            raise ValueError("sdr_input_type must be 'slot' or 'raw-residual'")
        locked_configuration = {
            "base_model": "LSTM",
            "fusion_type": sdr_input_type,
            "representation_type": "slot",
            "classification_completion": False,
            "graph_branch_mode": "both",
            "time_attn": False,
            "local_context_residual": False,
            "node_interaction_residual": False,
            "readout_type": "shared",
            "mmoe_variant": "dual-gate",
            "recurrent_padding_mode": "legacy",
            "postgraph_sequence_mode": "independent",
            "graph_message_calibration": "none",
        }
        for name, expected in locked_configuration.items():
            actual = configuration[name]
            matches = (
                actual is expected
                if isinstance(expected, bool)
                else actual == expected
            )
            if not matches:
                raise ValueError("{} must be {!r}".format(name, expected))
        if sdr_variant not in SDRConversationBackbone.VARIANTS:
            raise ValueError("sdr_variant must be 'sdr-public' or 'sdr-paper'")

        super().__init__(*args, **kwargs)

        for name in (
            "lstm",
            "gru",
            "graph_net_temporal",
            "graph_net_speaker",
        ):
            if hasattr(self, name):
                delattr(self, name)

        self.sdr_variant = sdr_variant
        self.sdr_input_type = sdr_input_type
        self.conversation_backbone = SDRConversationBackbone(
            variant=sdr_variant,
            input_dim=(
                self.latent_dim
                if self.sdr_input_type == "slot"
                else sum(self.dimensions)
            ),
            recurrent_hidden=configuration["D_e"],
            graph_hidden=configuration["graph_hidden_size"],
            n_speakers=self.n_speakers,
            window_past=self.window_past,
            window_future=self.window_future,
            dropout=dropout,
        )
        if self.conversation_backbone.output_dim != self.smax_fc.in_features:
            raise RuntimeError(
                "SDR backbone output width must match the Missing-M3 heads"
            )

    def encode_hidden(
        self,
        inputfeats,
        qmask,
        umask,
        seq_lengths,
        pre_graph_residual=None,
    ):
        if pre_graph_residual is not None:
            raise ValueError("pre_graph_residual is unsupported by the SDR backbone")
        return self.conversation_backbone(
            self._feature_tensor(inputfeats),
            qmask,
            umask,
            seq_lengths,
        )


__all__ = [
    "MissingM3SDRModel",
    "SDRConversationBackbone",
    "SDRRelationBranch",
]
