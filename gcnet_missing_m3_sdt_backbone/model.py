"""Full-context SDT-style Transformer backbone for conversation features."""

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEncoding(nn.Module):
    """Add a persistent, fixed sinusoidal encoding to ``[L, B, D]`` inputs."""

    def __init__(self, dim=384, max_len=512):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError("dim must be a positive even integer")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer")

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", encoding, persistent=True)

    def forward(self, values):
        if values.dim() != 3:
            raise ValueError("position encoding input must have shape [L, B, D]")
        if values.size(-1) != self.pe.size(-1):
            raise ValueError("position encoding dimension does not match the input")
        sequence_length = values.size(0)
        if sequence_length > self.pe.size(0):
            raise ValueError(
                "sequence length {} exceeds configured max_len {}".format(
                    sequence_length,
                    self.pe.size(0),
                )
            )
        position = self.pe[:sequence_length].unsqueeze(1)
        return values + position.to(dtype=values.dtype)


class PreNormTransformerLayer(nn.Module):
    """Torch-1.8-compatible pre-norm, full-context Transformer layer."""

    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.norm_first = True
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        normalized = self.norm1(src)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout1(attended)

        normalized = self.norm2(src)
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(normalized)))
        )
        return src + self.dropout2(feed_forward)


class SDTStyleConversationBackbone(nn.Module):
    """Encode padded conversations with speaker-aware full-context attention."""

    def __init__(
        self,
        input_dim=256,
        output_dim=250,
        n_speakers=1,
        d_model=384,
        num_heads=8,
        num_layers=5,
        ff_dim=704,
        dropout=0.5,
        max_len=512,
    ):
        super().__init__()
        self._validate_configuration(
            input_dim=input_dim,
            output_dim=output_dim,
            n_speakers=n_speakers,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_speakers = n_speakers
        self.max_len = max_len

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model, max_len)
        self.speaker_embedding = nn.Embedding(
            n_speakers + 1,
            d_model,
            padding_idx=n_speakers,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                PreNormTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.output_activation = nn.ReLU()

    @staticmethod
    def _validate_configuration(
        input_dim,
        output_dim,
        n_speakers,
        d_model,
        num_heads,
        num_layers,
        ff_dim,
        dropout,
        max_len,
    ):
        positive_integers = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "n_speakers": n_speakers,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "max_len": max_len,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal position encoding")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

    def _validate_inputs(self, values, qmask, umask, seq_lengths):
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
        if sequence_length > self.max_len:
            raise ValueError(
                "sequence length {} exceeds configured max_len {}".format(
                    sequence_length,
                    self.max_len,
                )
            )

        expected_mask_shape = (batch_size, sequence_length)
        if not isinstance(umask, Tensor) or tuple(umask.shape) != expected_mask_shape:
            raise ValueError("umask must have shape [B, L]")
        expected_qmask_shape = (batch_size, sequence_length, self.n_speakers)
        if not isinstance(qmask, Tensor) or tuple(qmask.shape) != expected_qmask_shape:
            raise ValueError("qmask must have shape [B, L, n_speakers]")
        if umask.device != values.device or qmask.device != values.device:
            raise ValueError("values, qmask, and umask must be on the same device")

        if not torch.all((umask == 0) | (umask == 1)).item():
            raise ValueError("umask must contain only binary values")
        if not torch.all((qmask == 0) | (qmask == 1)).item():
            raise ValueError("qmask must contain only binary values")

        try:
            lengths = torch.as_tensor(seq_lengths, device=umask.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("seq_lengths must be a one-dimensional integer sequence") from error
        if lengths.dim() != 1 or lengths.numel() != batch_size:
            raise ValueError("seq_lengths must contain one length per batch item")
        if lengths.dtype == torch.bool:
            raise ValueError("seq_lengths must contain integer lengths")
        if lengths.is_floating_point():
            if not torch.isfinite(lengths).all().item():
                raise ValueError("seq_lengths must be finite")
            if not torch.equal(lengths, lengths.round()):
                raise ValueError("seq_lengths must contain integer lengths")
        lengths = lengths.to(dtype=torch.long)
        if torch.any(lengths <= 0).item() or torch.any(lengths > sequence_length).item():
            raise ValueError("seq_lengths values must be in [1, L]")

        positions = torch.arange(sequence_length, device=umask.device).unsqueeze(0)
        expected_valid = positions < lengths.unsqueeze(1)
        if not torch.equal(umask.bool(), expected_valid):
            raise ValueError(
                "umask must be a contiguous valid prefix matching seq_lengths"
            )

        speaker_counts = qmask.to(dtype=torch.long).sum(dim=-1)
        if not torch.all(speaker_counts[expected_valid] == 1).item():
            raise ValueError(
                "each valid qmask position must identify exactly one speaker"
            )
        return expected_valid

    def forward(self, values, qmask, umask, seq_lengths):
        valid = self._validate_inputs(values, qmask, umask, seq_lengths)
        speaker_ids = qmask.to(dtype=torch.long).argmax(dim=-1)
        speaker_ids = speaker_ids.masked_fill(~valid, self.n_speakers)

        hidden = self.input_projection(values)
        hidden = self.position_encoding(hidden)
        hidden = hidden + self.speaker_embedding(speaker_ids).transpose(0, 1)
        hidden = self.input_dropout(hidden)

        padding_mask = ~valid
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)

        hidden = self.final_norm(hidden)
        output = self.output_activation(self.output_projection(hidden))
        output_padding = padding_mask.transpose(0, 1).unsqueeze(-1)
        return output.masked_fill(output_padding, 0.0)


__all__ = [
    "PreNormTransformerLayer",
    "SDTStyleConversationBackbone",
    "SinusoidalPositionEncoding",
]
