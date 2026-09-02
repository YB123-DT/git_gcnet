from __future__ import annotations

from gcnet_modality_jepa.model import GraphModel


class OriginalGCNetControl(GraphModel):
    """Original GCNet exposed through the shared missing-rate evaluator API."""

    reconstruction_loss_variant = "corrected-formal-repo"

    def __init__(
        self,
        base_model,
        adim,
        tdim,
        vdim,
        D_e,
        graph_hidden_size,
        n_speakers,
        window_past,
        window_future,
        n_classes,
        dropout=0.5,
        time_attn=False,
        no_cuda=False,
    ):
        super().__init__(
            base_model=base_model,
            adim=adim,
            tdim=tdim,
            vdim=vdim,
            D_e=D_e,
            graph_hidden_size=graph_hidden_size,
            n_speakers=n_speakers,
            window_past=window_past,
            window_future=window_future,
            n_classes=n_classes,
            dropout=dropout,
            time_attn=time_attn,
            no_cuda=no_cuda,
            enable_reconstruction=True,
            enable_stability_reconstruction=False,
            graph_branch_mode="both",
            recurrent_padding_mode="legacy",
            postgraph_sequence_mode="independent",
            graph_message_calibration="none",
        )

    def forward(
        self,
        inputfeats,
        availability,
        qmask,
        umask,
        seq_lengths,
        predict_missing=False,
    ):
        if predict_missing:
            raise ValueError(
                "OriginalGCNetControl does not support predict_missing=True"
            )
        del availability
        logits, reconstruction, hidden = super().forward(
            inputfeats,
            qmask,
            umask,
            seq_lengths,
        )
        return logits, reconstruction, hidden, None
