from __future__ import annotations

import torch

from gcnet_modality_jepa.model import GraphModel, ModalityPredictor


class ReplacementJEPAGraphModel(GraphModel):
    """GCNet encoder where JEPA replaces, rather than augments, reconstruction."""

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
        time_attn=True,
        no_cuda=False,
        predictor_dropout=0.1,
    ):
        super().__init__(
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
            dropout,
            time_attn,
            no_cuda,
            enable_reconstruction=False,
        )
        hidden_dim = 2 * D_e + graph_hidden_size
        rng_state = torch.get_rng_state()
        self.modality_predictor = ModalityPredictor(
            hidden_dim, adim, tdim, vdim, predictor_dropout
        )
        torch.set_rng_state(rng_state)

    def forward(
        self, inputfeats, qmask, umask, seq_lengths, predict_modalities=True
    ):
        log_prob, no_reconstruction, hidden = super().forward(
            inputfeats, qmask, umask, seq_lengths
        )
        predictions = (
            self.modality_predictor(hidden) if predict_modalities else None
        )
        return log_prob, no_reconstruction, hidden, predictions
