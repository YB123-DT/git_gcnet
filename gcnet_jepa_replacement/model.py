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
        enable_stability_reconstruction=False,
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
            enable_stability_reconstruction=enable_stability_reconstruction,
        )
        hidden_dim = 2 * D_e + graph_hidden_size
        rng_state = torch.get_rng_state()
        self.modality_predictor = ModalityPredictor(
            hidden_dim, adim, tdim, vdim, predictor_dropout
        )
        torch.set_rng_state(rng_state)

    def forward(
        self,
        inputfeats,
        qmask,
        umask,
        seq_lengths,
        predict_modalities=True,
        detach_predictor_input=False,
    ):
        log_prob, no_reconstruction, hidden = super().forward(
            inputfeats, qmask, umask, seq_lengths
        )
        predictor_input = hidden.detach() if detach_predictor_input else hidden
        predictions = self.modality_predictor(predictor_input) if predict_modalities else None
        return log_prob, no_reconstruction, hidden, predictions
