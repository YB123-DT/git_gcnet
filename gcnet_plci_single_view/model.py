"""Natural-view target selection for PLCI-JEPA."""

from gcnet_plci_jepa.model import PLCIJEPAGraphModel


class SingleViewPLCIJEPAGraphModel(PLCIJEPAGraphModel):
    """Reuse Natural GCNet states for training-only PLCI prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prediction_umask = None

    def predict_natural(
        self,
        student_latents,
        hidden,
        availability,
        umask,
    ):
        valid = self._validate_availability(
            availability,
            umask,
            allow_atv=True,
        )
        incomplete = availability.sum(dim=-1).lt(3) & valid
        prediction_umask = incomplete.T.to(dtype=umask.dtype)
        self.last_prediction_umask = prediction_umask.detach().clone()
        return self.predictor(
            student_latents,
            hidden,
            availability,
            prediction_umask,
        )
