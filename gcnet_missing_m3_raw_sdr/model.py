"""Registered raw-residual input treatment for the public SDR backbone."""

from __future__ import annotations

import inspect

from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3_sdr_backbone.model import MissingM3SDRModel


class MissingM3RawSDRModel(MissingM3SDRModel):
    """Thin fixed-identity wrapper around :class:`MissingM3SDRModel`."""

    _GRAPH_IDENTITY = {
        "fusion_type": "raw-residual",
        "representation_type": "slot",
    }

    def __init__(self, *args, **kwargs):
        sdr_variant = kwargs.pop("sdr_variant", "sdr-public")
        if sdr_variant != "sdr-public":
            raise ValueError("sdr_variant must be 'sdr-public'")
        sdr_input_type = kwargs.pop("sdr_input_type", "raw-residual")
        if sdr_input_type != "raw-residual":
            raise ValueError("sdr_input_type must be 'raw-residual'")

        bound = inspect.signature(MissingM3GraphModel.__init__).bind_partial(
            self,
            *args,
            **kwargs,
        )
        for name, expected in self._GRAPH_IDENTITY.items():
            if name in bound.arguments:
                if bound.arguments[name] != expected:
                    raise ValueError("{} must be {!r}".format(name, expected))
            else:
                kwargs[name] = expected

        super().__init__(
            *args,
            sdr_variant="sdr-public",
            sdr_input_type="raw-residual",
            **kwargs,
        )


__all__ = ["MissingM3RawSDRModel"]
