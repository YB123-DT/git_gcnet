"""Pattern-lattice conditional innovation JEPA components."""

from .patterns import (
    ACTIVE_PATTERNS,
    expand_modality_mask,
    sample_balanced_patterns,
)

__all__ = [
    "ACTIVE_PATTERNS",
    "expand_modality_mask",
    "sample_balanced_patterns",
]
