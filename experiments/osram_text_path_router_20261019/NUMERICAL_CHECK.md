# Candidate readout layout parity

Pre-training GPU smoke on source `84c841e` failed closed at the old-cache
single-removal logit equality gate; no formal training was started from it.
Original full logits, input IDs and masks were exact. Maximum single-removal
differences were7.1525574e-7 (MELD) and1.4305115e-6 (IEMOCAP-6).

A layerwise real MELD probe compared the new candidate helper with the successful
four-condition replay on the same frozen batch. Adapter inputs, LayerNorm outputs
and head-input values were identical, but head-input strides differed:
`(1600,35200,1)` from condition-first `torch.where` versus `(51200,1600,1)`
from original multiplicative padding. The differing layout can select a different
CUDA matrix multiplication path on particular batch shapes. The first full batch
had equal logits, so single-small-fixture numerical equality alone missed this.

CPU regression reproduced the layout mismatch exactly: new `(10,40,1)` versus
reference `(20,10,1)`. This test was RED before correction. The correction restores
the original `hidden * valid_mask.to(hidden.dtype)` operation before the frozen
classifier. Input-source masking remains safe `torch.where`; no read/write,
selector, source contribution, probability mixture or test tolerance is changed.

The corrected snapshot must pass real candidate parity before formal training.
Keep both failed smoke directories and their logs for provenance; do not mix them
with the corrected finite training runs or call them performance experiments.

Completed verification: corrected source `282b26a` passed both real GPU smoke
jobs, then all eight test-rate candidate parity gates for each dataset after the
finite training runs. Original equality assertions remained unchanged throughout.
