# Verification

Remote regression: 61 tests passed; one pre-existing PyG deprecation warning.
Checks include memory-only Adapter formula with Skip bias retained, default-off
behavior, Local-path perturbation leaving Adapter input unchanged, inactive Gap
and padding NaN sanitation, readout ablation masks, finite gradients, CLI/config
guards, and previous gate/readout regressions.

Actual cfg84 check on biggpu GPU6: A/T/V dimensions 512/1024/1024,
Local256, context1024, output1600. Adapter input is4096 rather than4352.
Shared non-Adapter initialization matches original Flat exactly. Original Adapter
construction precedes replacement under fork_rng so no later RNG stream changes.
Initial train/eval predictions have maximum absolute difference0.0 from Flat;
CUDA RNG remains equal. Three Adam steps have finite gradients and update
query_projector, emotion_adapter and local_skip. Nothing is frozen.

Preflight checked all three original configs and reference selection/provenance.
These are implementation checks, not evidence of a performance improvement.
