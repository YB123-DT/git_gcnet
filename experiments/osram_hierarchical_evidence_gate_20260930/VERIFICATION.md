# Verification — 2026-09-30

Implementation only; no formal training run or performance claim.

## Environment

- Server: biggpu, host GPU0 (not prohibited GPU4).
- Snapshot: `/data1/yb/remote_experiments/osram_hierarchical_gate_verification_20260930/code`.
- Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`.
- Parent revision: `31a3d86`, plus this scoped implementation.

## Results

`tests/verify_hierarchical_gate_cfg84.py` used original seed66 config at
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json`.
Inputs A/T/V=512/1024/1024, output_dim=1600. Actual context_dim=1024:
512 forward read channels plus a 512-channel zero backward slot;
forward_slot_reuse=false. The implementation preserves this existing interface.
An initial verifier assertion incorrectly assumed context_dim=512 and failed;
it was corrected after checking the original context concatenation, not by
altering Memory or Flat dimensions.

- All shared initial model state equal with flag on/off.
- Initial train and eval prediction max absolute difference: **0.0**.
- CUDA RNG state after each paired forward: identical.
- Three joint Adam steps: finite gradients, original query projector,
  emotion adapter, Local skip and both new gate output weights updated.
- All original OSRAM parameters remain trainable.

Remote pytest: **52 passed**, one existing torch_geometric deprecation warning:

```text
tests/test_hierarchical_evidence_gate.py
tests/test_evidence_gate_l1.py
tests/test_local_evidence_gate.py
tests/test_history_input_gate.py
tests/test_post_grn_integration.py
tests/test_memory_shift_integration.py
tests/test_osram_history_query.py
tests/test_conversation_mixed_masks.py
```

Checks include audio-only reweights .6/0/1.8/.6, K=1/3/4,
inactive/padding NaN safety and finite gradients, ablation isolation, filtered
Level2 conditioning, unchanged returned Memory contexts, original emotion-only
training loss, diagnostics aggregation and default-off behavior.

`git diff --check` passed. No new dependencies. No benchmark, convergence,
generalization or performance improvement has been verified. Feature diagnostics
include the preserved zero directional slot; they are not a measure of effective
memory utilization. Saturation is F<=.01 or F>=1.99. Normalized Level2 weights
do not guarantee bounded total representation norm or gates staying near one.
