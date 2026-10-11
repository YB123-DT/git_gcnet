# Counterfactual Evidence Readouts

INTERNAL DIAGNOSTIC ONLY — per-rate BEST Test-oracle.

Approved scope: MOSI seed66, 150 epochs, two methods only.

- `marginal`: original Flat plus CEMR leave-one-evidence-out effect encoder.
- `coalition`: original Flat plus CIR singleton/pair Möbius effect encoder.

Both run one OSRAM trajectory, preserve the trainable original Flat anchor, use
only the original emotion MSE, and add a zero-initialized readout residual. All
Flat subset views share one training-time Dropout mask per utterance.

Protocol: cyclic random missing, rates0.0--0.7, first100 epochs at LR.001 then
full-state continuation to150 at LR.0001, per-rate Test-oracle selection. No
JEPA, completion, Gate, Nested, paired views, persistent mix, auxiliary loss or
Memory modification. Results remain pending until both verified runs complete.
