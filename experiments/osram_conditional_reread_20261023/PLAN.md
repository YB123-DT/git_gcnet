# Conditional second Memory read: approved paired implementation

The user supplied A1–A6 and approved implementation with “继续做”. This is
not another V3 connection: a new query reads the actual decayed pre-write CHM.

## Fixed design

Local + forward Base + current availability -> LN -> 128 GELU -> H*K query;
per-head L2 normalization; one read of the same pre-write state; zero-initialized
H*V -> H*V projection corrects forward Base only. No new dropout, task loss,
writer, Gap changes, or current-state writes. Original Local is computed once
and reused. Disabled mode retains the original execution and RNG order.

The free and span variants share all parameter tensors and initialization.
Span projects onto actual Base plus currently active, residualized Gap queries
using an explicit numerical-rank threshold before normalization. No inactive
Gap queries enter that space. Projection must have finite gradients at rank
deficiency. No historical source tracker is required.

## Implementation and verification sequence

- [ ] Core module and red/green tests: projection, rank, zero output, causal
  masking, genuine access to directions outside original query span, gradients.
- [ ] Opt-in OSRAM scan integration and config/parser/model plumbing; tests
  against pre-change implementation for default parameters, forward/backward,
  RNG, single Local evaluation, one scan, raw Gap/write invariance and causality.
- [ ] Independent finite experiment runner, exact reference masks/config/data,
  complete BEST/last/optimizer/RNG state and paired initialization checks.
- [ ] CPU regression tests and real MOSI/MOSEI GPU smoke, then immutable source
  commit and resource-admitted parallel seed66 free/span training.
- [ ] Preserve all old runs, publish provenance and explicit running/completed
  state. No unrequested seeds, HPO, query rounds or new losses.

## Training scope

MOSI seed66: 150 epochs, 3e-4 through 100 then 3e-5. MOSEI seed66: 60 epochs,
1e-3. Existing s0 environment, original GCNet random mask v1/cyclic 0–0.7,
batch32, Adam, WD1e-5, MSE, Flat1600 and per-rate Test-best W-F1 remain fixed.
All results are INTERNAL TEST-ORACLE SCREEN, not independent generalization.

The user did not establish that inadequate original addressing is the main
bottleneck. Free not improving over span rejects that interpretation for this
implementation; equal improvement does not establish a new-access mechanism.
