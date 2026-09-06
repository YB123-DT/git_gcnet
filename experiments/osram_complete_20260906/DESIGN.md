# Full OSRAM backbone replacement

This directory records the CMU-MOSI backbone diagnostic for `feature/osram-complete`.
It is an internal diagnostic only, not a formal paper result.

## Scope

The only architectural change is replacement of the Missing-M3 conversational
backbone.  `ObservedSetEncoder`, Student Projectors, EMA Teacher, MMoE,
SmoothL1/InfoNCE JEPA objective, emotion head, cyclic rate schedule, feature
files, masks, optimizer, learning rate, batch size, and evaluation code are
otherwise unchanged.

The legacy backbone remains selectable as `--backbone-type gcnet`; the candidate
uses `--backbone-type osram --osram-predictor-mode structured`.

## OSRAM representation

For each utterance, the observed-set encoder supplies a fused node `e_t` and
three 256-dimensional Student slots `z_t^A,z_t^T,z_t^V`.  OSRAM produces:

- `Local`: a node-local projection of `e_t`;
- `Base`: an always-on conversational context;
- `Gap-A/Gap-T/Gap-V`: target-specific contexts, hard-masked by the actual
  missingness mask.

Each direction scans one conversation with a shared parameterization and an
independent memory state.  The forward and backward states are concatenated.
Keys include normalized slot content, node content, availability, and speaker
condition; values use only normalized slot content.  Speaker enters keys and
queries, never values.  The base query is always read, while each gap query is
read through the observed-address residual operator and retained only when its
target is absent.

The memory update is a single A/T/V block update, not sequential A-then-T-then-V
writes:

```text
M_tilde = alpha_h M_prev
R_t^m   = (I - K_obs (lambda_r I + K_obs^T K_obs)^(-1) K_obs^T) q_t^m
M_new   = M_tilde + (Vbar - M_tilde Kbar)
                    (lambda_w I + Kbar^T Kbar)^(-1) Kbar^T
```

Both linear systems use `torch.linalg.solve`; no explicit inverse is used.
The binary availability mask is factored outside the square root in the write
strength so absent slots have finite gradients.

The fixed emotion slots are `[Local; Base; Gap-A; Gap-T; Gap-V]`, with each gap
slot multiplied by `1 - availability`.  They are projected to the 500-dimensional
GCNet hidden interface.  The structured missing predictor receives the target
specific `[Base; Gap-target]` context and leaves the existing MMoE unchanged.

## Complete and missing conditions

With complete A/T/V input, all three gap slots are exactly zero but the base
context remains active.  With TV, only Gap-A is retained; with T-only, Gap-A and
Gap-V are retained.  Padding rows neither read nor write memory and emit a zero
hidden state.  Conversation boundaries reset both directional memories.

## Numerical correction

The first smoke test exposed a NaN gradient in `beta_logits` caused by taking
`sqrt(availability * beta)` at zero availability.  The implementation now uses
`availability * sqrt(beta)`, which is algebraically identical for the binary
mask and has a finite derivative.  The regression test
`test_osram_block_write_absent_slots_have_finite_gradients` locks this behavior.
