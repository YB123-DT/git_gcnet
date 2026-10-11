# Flat History Recency Metadata Design

## Goal

Test whether strictly causal modality-history metadata gives the original cfg84
Flat readout useful information beyond Local, Base, and active Gap values.

## Definition

For valid utterance `t` and modality `m`, compute only from indices `i < t`:

```text
c[t,m] = 1 if any valid i<t has availability[i,m]=1, else 0
d[t,m] = t-last_seen[m] when c[t,m]=1, else 0
R[t]   = [c_A,c_T,c_V,d_A,d_T,d_V]
```

The pair `(c=0,d=0)` is the dedicated never-seen state. A modality observed at
the immediately preceding utterance has `(c=1,d=1)`, so it is unambiguous.
Distances are raw utterance intervals and are not normalized or transformed.
Padding produces six zeros and never updates history. The current utterance is
incorporated only after `R[t]` has been emitted.

## Integration

Add one opt-in cfg84 switch, default false. When enabled, concatenate `R_t` to
the existing Flat adapter input:

```text
[Local, Base, masked Gap-A/T/V, R] -> original Flat adapter topology
```

The Local skip, emotion normalization, classifier, CHM scan, read-before-write,
joint Block Write, Query, Base, Gap, mask schedule, objective, and task head are
unchanged. The method is mutually exclusive with existing experimental readout
modules so that the configuration differs from cfg84 only by this metadata.

## Verification and experiment

Tests must prove strict-past semantics, never-seen and padding behavior, default
path equality, configuration exclusivity, finite gradients, and that changing
only current availability cannot alter current `R`. Run a real MOSI batch smoke,
then one seed-66, 150-epoch per-rate Test-oracle internal screen on the same
healthy biggpu GPU0 already hosting the two evidence-readout screens. Use an
immutable source snapshot and an independent result directory.

