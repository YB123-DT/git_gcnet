# Source-role tokens redo

INTERNAL TEST-ORACLE SCREEN

Status: PLANNED — runner integration prepared; no new training scores claimed.

This controlled redo keeps Base/Gap-A/Gap-T/Gap-V × historical Value-source
A/T/V as 12 independent 128-dimensional tokens before Attention. It removes
the previous early active-Gap mean. The previous mean followed the original
user-provided formula; this is a design revision, not a correction of an
unfaithful implementation of that formula.

The same original Flat pre-norm zero-initialized residual connection is retained
explicitly to keep the final readout interface fixed. Token encoding and parameter
count also change; this is not a one-factor causal test of mean pooling. CHM updates, original Query/K/V,
read-before-write, Local Skip, task head/loss and GCNet missing stay unchanged.

| Dataset | Seed | Budget | New W-F1 mean | New high W-F1 | Status |
|---|---:|---:|---:|---:|---|
| MOSI | 66 | 150 epochs | N/A | N/A | pending verification/deployment |
| MOSEI | 66 | 60 epochs | N/A | N/A | pending verification/deployment |

Configuration and reproduction command: [README.md](README.md).
Immutable source SHA, new parameter count, GPU smoke cost and actual run location
must be filled from verification and launch evidence, not inferred in advance.
Model correctness and successful smoke do not establish performance improvement.
