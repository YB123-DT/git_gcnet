# Temporal/Speaker relation-order audit

## Scope

This is the Stage 0 gate from
`GCNet_backbone_replacement_research_design_20260905.md`. It is training-free
and reads only train/validation conversations. No ORTB or other replacement
backbone was trained. The future backbone experiment will keep the selected
**cyclic** schedule, but cyclic is not itself used by this audit because no
optimizer step is performed.

The probe uses the existing `windowp=2`, `windowf=2` topology and frozen
utterance features. Each node feature is deterministically projected to 128
dimensions. The first-order probe receives `[X, T(X), S(X)]`; the ordered
probe additionally receives `[S(T(X)), T(S(X))]`. A fixed ridge regularizer
(`alpha=10`) is used for both probes. Test data is not iterated.

## Structural commutator

Typed adjacency operators are receiver-row/sender-column matrices. For each
conversation and every nonempty Temporal-type/Speaker-type pair, we compute

\[
C_{r,s}=\frac{\|A_T^rA_S^s-A_S^sA_T^r\|_F}
{\|A_T^rA_S^s\|_F+\|A_S^sA_T^r\|_F+\epsilon}.
\]

The observed pair mean is nonzero in both datasets, so the relation matrices
do not commute exactly. However, the relation-label shuffle baseline is
larger, meaning that nonzero algebraic non-commutativity alone is not evidence
that the original labels carry useful order information.

| Dataset | Split | Observed typed-pair mean | Shuffle typed-pair mean | Observed median | Fixed-weight family score |
|---|---|---:|---:|---:|---:|
| MOSI | train | 0.1109 | 0.4099 | 0.1663 | 0.2967 |
| MOSI | validation | 0.1153 | 0.4088 | 0.1730 | 0.3055 |
| IEMOCAP-6 | train | 0.4106 | 0.6141 | 0.4106 | 0.7607 |
| IEMOCAP-6 | validation | 0.4068 | 0.6140 | 0.4068 | 0.7607 |

The typed-pair values are independent of missing rate because the graph
topology and speaker sequence are unchanged; the audit nevertheless repeats
the feature probe at `η=0.0`, `0.5`, and `0.7`.

## Frozen-feature order probe

### CMU-MOSI (validation correlation; lower MAE is better)

| Rate | First-order correlation | Ordered correlation | Delta | First-order MAE | Ordered MAE | Delta MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.3469 | 0.2259 | -0.1210 | 1.3002 | 1.6306 | +0.3304 |
| 0.5 | 0.0977 | 0.0696 | -0.0282 | 1.5247 | 1.6867 | +0.1620 |
| 0.7 | 0.2024 | 0.1638 | -0.0386 | 1.4213 | 1.6177 | +0.1947 |

The rate-0.5 values are means over seeds 66/67/68; the rate-0.0 values are
identical because complete features do not depend on the mask seed. Ordered
features do not improve any of the three-seed MOSI mean metrics.

### IEMOCAP-6 (validation macro-F1; higher is better)

| Rate | First-order macro-F1 | Ordered macro-F1 | Delta | First-order accuracy | Ordered accuracy | Delta accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.2613 | 0.2727 | +0.0114 | 0.2933 | 0.2957 | +0.0025 |
| 0.5 | 0.2224 | 0.2237 | +0.0013 | 0.2740 | 0.2686 | -0.0086 |
| 0.7 | 0.2206 | 0.2115 | -0.0091 | 0.2709 | 0.2561 | -0.0148 |

The rate-0.0 gain is identical across the three seeds; at high missingness the
ordered probe is consistently worse (all 3/3 seeds negative at `η=0.7`).

## Gate decision

The audit fails the required Primary gate for a missing-modality backbone:

- the observed typed commutator is nonzero, but weaker than the label-shuffle
  baseline in both datasets;
- MOSI ordered features are worse at all three tested rates;
- IEMOCAP-6 has a small complete-input gain, but no high-missing-rate gain and
  a consistent loss at `η=0.7`;
- there is no stable cross-dataset, cross-missing-rate order signal that would
  justify adding ordered relation paths.

Therefore **do not implement or train ORTB under this evidence**. Keep cyclic
as the mixed-rate policy for subsequent experiments, but close this particular
ordered-relation replacement route rather than spending a full 5-seed×8-rate
training budget on it. A future backbone proposal needs a different Stage 0
mechanism audit first.

## Reproducibility

- Script: `scripts/audit_relation_order.py`
- MOSI artifact: `mosi.json`
- IEMOCAP-6 artifact: `iemocap6.json`
- Probe seeds: 66, 67, 68
- Rates: 0.0, 0.5, 0.7
- Test loader was constructed by the shared loader factory but never iterated;
  no test labels, predictions, or test metrics enter this report.

