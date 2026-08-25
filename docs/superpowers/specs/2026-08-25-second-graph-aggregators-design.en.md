# GCNet Second-Layer Aggregator Design (English)

This file is the English mirror of [the canonical design](2026-08-25-second-graph-aggregators-design.md). The canonical document contains the complete source trace, formulas, implementation boundaries, verification contract, experiment gate, and synchronization protocol and is authoritative if the mirrors ever diverge.

## Locked decision

- Replace only both second-layer `GraphConv` neighborhood aggregators.
- Evaluate source-faithful GenAgg first and scaled Soft Medoid second.
- Keep first-layer RGCN, topology, relations, BiLSTMs, branch addition, reconstruction, classifier, optimizer, and natural mask protocol unchanged.
- Inherit the existing 40 Original NPZ archives; never retrain Original.
- Run no epoch-level smoke test.

## Mechanisms

GenAgg uses

\[
f^{-1}\left(n^{\alpha-1}\sum_j f(x_j-\beta\mu)\right)
\]

with the paper-era `1-2-2-4` MLP and inverse, Mish, BatchNorm, Kaiming initialization, learnable \(\alpha,\beta\), and explicit inverse-consistency loss. It adds 59 parameters per graph branch.

Scaled Soft Medoid uses

\[
n\sum_j \operatorname{softmax}_j\left(
-\sum_k\lVert m_j-m_k\rVert_2/T
\right)m_j
\]

with fixed `T=1.0`, preserving Original sum in the high-temperature limit and adding no parameters.

## First-wave experiment

Run exactly 12 candidate jobs: `{genagg, soft_medoid}` × missing `{0.0, 0.7}` × seeds `{66,67,68}`. Schedule three jobs on each of four GPUs. Launch the reduced grid under `stage=formal`, because `stage=gate` has different immutable paths and provenance. Reuse these exact formal artifacts if the candidate advances.

The full acceptance and provenance rules are defined in the canonical document.
