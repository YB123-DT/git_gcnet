# Additional GCNet Graph Candidate Decision (English Mirror)

The canonical decision is [here](2026-08-25-additional-graph-candidates-design.md).

- Implement SSMA Conv2 as the source-grounded NeurIPS 2024 candidate.
- Reject the unsupported “Relation-Track MMP” and pure late-fusion labels: linear delayed summation is identical to Original when all two-hop transitions are retained. Evaluate only the explicitly custom Relation-Track Diagonal Routing (RTDR) transition-mask hypothesis.
- Reject naive Ego–Neighbor Separation as algebraically redundant with the existing GraphConv neighbor/root parameterization.
- Reject direct Centered Clipping because its historical center, detached optimizer state, threshold scale, and Byzantine-minority assumptions do not transfer to degree-3–5 transient conversation nodes.
- Screen GenAgg, Soft Medoid, SSMA, and RTDR on the same formal-compatible six-task subset per candidate without rerunning Original.
