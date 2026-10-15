# Isolated compatibility and correctness work

1. Copy exact GCNet and SDR-GNN source snapshots under this directory; never change active source repositories.
2. Record existing defects before patching: speaker limit, nondeterministic relation enumeration, SDR tuple concatenation dropping speaker branch, attention batch/sequence inversion, fixed feature width, and runtime hypergraph propagation.
3. Preserve all recurrent, relation, high-order, hypergraph, reconstruction and classification modules. Extend speaker relation count to N*N; preserve future windows and bidirectional recurrence.
4. Regression-check nine speakers, both branches' classification gradients, finite forward/backward, hypergraph execution, attention axes and graph batch isolation on CPU. Report any padding/batch dependence inherited from the original recurrence.
5. Use explicit SDR-GNN Implementation-fixed naming, provenance and patch files. No formal GPU training here.
