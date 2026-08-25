# Context Snapshot: GCNet Second-Layer Aggregators

Chinese mirror: [second-graph-aggregators-20260825T140248Z.zh.md](second-graph-aggregators-20260825T140248Z.zh.md). English mirror: [second-graph-aggregators-20260825T140248Z.en.md](second-graph-aggregators-20260825T140248Z.en.md).

## Task

Implement and experimentally evaluate GenAgg and scaled Soft Medoid as replacements for both second-layer GCNet GraphConv neighborhood aggregators.

## Desired outcome

Produce source-faithful, Torch 1.8/PyG 2.0.1-compatible implementations, verified code, a locked 12-job IEMOCAPSix fold-5 gate using inherited Original controls, and evidence-based decisions about formal continuation.

## Known facts

- `GraphNetwork.conv2` is PyG `GraphConv(hidden_size, hidden_size)` with default add aggregation.
- Temporal and speaker branches own separate `GraphNetwork` instances.
- Locked graph windows are 2/2, giving at most five incoming edges for an interior utterance, including a self edge.
- GenAgg paper-era code requires APIs absent from the official environment.
- Scaled Soft Medoid must multiply its convex combination by neighborhood cardinality to recover add at high temperature.
- Forty Original task NPZ files already exist locally and on biggpu and must be inherited.
- Correct baseline test command passed 158 tests with one expected skip.
- Local and biggpu `/data2` filesystems are distinct.

## Constraints

- Do not alter first-layer RGCN, topology, relations, recurrent context, branch fusion, reconstruction, classifier, optimizer, or natural mask protocol.
- Do not rerun Original.
- Do not run epoch-level smoke tests.
- Add no dependency.
- Preserve legacy add-path RNG, keys, parameters, outputs, and filenames.
- Use four GPUs with three jobs per GPU for the first 12 candidate jobs.

## Open questions resolved by protocol

- GenAgg uses the paper-era MLP, BatchNorm, Mish, learnable alpha/beta, and inverse loss weight 1.0.
- Soft Medoid uses fixed temperature 1.0 for the first test.
- Only candidates satisfying the predeclared paired gate advance.

## Likely touchpoints

- `gcnet/second_graph_aggregation.py`
- `gcnet/model.py`
- `gcnet/train_gcnet.py`
- `tests/test_second_graph_aggregation.py`
- `tests/test_second_graph_aggregation_integration.py`
- `tests/test_training_protocol.py`
- `experiments/mpfilm_iemocap6/run_locked_ab.py`
- `tests/test_mpfilm_runner.py`
- `experiments/second_graph_aggregation_iemocap6/`

