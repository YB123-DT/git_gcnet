# Context Snapshot: GCNet Second-Layer Aggregators

Chinese mirror: [second-graph-aggregators-20260825T140248Z.zh.md](second-graph-aggregators-20260825T140248Z.zh.md). English mirror: [second-graph-aggregators-20260825T140248Z.en.md](second-graph-aggregators-20260825T140248Z.en.md).

## Task

Implement and experimentally evaluate GenAgg, scaled Soft Medoid, and SSMA at GCNet's second graph layer, plus a clearly labeled custom RTDR relation-transition-routing hypothesis. Record mathematical rejection evidence for pure linear RTLF, redundant Ego–Neighbor Separation, and non-transferable Centered Clipping.

## Desired outcome

Produce Torch 1.8/PyG 2.0.1-compatible implementations, verified code, two locked 12-job IEMOCAPSix fold-5 phases using inherited Original controls (24 candidate tasks total), and evidence-based decisions about formal continuation.

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

- GenAgg, Soft Medoid, and SSMA alter only both `conv2` aggregators. RTDR may decompose the existing first-layer RGCN messages but must reuse its exact weights, normalization, topology, and relations; all recurrent context, branch fusion, reconstruction, classifier, optimizer, and natural mask protocol remain unchanged.
- Do not rerun Original.
- Do not falsely attribute custom RTDR to MrMP or describe it as pure late fusion, and do not spend training budget on algebraically redundant or source-incompatible candidates.
- Do not run epoch-level smoke tests.
- Add no dependency.
- Preserve legacy add-path RNG, keys, parameters, outputs, and filenames.
- Use four GPUs with three jobs per GPU: Phase A has 12 GenAgg/Soft Medoid jobs and Phase B has 12 SSMA/RTDR jobs.
- After verified completion, synchronize results locally and automatically commit/push the completed code and results to `YB123-DT/git_gcnet`; do not publish incomplete jobs as completed versions.

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
