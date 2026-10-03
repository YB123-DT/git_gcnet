# Graph block implementation verification

Date: 2026-10-03. Scope: four independent PyTorch readout processors, their
unit tests, this audit, and the explicitly approved EGT correction in graph.json.
No source implementation was copied; no dependencies, OSRAM operations, losses,
feature targets, history caches, training runs, or Git operations were added.

## Implemented cores

| Candidate | Local implementation | Preserved processing mechanism |
|---|---|---|
| `rrn_evidence` | `RRNCore` | Five shared pair-message/sum/input-reinjection/LSTM steps |
| `egt_evidence` | `EGTCore`, `EGTLayer` | Two complete node/edge layers, then author-supported node-ended final layer; global gated attention, dynamic centrality, residual FFNs |
| `residual_gated_graph_evidence` | `GatedGraphCore`, `GatedGraphCell` | Three complete two-convolution residual cells; endpoint-derived vector gates and transformed neighbor sums |
| `pna_evidence` | `PNACore`, `PNALayer` | Four full layers, four towers, endpoint/edge messages, mean/std/min/max times identity/amplification/attenuation, post-transform and tower mixing |

API: `build_graph(method, latent_dim, num_heads, value_dim)` returns a module
with `output_dim=128`. Forward accepts Local `[N,D]`, four evidence roles
`[N,4,H*V]`, active `[N,4]`, availability `[N,3]`, and returns `[N,128]`.

The common tokenizer is reused with per-head projections and `normalize=True`.
Head indices are the actual producer layout; canonical roles are Local, Base,
Gap-A, Gap-T, Gap-V. Static undirected relations (represented in both directions)
connect Local to all heads, same-role heads, and same-index cross-role heads.
They contain no labels, estimated reliability, artificial geometry, or history
timestamps. EGT keeps all active ordered pairs, including self; the sparse
relations only initialize its pair features.

Active-pattern grouping removes inactive heads before the core: actual H=8
produces 9/17/25 nodes, not a dense 33-node graph with inactive work. Typed
five-role means, a 640-to-128 projection, and LayerNorm provide the shared
readout. The root-owned wrapper is responsible for the common zero-start Flat
residual bridge; this module does not alter Flat.

## Source trace and explicit adaptations

- RRN: [paper, section 2](https://arxiv.org/abs/1711.08028);
  [author message passing](https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/d3a5a27f6e73e51aac1edb9556ac47be3350d748/message_passing.py#L4-L31),
  [author recurrent steps](https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/d3a5a27f6e73e51aac1edb9556ac47be3350d748/tasks/babi/rrn.py#L119-L142).
  Original input is reinjected at every step; the first message pass uses input
  node states while LSTM state starts at zero. Source per-step supervised losses
  are not transferred. No license was found in the author repository; implementation
  here was independently derived from the mathematical mechanism.
- EGT: [paper, sections 3.2-3.3](https://arxiv.org/abs/2108.03348);
  [author layer](https://github.com/shamim-hussain/egt_pytorch/blob/9e66956a5fdc6f6e8a865863d029468380bb63e5/lib/models/egt_layers.py#L25-L238),
  [author node-ended configuration](https://github.com/shamim-hussain/egt_pytorch/blob/9e66956a5fdc6f6e8a865863d029468380bb63e5/lib/models/egt.py#L94-L101), MIT.
  The initially accepted card updated edges in all three layers but pooled only
  final nodes. Consequently, final edge-update/FFN parameters would have no loss
  path. Before any training the leader approved two full joint layers followed by
  `edge_update=False` in the final layer. No extra edge pooling was introduced.
  Node width128, edge width32, eight attention heads, clipping[-5,5], ELU FFNs
  with expansion2 and dynamic centrality are retained. No positional SVD,
  structural objective, virtual node, or stochastic attention regularization.
- Residual GatedGCN: [paper, equations 9,11,12](https://arxiv.org/abs/1711.07553);
  [author notebook](https://github.com/xbresson/spatial_graph_convnets/blob/f1da7d7f91a2ce0f324c43d787de05d0cc47e4cf/01_residual_gated_graph_convnets_subgraph_matching.ipynb),
  raw JSON119-212, code cell3 `OurConvNetcell`, MIT. Original unnormalized
  neighbor sums and recomputed vector gates are retained; no later
  benchmarking-gnns edge-state variant is silently substituted. BatchNorm is
  replaced by per-node LayerNorm to prevent cross-utterance/statistical coupling.
- PNA: [paper, equations 5-8](https://arxiv.org/abs/2004.05718);
  [author PNAConv](https://github.com/lukecavabarrett/pna/blob/6867d9aadcbff620542683a9168908c5c1ca6d8f/models/pytorch_geometric/pna.py#L56-L159),
  [statistics](https://github.com/lukecavabarrett/pna/blob/6867d9aadcbff620542683a9168908c5c1ca6d8f/models/pytorch_geometric/aggregators.py#L13-L32),
  [degree scalers](https://github.com/lukecavabarrett/pna/blob/6867d9aadcbff620542683a9168908c5c1ca6d8f/models/pytorch_geometric/scalers.py#L8-L19), MIT.
  The fixed, label-free degree reference is the node-weighted average over all
  seven valid availability patterns: delta=2.3785469096888603 for H=8. This
  explicitly replaces the source's empirical training-degree normalization.
  Per-node LayerNorm/ReLU follows each full layer. Neighbor reductions use the
  source-neighbor axis, population std and epsilon1e-5; half-precision statistics
  accumulate in FP32. Source task heads, GRU wrapper and multi-task objectives
  are not part of the transferred PNA core.

These are new-task core-block adaptations, not source task reproductions or
claims of accuracy gains. Mathematical details and paper metadata remain in
`../graph.json`.

## TDD and executed checks

Working directory: `/data2/yb/paper/GCNet_TPAMI/.worktrees/osram-reg-only`.

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_graph.py -v
```

1. Before production code: **9 failures**, each explicitly asserting that the
   graph implementation was unavailable. No tests passed silently against old code.
2. Initial four-core implementation: **9 tests passed**.
3. Active-graph packing regression was added: **4 subtest failures**, each
   observing `(33, False)` rather than active-only `(9, True), (17, True),
   `(25, True)` core calls. The independent complete PNA layer oracle was also added.
4. Active grouping implemented; complete suite: **11 tests passed**.
5. Real H8/V64 backward assertions added; rerun: **11 tests passed**, exit0,
   2.572 seconds for unittest's reported test duration.

Additional commands executed successfully:

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m py_compile gcnet_missing_m3/meaningful_blocks_graph.py tests/test_meaningful_graph.py
jq -e '.candidates | length == 4' experiments/osram_meaningful20_20261003/graph.json
```

Coverage includes independent tiny equations for RRN, EGT node/edge updates,
both gated convolutions, PNA statistics and full towers; real head relation
topology/degrees; all seven observed-modality masks; inactive NaN/Inf poisoning;
availability overriding overpermissive active flags; exactly zero inactive
input gradients; finite nonzero core gradients and an SGD parameter update;
batch independence including train-mode behavior; empty/all-inactive safety;
no forward RNG consumption; and strict state-dict roundtrip.

## Remaining verification boundary

Only CPU module-level tests and syntax compilation were run in this lane.
No CUDA test, training, final task performance, throughput, GPU peak memory,
full-model integration, linter, or project type-checker result is claimed here.
The root lane owns integration, GPU checks, experiment scheduling and external
independent review. Original OSRAM, data, objective and checkpoint selection
were not edited.
