# Independent graph-family CPU review

Verdict: **APPROVE CPU readiness** for all four IDs, 2026-10-03. No material
implementation/card mismatch or masking/gradient blocker found. This is not
permission to bypass integration, CUDA/resource, immutable-source or dispatch
gates. Reviewer did not implement the graph family and changed no source/tests.

Reviewed: `meaningful_blocks_graph.py`, `test_meaningful_graph.py`, the four
accepted entries in `graph.json`, and root common masking/tokenization helpers.
The EGT card now includes the approved two-full-layers plus node-ended-final
layer amendment, consistent with the implementation.

## Mechanism checks

- RRN: first messages use embedded input nodes, while LSTM hidden/cell start
  at zero. Five rounds share message/post/LSTM parameters; every post stage
  receives original node inputs. Confirmed against the pinned author's
  [bAbI recurrent path](https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/d3a5a27f6e73e51aac1edb9556ac47be3350d748/tasks/babi/rrn.py).
- EGT: dense valid pairs, dot-score clipping before edge bias, sigmoid gates
  after key softmax, log1p gate-degree scaling, pre-softmax edge updates,
  separate residual node/edge FFNs. The two retained edge updates both feed
  subsequent node computation; the last layer has no unused edge-output
  parameters. [Pinned attention source](https://github.com/shamim-hussain/egt_pytorch/blob/9e66956a5fdc6f6e8a865863d029468380bb63e5/lib/models/egt_layers.py).
- Original GatedGCN: endpoint-dependent vector gates multiply transformed
  neighbor values; sums are not normalized by gate mass. Each of three cells
  retains two convolutions and a projected cell-input residual. Per-node LN
  is the documented replacement for source BN, not the later normalized-edge
  variant. [Pinned original notebook](https://github.com/xbresson/spatial_graph_convnets/blob/f1da7d7f91a2ce0f324c43d787de05d0cc47e4cf/01_residual_gated_graph_convnets_subgraph_matching.ipynb).
- PNA: four divided-input towers, shared per-layer edge embedding, distinct
  tower message/update MLPs, actual neighbor-axis mean/std/min/max, all three
  degree scalers and cross-tower mixing, four layers. The fixed seven-mask
  degree reference matches 2.3785469096888603 and excludes test statistics.
  [Pinned tower/message/aggregate source](https://github.com/lukecavabarrett/pna/blob/6867d9aadcbff620542683a9168908c5c1ca6d8f/models/pytorch_geometric/pna.py).

## Independently executed evidence

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_graph.py
Ran 11 tests in 2.764s — OK
```

Tests cover explicit equations, topology, fixed depths, all seven availability
patterns, poisoned inactive values, compact 9/17/25-node execution, inactive
gradients, batch independence, empty/all-inactive rows, strict state reload,
forward RNG preservation and unknown-ID rejection.

Additional read-only CPU probes:

1. Seed461; actual256/8/64 interface; float64; all seven masks. Weighted-square
   output objective gave finite, nonzero gradients in **every named parameter
   tensor** of each method, not just one selected core parameter. No missing
   or zero-gradient group was found.
2. Repeated evaluation was exactly equal and every state-dict tensor remained
   unchanged for all four methods.
3. Seed813; float64 three-node loop-free graph; width8; original configured
   depths. `torch.autograd.gradcheck(..., fast_mode=True, atol=1e-4,
   rtol=1e-3)` passed separately for RRN, EGT, GatedGCN and PNA input gradients.
   These random fixtures are local numerical checks, not a derivative claim
   at min/max ties or activation kinks.

No GPU was probed or launched. Full-model default-off/zero-bridge parity,
first-valid wrapper behavior, original task gradients after bridge warmup,
scan/query counts, and healthy-bigGPU real-shape peak/time profiling remain
the integrator's admission requirements.
