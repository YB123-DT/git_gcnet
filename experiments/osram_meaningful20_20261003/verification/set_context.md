# Set/context core implementation audit

Date: 2026-10-03. Scope: three accepted family cores only. No formal training,
remote launch, CUDA preflight, checkpoint-selection result or performance gain
is claimed here. The common Flat integration remains a separately verified
responsibility.

## Implemented interface

`build_set_context(method, latent_dim, num_heads, value_dim)` returns a module
with `output_dim=128`. Its forward takes compact `local[N,D]`,
`evidence[N,4,H*V]`, `active[N,4]`, `availability[N,3]`; evidence order is
Base, Gap-A, Gap-T, Gap-V. The common `HeadTokenizer` uses independent
per-head projections shared across evidence roles, no tokenizer normalization,
and genuine role/head IDs. Equal-mask groups are compacted before core work;
all-inactive/empty rows return zero. No OSRAM memory operation or task loss is
defined in this family. The outer wrapper owns first-history/padding exclusion,
RNG-isolated construction and the zero-initialized output bridge.

## Source-to-core audit

### `perceiver_io`

[Paper, section 3](https://arxiv.org/pdf/2107.14795) and
[pinned author implementation](https://github.com/google-deepmind/deepmind-research/blob/9176a9f23ced8e3d6024718d757be8d67cfb6927/perceiver/perceiver.py)
provide the encode/process/decode structure. The implementation retains one
input cross-attention, three independent latent self-attention blocks and one
Local-conditioned cross-attention decoder. All are pre-normalized, include
their feedforward sublayer and use four attention heads. Eight learned latent
vectors have width128; feedforward widening is1. Attention scales by
`sqrt(head_width)`. Encoder query residual is enabled; decoder query residual
is disabled. The current Local query is independently projected from the input
Local token embedding.

Adaptations: smaller fixed widths/depth, role/head evidence input and a
discriminative output feature; no source data-domain preprocessors or extra
task losses. Apache-2.0 source referenced; this is an independent PyTorch
mathematical implementation, not a JAX dependency.

### `dgcnn_dynamic_edgeconv`

[Paper](https://arxiv.org/pdf/1801.07829) and
[pinned author classification backbone](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py)
ground the four recomputed graphs and center-relative EdgeConv messages.
Stage widths are64,64,128,256. Each stage recomputes four nearest neighbors in
its current feature space, including self among eligible candidates; exact
ties use canonical role/head identity. Edge messages use `[neighbor-center,
center]`, then a learned map, per-edge LayerNorm, LeakyReLU and neighbor max.
All four outputs are concatenated before128-dimensional projection and both
global max and mean readout.

Adaptations: evidence-space rather than xyz input, k4 for the small real-head
set, LayerNorm replacing BatchNorm, and the proposed task output. No FPS,
radius graph, invented spatial layout or static-graph substitute. The MIT
author source is referenced; nearest-neighbor indices remain discrete.

### `graph_multiset_transformer`

[Paper, equations6–10](https://arxiv.org/pdf/2102.11533),
[pinned architecture source](https://github.com/JinheonBaek/GMT/blob/a178b590b158b21423ac0dbc3fcf1b41a94fc52a/models/nets.py)
and [pinned attention source](https://github.com/JinheonBaek/GMT/blob/a178b590b158b21423ac0dbc3fcf1b41a94fc52a/models/layers.py)
ground two normalized GCN layers, their concatenated outputs, graph-aware
four-seed pooling, one interseed attention block and final one-seed pooling.
The Local-star/same-head graph inserts self loops once. First-pooling keys and
values come from separate GCN maps; later pooling uses the identity-graph
case. The explicitly selected author numerical convention is preserved:
projected-query residual, `sqrt(total_width)` score scaling and residual
single-layer ReLU feedforward with two LayerNorms. Widths are256 at first
pooling and128 thereafter.

Adaptations: proposed role/head adjacency, small seed counts and source
classifier removal. No software license was verified for the author repo;
no repository source is copied or translated. This implementation is written
independently from mathematical operations, with attribution and no claim of
source benchmark or graph-isomorphism guarantees.

## Executed checks

TDD red: before implementation, eight tests failed with the explicit assertion
`set/context core implementation is missing`. After implementation and added
actual-head/gradient coverage:

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_set_context -v
Ran 10 tests in 2.208s — OK

/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m py_compile gcnet_missing_m3/meaningful_blocks_set_context.py tests/test_meaningful_set_context.py
exit 0
```

Each of the three IDs individually passes all-seven-availability forward,
inactive NaN/Inf invariance, poisoned inactive backward, finite input and
parameter gradients, updates in each intended structural parameter group,
strict state round trip, repeated evaluation, cross-example independence,
unchanged forward RNG and empty/all-inactive guards. Numeric tests check
Perceiver attention scaling/query residual, complete stage counts, four
DGCNN graph computations/tie behavior/edge reduction, GMT graph normalization,
graph-derived K/V/total-width scaling/pooling chain, and smooth float64
primitive gradchecks. Actual Local256 and eight64-dimensional heads are
included in CPU forward/backward coverage.

These tests do not substitute for parent-owned default-off/zero-bridge Flat
parity, actual-protocol CUDA resource measurement, full training-state resume,
or an independent source/code review. No new dependency was added.
