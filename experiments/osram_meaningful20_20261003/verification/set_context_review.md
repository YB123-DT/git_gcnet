# Independent set/context review

Date: 2026-10-03. Reviewer lane: `decision_spec_review`.
Recommendation: **APPROVE for the specified FP32 core protocol**, with one
non-blocking MEDIUM mixed-precision compatibility finding. No HIGH/CRITICAL or
source-core mismatch found. This does not approve a training launch or claim
task-performance improvement.

Scope inspected: `meaningful_blocks_set_context.py`, its dedicated tests,
`meaningful_blocks_common.py`, the accepted research cards and implementation
audit. Only this review document was written. Code-review and
verification-before-completion skills guided the direct review; no recursive
delegation, code edits, Git operation or GPU training was performed.

## Finding

**MEDIUM, non-blocking for current FP32 protocol — mixed-precision output scatter**

At `gcnet_missing_m3/meaningful_blocks_set_context.py:44`, the destination is
allocated using `local.dtype`, while `_forward_group` can return the autocast
dtype. A CPU bfloat16-autocast call with float32 inputs reproducibly fails for
all three IDs with an index-put source/destination dtype mismatch. Cast the
returned group feature to the destination dtype, or establish an explicit
autocast-disabled policy, and add a regression before enabling AMP. The current
`train_gcnet.py` contains no `autocast`, `GradScaler`, `fp16` or `bfloat16` usage;
the approved protocol therefore does not currently exercise this path.

Minimal reproduction (after normal model/input construction):

```python
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    model(local_float32, evidence_float32, active, availability)
```

No implementation changes were made by this reviewer.

## Source-core conclusions

### Perceiver IO — PASS

The [primary paper, section 3](https://arxiv.org/pdf/2107.14795) and
[pinned official implementation](https://github.com/google-deepmind/deepmind-research/blob/9176a9f23ced8e3d6024718d757be8d67cfb6927/perceiver/perceiver.py)
were opened directly. Official `CrossAttention` retains its feedforward residual
whether query residual is enabled or disabled; `BasicDecoder` defaults
`use_query_residual=False`. Encoder and processor include their residual MLPs.

Local lines 71–120 preserve the full encode → three independent latent processors
→ Local-conditioned decode chain, including both normalizations and FFN in each
cross block. Encoder query residual is enabled, decoder query residual disabled.
Attention scales by head width. Width128, eight latents, four heads, three
processors and task feature output are explicit adaptations, not default-source
hyperparameter claims. The source allows omission of its final task projection;
the outer Flat bridge owns the target feature mapping here.

### DGCNN — PASS, self-tie semantics clarified

The [primary paper, section 3.2](https://arxiv.org/pdf/1801.07829) and
[pinned author classification code](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py)
were opened directly. Author `knn` does not remove self or force a separate
self-neighbor; four calls to `get_graph_feature` use each stage's current features.

Local lines 123–171 retain four graph recomputations, center-relative messages,
neighbor maximum, multiscale concatenation and global max/mean. LayerNorm for
BatchNorm and canonical tie-breaking are disclosed adaptations. Self remains
**eligible**, not guaranteed when more than k nodes have exactly identical
features. Independent all-zero example with IDs `[8,2,9,0,4]` and k4 selects
IDs `[0,2,4,8]` for every center; ID9 therefore lacks itself. This is consistent
with the source's candidate-set semantics and the card's deterministic tie rule,
not evidence that self was explicitly excluded. Existing tests cover tied
permutation behavior and all four graph input widths.

### Graph Multiset Transformer — PASS

The [primary paper, equations 6–10 and Appendix B](https://arxiv.org/pdf/2102.11533),
[pinned MAB implementation](https://github.com/JinheonBaek/GMT/blob/a178b590b158b21423ac0dbc3fcf1b41a94fc52a/models/layers.py)
and [pinned full architecture](https://github.com/JinheonBaek/GMT/blob/a178b590b158b21423ac0dbc3fcf1b41a94fc52a/models/nets.py)
were opened directly. The author MAB constructs separate GCN key/value modules,
uses projected-query residual, divides logits by sqrt(total value width), and
uses a single ReLU linear residual with optional two LayerNorms.

Local lines 174–258 preserve two GCN outputs and their concatenation, four-seed
graph pooling, interseed attention and final one-seed identity-graph pooling.
The proposed Local-star/same-head graph counts self loops once and computes
degrees on compact active tokens. Bias follows propagation, consistent with
[the official PyG GCNConv implementation](https://github.com/pyg-team/pytorch_geometric/blob/1.6.0/torch_geometric/nn/conv/gcn_conv.py).
Both graph-pooling K and V depend on normalized adjacency; scale factors are
sqrt256 then sqrt128, not sqrt(head width). The lack of a verified GMT repository
license remains disclosed; this is an independent mathematical adaptation.

## Fresh verification

Environment: `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python`,
PyTorch 2.2.2+cu121, CPU, one intra-op thread.

```text
python -m unittest discover -s tests -p test_meaningful_set_context.py -v
Ran 10 tests in 2.020s — OK

python -m py_compile gcnet_missing_m3/meaningful_blocks_set_context.py tests/test_meaningful_set_context.py
exit 0
```

The suite covers seven availability patterns, inactive NaN/Inf invariance,
poisoned-read backward, all parameter gradients, structural-group updates,
strict reload, unchanged forward RNG/state, repeated evaluation, batch
independence, real eight-head inputs, numerical attention/EdgeConv/GCN checks,
and float64 primitive gradchecks.

Additional independent check: Local256, eight64d heads, seed66, seven supported
availability patterns plus an all-inactive row with NaN Local/evidence; three
Adam steps at 1e-3 on mean-square output plus mean first coordinate. All rows and
all input/parameter gradients stayed finite; inactive outputs and their input
gradients were exact zero after updates. Every parameter tensor changed:

| ID | Parameters | Updated tensors |
| --- | ---: | ---: |
| perceiver_io | 633,472 | 107/107 |
| dgcnn_dynamic_edgeconv | 307,328 | 37/37 |
| graph_multiset_transformer | 598,272 | 64/64 |

Outer first-valid/no-history guards, zero-bridge parity, default-off RNG parity,
OSRAM scan/read hooks, real training/checkpoint selection, GPU resource use and
full training-state resume remain the integration owner's responsibility.
