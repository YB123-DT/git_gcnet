# Hypergraph family implementation verification

Status: four CPU-tested implementations; independent source audit, integrated
Flat/scan/RNG parity and admitted biggpu CUDA preflight remain required before
training. No experiment was launched by this lane.

## Scope and fixed mechanisms

Owned files are `gcnet_missing_m3/meaningful_blocks_hypergraph.py`,
`tests/test_meaningful_hypergraph.py`, and this record. The root-owned
`meaningful_blocks_common.HeadTokenizer` supplies genuine head projections,
type/head embeddings and safe masking, with `shared_projection=False` and
`normalize=False` as the accepted hypergraph card specifies.

Every processed utterance has 9/17/25 nodes and 9/10/11 hyperedges: eight
Local-anchored same-head groups, one Local-anchored Base group, and one group
per active Gap modality. Input role order is Local, Base, Gap-A, Gap-T, Gap-V.
All quantities are ephemeral. There are no memory calls, new losses, dropout,
batch statistics or across-utterance graph edges. Padding/no-history exclusion
and zero-output injection belong to the outer wrapper; standalone all-inactive
rows and empty batches return zeros without invoking a graph core.

| ID | Complete core | Parameters including tokenizer/readout, excluding outer bridge |
| --- | --- | ---: |
| allset_transformer | Two complete V→E→V PMA rounds; independent four-head pooling in each direction, learned seeds, grouped softmax, seed/FFN residuals and LayerNorm | 417152 |
| ed_hnn | Three shared sum/recipient-conditioned-return/restart/MLP iterations, alpha 0.1 | 282368 |
| hyper_sagnn | One four-head leave-self-out dynamic encoder, separate static tanh FFN, separate normalization, squared discrepancy, head/type edge pooling | 283776 |
| sheaf_hypergnn_diag | Two dynamically predicted diagonal incidence-map layers; stalk4×channel32, normalized lifted transport, learned stalk/feature transforms; ELU after first only | 153064 |

Counts use `latent_dim=256,num_heads=8,value_dim=64`; output dimension is128.
Other than Hyper-SAGNN, readout is updated Local, mean Base and mean active Gap,
concatenated and projected384→128. Hyper-SAGNN instead concatenates original
projected Local, mean same-head edge discrepancy and mean evidence-type edge
discrepancy, as its card requires.

## Source-specific decisions for independent review

- AllSet follows pinned `src/layers.py:PMA`: seed-key LeakyReLU0.2 scoring,
  value pooling plus seed, LN, residual ReLU(two-layer FFN), LN. ReLU follows
  each half-round. No generic Q/K/V self-attention substitution.
  [Author source](https://github.com/jianhao2016/AllSet/blob/6281a2f1a91f6f26040777bb0b2578fc035dc57a/src/layers.py).
- ED-HNN retains recipient features in each edge-to-node message, two sum
  reductions and restart to initial ReLU-projected nodes. One diffusion module
  is reused for all three rounds.
  [Author source](https://github.com/Graph-COM/ED-HNN/blob/fea3b8f5f11c2eb7265dd4d880e7b5796eddf958/models/edgnn.py).
- Hyper-SAGNN's actual `EncoderLayer.forward` sends original static inputs to
  `pff_n2`, although attention computes an unused `fc2` static projection.
  The implementation keeps the effective original-input static path and omits
  that dead projection. Both position-wise FFNs use tanh; only the dynamic FFN
  has a residual. Separate classifier-stage final LNs are retained. The source
  scalar sigmoid classifier is explicitly replaced by vector discrepancy
  pooling. Softmax uses explicit self exclusion on compact, size≥2 edges;
  the source's approximately normalized epsilon-denominator softmax is not
  copied. No self entry or inactive occurrence contributes.
  [Author source](https://github.com/ma-compbio/Hyper-SAGNN/blob/69f2fbe21c455aca084497fb2d26a8207a95decd/Code/Modules.py).
- Sheaf follows the accepted **author-code** operator convention:
  `A=D^(-1/2) H B^(-1) H.T D^(-1/2)` and
  `P=D^(-1/2)+A-2*node_blockdiag(A)`. The leading term is not identity:
  source first pre-normalizes the input before its `I+Q-2*blockdiag(Q)`
  operation. It does not silently substitute another I−L
  convention or claim the paper's energy theorem. Diagonal map prediction is
  `tanh(Linear64→4([node_stalk_mean,edge_mean]))`; D is clamped at1e-6.
  Learned stalk/feature maps are bias-free as in the card's matrix equation,
  followed by the explicit channel bias. Source input-head embeddings and
  output head are adapted, and source persistent shape/hyperedge caches are
  absent. Independent dense equations were written; no unlicensed source was
  vendored. [Author operator](https://github.com/IuliaDuta/sheaf_HNN/blob/45a5ebc16ec4b8e865431f9035eefc7719e7e7dc/layers.py),
  [map builder](https://github.com/IuliaDuta/sheaf_HNN/blob/45a5ebc16ec4b8e865431f9035eefc7719e7e7dc/sheaf_builder.py).

## TDD and observed verification

The initial seven tests were written before the implementation. Running them
gave seven assertion failures stating that the hypergraph family did not yet
exist. The ED recipient-conditioned equation test passed first, followed by
the complete seven-test suite. Three additional coverage tests and nonzero
parameter-gradient assertions were then added.

Command (2026-10-03):

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_hypergraph.py
Ran 10 tests in 1.700s — OK
```

The ten tests verify exact graph membership; independent PMA/sum-message/
lifted-sheaf/reference-attention equations; all seven supported availability
patterns plus all-inactive rows; inactive NaN and all-inactive Inf isolation;
finite gradients including every expected parameter tensor having a nonzero
gradient; zero inactive-evidence gradients; optimizer changes; batch
independence; zero/large finite inputs; no eval state/RNG mutation; strict
checkpoint reload; empty batches; metadata-preserving graph permutations;
CPU float64 input gradcheck for all four differentiable primitives; and fixed
source-specific module/iteration counts. LayerNorm/ReLU fixtures are away from
nondifferentiable boundaries; no broad differentiability claim is made there.

Not yet verified here: real-shape CUDA memory/time, original task gradient
integration after the outer zero bridge warms up, upstream scan/query parity,
first-valid-row wrapper guards, default-off historical checkpoint parity,
independent review or performance. Those remain the common admission gates.

## Independent-review correction

The independent drift reviewer found that the initial implementation and card
incorrectly combined source input pre-normalization into an identity leading
term. Three source-order regression assertions were observed failing before
the correction: the explicit two-node/stalk1/maps2/Z=[1,3] example (correct
output [1.5,.5], previous [2,2]); the lifted diagonal fixture; and a source-order
reference that separately forms `X_pre=D^(-1/2) Z` and
`Q=D^(-1/2) H B^(-1) H.T`. The corrected implementation uses
`P=D^(-1/2)+A-2*blockdiag(A)` and the source-order reference also checks
gradients with respect to both restriction maps and input features.
The historical ten-test result above did not validate that omitted factor;
the corrected suite and independent rereview supersede its sheaf conclusion.

Corrected-suite rerun: same unittest command, **12 tests in1.918s — OK**.
Independent rereview was requested from the drift reviewer; approval remains
pending rather than inferred from this self-verification.
