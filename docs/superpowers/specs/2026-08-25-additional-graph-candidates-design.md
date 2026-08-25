# Additional GCNet Graph Candidate Decision

Chinese translation: [2026-08-25-additional-graph-candidates-design.zh.md](2026-08-25-additional-graph-candidates-design.zh.md). English mirror: [2026-08-25-additional-graph-candidates-design.en.md](2026-08-25-additional-graph-candidates-design.en.md).

## Decision table

| Proposed name | Verified source/mechanism | Decision | Reason |
|---|---|---|---|
| SSMA Conv2 | Sequential Signal Mixing Aggregation, NeurIPS 2024 | Implement as a formal candidate | It introduces explicit cross-neighbor products that sum aggregation cannot represent before compression. |
| Relation-Track MMP | No matching source; MrMP mixes relations inside every layer | Reject the name and pure late-fusion claim; evaluate custom RTDR only | Linear delayed summation is identical to Original; a nontrivial zero-parameter test must explicitly alter the two-hop relation-transition mask. |
| Ego–Neighbor Separation | H2GCN D1, NeurIPS 2020 | Reject as a formal replacement; retain mathematical redundancy evidence | GCNet GraphConv already has independent neighbor and root transforms; immediate concat-linear is the same function class. |
| Centered Clipping | Byzantine-robust federated optimization, ICML 2021 | Reject/park | A faithful transfer leaves the center initialization, iteration count, threshold scale, and good-vs-outlier semantics ungrounded; the theorem also assumes at most 10% Byzantine inputs while GCNet has degree 3–5. |

## SSMA Conv2

Primary source: Mitchell Keren Taraday, Almog David, and Chaim Baskin, “Sequential Signal Mixing Aggregation for Message Passing Graph Neural Networks,” NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2409.19414), [official code at inspected commit](https://github.com/AlmogDavid/SSMA/tree/9d128c902acf47343b5baf5150a78dfb6a64fb3e). The repository README says MIT, but no root LICENSE or GitHub license metadata was found; implementation must be independently derived from the paper equations.

For raw incoming neighbor states \(m_j\in\mathbb R^d\), lock \(\kappa=5\), \(m_1=6\), \(m_2=5(d-1)+1\). Construct the fixed coefficient signal without a dense affine parameter:

\[
\Phi(m_j)[0,0:d]=-m_j,\qquad \Phi(m_j)[1,0]=1.
\]

All other coefficients are zero. The normalized signal mixing is

\[
Z_j=\operatorname{FFT2}(\Phi(m_j)),
\quad A=\exp\left(\operatorname{mean}_j\log(|Z_j|+10^{-6})\right),
\quad \Theta=\sum_j\arg Z_j,
\]

\[
C=\Re\operatorname{IFFT2}(\operatorname{polar}(A,\Theta)),
\qquad a_i=W_c\operatorname{vec}(C)+b_c.
\]

The GCNet adaptation installs the official core `use_attention=False` SSMA variant as the `GraphConv` aggregation primitive: it mixes raw incoming states first, compresses with the paper base full compressor `Linear(2976,100)`, and only then applies the existing `lin_l` transform. The existing `lin_r(x_i)` is added once. This matches PyG's `GraphConv(aggr=...)` order and changes aggregation rather than silently changing the message transform. It keeps every original edge, including the explicit self edge, and adds no attention or sampling. Empty target neighborhoods are explicitly zeroed after compression. It changes both `conv2` instances only. The formal model adds 297,700 parameters per branch and 595,400 total. Extra initialization is RNG-forked so every shared GCNet parameter remains identical to Original.

The official code requires newer aggregation hooks. The compatibility implementation uses only Torch 1.8 `torch.fft`, `torch.polar`, and torch-scatter. Official-environment complex backward is a mandatory one-time gate before training.

Required proofs include explicit circular-convolution equivalence on a tiny signal, permutation invariance, nonzero cross-neighbor mixed derivative versus zero for add, degree cap validation, zero-degree root-plus-bias, exact parameter delta, finite gradients, and both-branch-only integration.

## Relation-Track Diagonal Routing (RTDR)

The exact acronym candidate, “Multi-Relation Message Passing for Multi-Label Text Classification” (ICASSP 2022, [arXiv](https://arxiv.org/abs/2202.04844)), explicitly sums relation messages within every layer. It therefore cannot anchor a method whose purpose is to delay relation mixing. CompGCN has the same early summation. SeHGNN preserves semantic/metapath vectors but replaces message passing with precomputation and semantic attention, so it is not an exact anchor either.

Pure delayed relation fusion is not a valid nontrivial candidate. Let \(\mu^r\) be the first-layer relation message and \(A_q\) the second-hop aggregation operator for relation \(q\). With tied linear `GraphConv` weights and every transition retained,

\[
\sum_q A_q\sum_r\mu^r=\sum_{q,r}A_q\mu^r.
\]

Therefore “mix early” and “mix late” commute exactly. Any observed difference would necessarily come from an unstated nonlinearity, new parameters, or changed paths.

The only zero-parameter, single-variable version retained for evaluation is named **custom Relation-Track Diagonal Routing (RTDR)**, not MMP and not a paper transfer. It explicitly compares a full two-hop relation-transition matrix with a diagonal transition mask. Here \(A_q\) is the exact unnormalized add-scatter over second-hop edges whose `edge_type` is \(q\), including that relation's assigned self edges, and \(A=\sum_q A_q\). Further, \(c=xW_{root}+b_1\) is the shared first-layer root/bias track, \(z=c+\sum_r\mu^r\), and \(L,R,b_2\) are the existing `conv2` neighbor transform, root transform, and bias:

\[
y_i^{\mathrm{original}}
=L\!\left(Ac+\sum_{q,r}A_q\mu^r\right)+Rz_i+b_2,
\qquad
y_i^{\mathrm{RTDR}}
=L\!\left(Ac+\sum_r A_r\mu^r\right)+Rz_i+b_2.
\]

The common track uses every original edge, `conv2` root uses the complete \(z\) once, and `conv2` bias is added once. Only off-diagonal two-hop transitions \(q\ne r\) are removed; topology, relation definitions, and all weights remain unchanged.

The required controls are:

1. `early` executes the untouched Original path and must be bit-exact in forward, backward, RNG, parameter keys, and parameter count.
2. A `full-transition` decomposition must reproduce Original within forward tolerance \(10^{-6}\) and backward tolerance \(10^{-5}\), because its relation summation order differs.
3. `diagonal` changes only the transition mask from all \((q,r)\) pairs to \(q=r\), with zero new trainable parameters.
4. A single-relation graph and a root/bias-only graph must reduce exactly to Original.

RTDR remains a custom architecture hypothesis. Its defensible claim concerns relation-transition routing, not delayed fusion and not a cross-domain paper-module transfer.

## Rejected candidates

H2GCN D1 motivates excluding ego nodes from neighborhood aggregation, but GCNet already computes

\[
W_n(x_i+\sum_{j\ne i}x_j)+W_e x_i
=W_n\sum_{j\ne i}x_j+(W_n+W_e)x_i.
\]

Because \(W_e\) is free, this is the same linear function class as separate ego and neighbor channels followed immediately by a linear combine. A self-edge de-duplication control may document optimization effects, but it is not a new expressive module and receives no formal training budget now.

Both directions are explicit: a separated neighbor/ego pair \((A,B)\) maps to current parameters \(W_n=A, W_e=B-A\); conversely current \((W_n,W_e)\) maps to separated \((A,B)=(W_n,W_n+W_e)\).

Centered Clipping permits an arbitrary initial center, while the source implementation defaults to a previous-round aggregate. In GCNet, however, neither that initialization nor the iteration count and threshold/radius scale has a grounded analogue; conversation nodes do not persist across shuffled mini-batches. One unusual neighbor already occupies 20–33% of a GCNet neighborhood, and missing rate 0.7 violates the good-majority interpretation behind the \(\delta\leq0.1\) theorem. A stateless one-step adaptation would be a new clipping rule, not a defensible source-faithful transfer, and overlaps the graph-grounded Soft Medoid candidate.

## Experiment update

The source-grounded/custom-distinct candidates eligible for training are now GenAgg, Soft Medoid, SSMA, and RTDR. Each candidate first uses the same formal subset: missing `{0.0,0.7}` × seeds `{66,67,68}`. This is 24 new tasks total without Original jobs: Phase A is the already registered GenAgg + Soft Medoid 12-task wave; Phase B is the SSMA + RTDR 12-task wave after their selectors and runner mappings pass tests. Both phases use the existing four-GPU/three-worker scheduler, and every task remains directly reusable in formal expansion.

After all formal jobs terminate and local provenance checks pass, the workflow synchronizes the completed artifacts back to `/data2/yb/paper`, writes bilingual result summaries, commits the completed code/results with Lore provenance, and pushes them to `https://github.com/YB123-DT/git_gcnet`. Failed or incomplete task directories are not published as completed versions.

SSMA must pass its mechanism gate and then beat a parameter-matched sum-plus-MLP control before a mechanism claim. RTDR must first pass exact Original-path and full-transition equivalence; its diagonal-routing result is reported as a custom hypothesis.
