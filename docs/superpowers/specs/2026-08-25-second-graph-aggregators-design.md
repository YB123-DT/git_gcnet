# GCNet Second-Layer Aggregator Design

Canonical English design. Chinese translation: [2026-08-25-second-graph-aggregators-design.zh.md](2026-08-25-second-graph-aggregators-design.zh.md). English mirror: [2026-08-25-second-graph-aggregators-design.en.md](2026-08-25-second-graph-aggregators-design.en.md).

## Decision

Evaluate two cross-domain replacements for the neighborhood aggregation primitive in GCNet's second graph layer:

1. GenAgg is the primary candidate.
2. Scaled Soft Medoid is the backup candidate.

The experiment changes neither the first-layer RGCN nor graph topology, relation definitions, recurrent context, branch fusion, reconstruction, classifier, optimizer, natural missing protocol, folds, or seeds. Existing Original results are inherited and never retrained.

## Traceable sources

- Ryan Kortvelesy, Steven Morad, and Amanda Prorok, “Generalised f-Mean Aggregation for Graph Neural Networks,” NeurIPS 2023. [arXiv](https://arxiv.org/abs/2306.13826), [official implementation at the paper-era release](https://github.com/Acciorocketships/generalised-aggregation/blob/3c95c10afac4bda77afc30e80a7481c7e537fca1/genagg/genagg.py).
- Simon Geisler, Daniel Zügner, and Stephan Günnemann, “Reliable Graph Neural Networks via Robust Aggregation,” NeurIPS 2020. [arXiv](https://arxiv.org/abs/2010.15651), [official implementation](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/blob/4f94140afb7fd2ef5bf77f45a5efc7b2d6eb2a09/rgnn/means.py).

On 2026-08-25, exact arXiv searches combining `GenAgg`, `generalised f-mean`, `Soft Medoid`, `medoid`, `learnable aggregation`, or `robust aggregation` with `multimodal sentiment`, `emotion recognition`, or `emotion` returned no matching records. The local MSA/MERC/ERC Markdown, TeX, and text corpus also contained no exact mechanism match. This supports a cross-domain-transfer claim, not an absolute first-use claim.

## Verified GCNet insertion point

`gcnet/model.py::GraphNetwork` currently constructs:

```python
self.conv1 = RGCNConv(...)
self.conv2 = GraphConv(hidden_size, hidden_size)
```

PyG 2.0.1 was inspected in the official biggpu training environment. `GraphConv` defaults to `aggr="add"` and computes

\[
y_i = W_{\mathrm{neighbor}}\sum_{j\in\mathcal N(i)}x_j
      + W_{\mathrm{root}}x_i + b.
\]

The temporal and speaker branches own separate `GraphNetwork` instances, so they also own separate second-layer aggregators. With the locked `windowp=2` and `windowf=2`, an interior utterance has at most five incoming graph edges, including the explicit self edge. The separate GraphConv root transform remains separate.

## Candidate A: source-faithful GenAgg

For incoming raw neighbor features \(x_j\), GenAgg applies the augmented f-mean

\[
\operatorname{GenAgg}(X_i)=
f^{-1}\left(n_i^{\alpha-1}\sum_j
f(x_j-\beta\mu_i)\right).
\]

The result then passes through the existing GraphConv neighbor linear transform. This ordering matches the GenAgg authors' use of `GraphConv(aggr=GenAgg())`. The root transform and bias remain unchanged.

The implementation uses the paper-experiment MLP dimensions `1-2-2-4` and inverse `4-2-2-1`, Mish, BatchNorm, Kaiming-normal initialization of Linear weights only, default Linear bias initialization, and learnable scalars \(\alpha\) and \(\beta\). For \(x_c=x_j-\beta\mu_i\), the exact inverse objective is

\[
\mathcal L_{\mathrm{inv}}=\operatorname{mean}
\left[\left(\left|f^{-1}(f(x_c))\right|-|x_c|\right)^2\right].
\]

The compatibility port reuses the already computed \(f(x_c)\), rather than re-running the encoder inside a hook, so BatchNorm running statistics update once instead of twice. This is an explicit engineering deviation from the released hook while preserving the paper objective and gradient. Temporal and speaker losses are summed and added once to the normal training loss with weight 1.0; no module calls `.backward()` during forward.

The paper-era code requires a newer PyG aggregation API and `nn.Mish`, neither of which exists in the locked Torch 1.8/PyG 2.0.1 environment. The compatibility port therefore implements Mish as `x * tanh(softplus(x))` and aggregation with `torch_scatter`; it adds no dependency.

With the paper-experiment BatchNorm architecture, the implementation contributes 59 trainable parameters per branch and 118 across both branches: 22 forward-map Linear parameters, 19 inverse-map Linear parameters, 16 BatchNorm affine parameters, and two scalars. The current package default disables BatchNorm and is not the selected experiment configuration. Construction must fork and restore the CPU RNG around only the new parameters so all shared GCNet tensors retain the Original initialization.

Rationale: natural missingness changes the distribution of first-layer relational messages. A fixed sum can only accumulate them, while GenAgg can learn cardinality dependence, centralization, and a nonlinear scalar transform without assuming that most neighbors are complete or clean.

## Candidate B: scaled Soft Medoid

For transformed neighbor messages \(m_j\), compute

\[
d_j=\sum_k\lVert m_j-m_k\rVert_2,
\qquad
s_j=\operatorname{softmax}(-d_j/T),
\qquad
\operatorname{Agg}(M_i)=n_i\sum_js_jm_j.
\]

The neighbor transform is applied without bias before distances; the original neighbor bias is added once after aggregation. The root transform remains unchanged. Packed padding is excluded from both distance sums and the candidate softmax, and \(n_i\) is the true incoming-edge count. A zero-degree target bypasses softmax and receives a zero neighbor aggregate, giving exactly `lin_r(x_i) + lin_l.bias`. This makes the operation identical to Original add aggregation for a single neighbor or a homogeneous neighborhood, and convergent to add as \(T\to\infty\). Temperature is fixed at the source default `T=1.0` for the first experiment and is not learned.

The implementation packs incoming edges into `[N, max_degree, D]` and computes pairwise distances only inside each neighborhood. It introduces zero trainable parameters. No dense `[N,N,D]` tensor, adjacency learning, attention layer, or top-k truncation is allowed.

Rationale: strongly missing utterances may create geometric outliers in message space. Soft Medoid suppresses isolated messages. Its prior is weaker than GenAgg because locked GCNet neighborhoods contain only three to five points and, at missing rate 0.7, incomplete neighbors need not be a minority.

## Rejected alternatives

- Baseline-preserving sum initialization for GenAgg was rejected for the first run because it changes the source implementation and weakens mechanism attribution.
- Mask-conditioned GenAgg or Soft Medoid was rejected because it confounds aggregation replacement with the missing-pattern conditioning already tested by MPFiLM and CP-LECC.
- Additional temperature search, parameter-matched controls, edge attention, learned topology, and new losses unrelated to GenAgg inversion are deferred until a candidate shows positive evidence.

## Interface and unchanged behavior

Add one CLI/model selector:

```text
--second-graph-aggregation add|genagg|soft_medoid
```

`add` is the default and must construct the exact existing PyG `GraphConv`, with identical state-dict keys, parameter count, RNG progression, outputs, and gradients. GenAgg and Soft Medoid replace both `conv2` instances only. They consume the existing explicit self edge as a normal neighbor and retain the separate GraphConv root transform; no self edge is added, removed, or deduplicated. Existing `--graph-conv-variant` continues to control `conv1` and remains `original` in this experiment.

## Minimal verification

No epoch-level smoke run is permitted. Verification is limited to deterministic unit and one-batch checks:

1. Original `add` parameter/RNG/forward/backward equivalence.
2. GenAgg hand-computed reduction in evaluation mode, fixed identity \(f\) with `alpha=1,beta=0` sum special case, exact inverse-MSE gradients, finite CPU backward, and exact added parameter count.
3. Soft Medoid hand-computed weights, packed-versus-ragged and permutation equivalence, zero-degree behavior, homogeneous and single-neighbor add equivalence, finite backward, and zero parameter delta versus GraphConv.
4. Both GCNet branches use the selected second layer while every first layer stays Original RGCN.
5. Python 3.8 source compatibility and CLI provenance.
6. One FP32 GPU forward/backward in the locked biggpu training environment after synchronization.

The established test command is:

```bash
PYTHONPATH=gcnet:. /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  -m unittest discover -s tests -v
```

Training uses `/data2/yb/reproduction_envs/gcnet-official/bin/python` on biggpu. The test environment and training environment have different responsibilities and must not be interchanged.

## Falsifiable experiment protocol

Dataset and protocol remain locked to IEMOCAPSix, fold 5, features `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, and `manet_UTT`, hidden 200, graph hidden 100, window 2/2, 100 epochs, seeds 66-70, and the immutable stage-aware mask bank.

The 40-task Original control already exists at:

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

It contains 40 NPZ archives and is inherited by exact task key and mask SHA256.

The first wave has exactly 12 new tasks: two candidates × missing rates `{0.0, 0.7}` × seeds `{66,67,68}`. Four GPUs run three jobs each. It is launched with `stage=formal` and an explicit reduced grid, not `stage=gate`, because the existing runner binds stage into both artifact paths and immutable command provenance. The same formal paths are therefore reused if a candidate advances.

The inherited Original archives predate the current branch-fusion, pre/post-context, selected-path-count, and second-aggregation fields. Their dedicated validator treats absent historical fields as the locked legacy defaults (`addition`, `bilstm`, `bilstm`, `add`) while still requiring the original run manifest, command, fold, feature, seed, rate, parameter count, and mask SHA. Candidate jobs use the current strict payload validator. Original and candidate roots must be distinct.

A candidate advances only if all archives pass provenance validation, both rate-level paired mean F1 deltas are positive, the seed-macro paired delta is positive, at least two of three seed-macro deltas are positive, and no run is non-finite or collapsed. An advancing candidate is completed to eight rates × five seeds without rerunning its first-wave tasks. A rejected candidate is recorded and stopped.

## Synchronization contract

The authoritative editable branch is the local worktree:

```text
/data2/yb/paper/GCNet_TPAMI/.worktrees/second-graph-aggregators
```

The biggpu execution copy is:

```text
/data2/yb/paper/GCNet_second_graph_aggregators_20260825
```

The two `/data2` trees are not shared. Code is synchronized local-to-biggpu before testing/training; results and manifests are synchronized back to the local worktree after completion. A source-tree SHA256 manifest is compared before launch and after return.
