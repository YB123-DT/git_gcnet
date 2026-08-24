# Complete-Preserving Missing-Pattern FiLM RGCN Design

## Objective

GCNet's first `RGCNConv` uses a fixed relation-wise mean. The proposed module adapts the non-MSA/MERC GNN-FiLM primitive so that target content and target missing pattern modulate a source message containing source content and source pattern. Both temporal and speaker first layers are replaced; topology, the second graph layer, recurrent context, attention, reconstruction, classifier, losses, and fusion stay unchanged.

The contribution is not generic relation-aware attention, which already exists in RGAT, DualGATs, and HGT. The testable contribution is ordered source-target missing-pattern conditioning, feature-wise modulation on GCNet's fixed relational graph, and an exact complete-preserving constraint.

## Locked operation

Six contrast coordinates encode A, T, V, AT, AV, and TV; ATV is the zero vector and 000 is rejected. Completeness is computed directly from the raw three-bit mask.

For relation `r` and edge `u -> v`:

\[
q_{u,r}=h_uW_r+p_uP_r,
\quad [\Delta\gamma_{v,r},\Delta\beta_{v,r}]=[h_v\Vert p_v]G_r,
\]

\[
a_{uv}=1-c_uc_v,
\quad m_{u\to v}^{r}=(1+a_{uv}\Delta\gamma_{v,r})\odot q_{u,r}+a_{uv}\Delta\beta_{v,r}.
\]

Messages use PyG-compatible relation-wise `scatter_mean`, followed by the unchanged root transform and bias. `P_r` and `G_r` are zero initialized. Consequently every variant starts from Original for any pattern, and all-complete inputs permanently disable the new path.

## Variants and protocol

- `original`: official PyG `RGCNConv`.
- `pattern_only`: source pattern correction plus target-pattern-only FiLM.
- `full`: the complete formula.
- `content_film_control`: mask-independent target-content FiLM with parameter padding to match Full.

Masks are generated once with the official `random_mask` rule and stored by conversation and utterance. The bank is shared by every arm. Node masks are flattened conversation-first to match `batch_graphify`.

The first experiment uses IEMOCAP-6 fold 5. Unit tests cover complete forward/backward equivalence, zero gradients for new parameters under complete input, all seven encodings, the 0.7 regime, homogeneous neighborhoods, single-neighbor relations, node ordering, parameter counts, and CPU/GPU FP32. Formal evaluation then uses eight missing rates and five seeds with the same mask banks across arms.

Known risks are IEMOCAP overfitting from FiLM capacity, unconstrained scaling, and an inaccessible 2026 HGDN paper that prevents a defensible "first" claim.
