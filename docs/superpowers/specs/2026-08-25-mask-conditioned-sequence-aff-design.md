# Mask-Conditioned Sequence AFF for GCNet

Date: 2026-08-25

## Primary anchor

Yimian Dai, Fabian Gieseke, Stefan Oehmcke, Yiquan Wu, and Kobus Barnard,
“Attentional Feature Fusion,” WACV 2021, arXiv:2009.14082.

Primary evidence:

- paper and source: <https://arxiv.org/abs/2009.14082>;
- authors' implementation: <https://github.com/YimianDai/open-aff>.

The paper defines AFF for two same-shaped feature maps as complementary soft
selection driven by MS-CAM:

```text
Z = M(X + Y) ⊙ X + (1 - M(X + Y)) ⊙ Y.
```

MS-CAM sums a local point-wise channel context and a global pooled channel
context before sigmoid. The official implementation uses a factor of two in
the final fusion, so a neutral 0.5 gate recovers direct addition.

## GCNet adaptation target

Replace only this operation in `GraphModel.forward`:

```python
hidden = hidden1 + hidden2
```

where `hidden1` and `hidden2` are the temporal-graph and speaker-graph branch
outputs with shape `[T,B,D_h]`. Graph construction, graph convolutions,
pre/post BiLSTMs, attention, reconstruction, losses, classifier, optimizer and
mask protocol remain unchanged.

## Mask-conditioned Sequence AFF

Let `X=hidden1`, `Y=hidden2`, `S=X+Y`, and let `p_t` be the six-dimensional
one-hot code for incomplete patterns A/T/V/AT/AV/TV; ATV maps to zero.

The local context is utterance-wise:

```text
L_t = W_L2 ReLU(LN(W_L1 [S_t || p_t])).
```

The global context is conversation-wise. With valid-utterance mask `u_t`:

```text
S_bar = sum_t u_t S_t / sum_t u_t
p_bar = sum_t u_t p_t / sum_t u_t
G = W_G2 ReLU(LN(W_G1 [S_bar || p_bar])).
```

The channel gate and AFF candidate are:

```text
w_t = sigmoid(L_t + G)
A_t = 2 * (w_t ⊙ X_t + (1 - w_t) ⊙ Y_t).
```

The complete-preserving output is:

```text
H_t = S_t + incomplete_t * (A_t - S_t).
```

`incomplete_t=0` for ATV and padded positions, otherwise one. Thus every
complete utterance is bitwise-preserved even when other utterances in the same
conversation are incomplete.

## Sequence-specific choices

- Paper PWConv becomes per-utterance Linear because GCNet tensors are
  `[T,B,D]`, not image maps.
- Paper spatial global average pooling becomes masked temporal mean within each
  conversation.
- LayerNorm replaces BatchNorm because dialogue batches have variable sequence
  length and often small effective batch size.
- Reduction ratio is `r=4`; bottleneck width is `max(D_h//4,1)`.
- The two output Linear layers are zero initialized. Therefore `w=0.5` at
  initialization and the module initially reproduces direct addition for all
  masks, not only ATV.
- The factor two follows the authors' released AFF implementation and is
  required for addition-preserving neutral initialization.

## Interface and variants

Create `MaskConditionedSequenceAFF(channels, reduction=4, pattern_dim=6)` in a
focused module. `GraphModel` receives:

```text
branch_fusion = addition | mask_sequence_aff
```

Default is `addition`. The new module is instantiated from a forked CPU RNG
state so official GCNet shared parameters and construction RNG remain
unchanged. Result filename and NPZ provenance record the fusion choice and both
stored-total and selected-path parameter counts.

## Required tests

1. Seven-pattern sequence encoding aligns with `[T,B]` and treats padding as
   inactive without accepting all-zero valid masks.
2. Output shape is `[T,B,D_h]`; invalid mask/umask shapes fail clearly.
3. All-ATV forward and gradients for X/Y equal direct addition exactly.
4. Zero-initialized module equals addition for every valid missing pattern.
5. Non-zero gate parameters make temporal/speaker selection content-,
   pattern-, and conversation-dependent.
6. Changing a padded timestep cannot change a valid timestep's global context.
7. Default addition model preserves official parameters, RNG and outputs.
8. CLI, filename, NPZ and parameter counts identify the fusion arm.
9. Existing BiLSTM, MPFiLM, CP-LECC and training tests remain green.

## First experiment

Run a single paired IEMOCAPSix fold-5 A/B:

- arms: official `addition` and `mask_sequence_aff`;
- rates: `0.0,0.1,...,0.7`;
- seeds: `66,67,68,69,70`;
- 100 epochs and the existing stage-aware paired mask bundles;
- same graph, context, optimizer and training configuration.

This is 80 jobs. `rate=0.0` is an equivalence/noise audit; the primary method
claim is the paired eight-rate macro effect and its sign consistency. No iAFF,
extra loss, new graph, or additional fusion variant is added before this A/B.

## Interpretation boundary

This is a cross-domain adaptation of AFF, not a claim that AFF itself is novel.
The research contribution being tested is whether modality-pattern and
conversation-conditioned complementary selection is a better GCNet branch
fusion primitive than unconditional addition. IEMOCAP fold-5 remains a
screening protocol because its existing loader reuses the held-out session for
validation and test samples.
