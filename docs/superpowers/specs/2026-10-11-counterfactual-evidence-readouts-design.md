# Counterfactual Evidence Readouts Design

## Status and scope

This specification defines two readout-only MOSI experiments on the cfg84
no-JEPA causal OSRAM baseline:

- A: Counterfactual Evidence Marginal Readout (CEMR).
- B: Coalition Interaction Readout (CIR).

Both retain the original Flat anchor. Neither changes ObservedSetEncoder,
queries, Key/Value projection, Memory read/write, availability schedules,
Local path, Local skip, task head, task loss, or inference-time Memory scan.
There is no JEPA, completion, Gate, persistent-missing mix, contrastive loss,
auxiliary loss, second Memory trajectory, or label-derived input.

The first screening consists of seed66 only, 150 epochs, with the established
cfg84 Test-oracle protocol. It is an internal diagnostic and not a paper result.

## Shared anchor and evidence definition

At utterance t, let the four historical evidence slots be

\[
E_t=(B_t,G_t^A,G_t^T,G_t^V).
\]

Base is active only on valid utterances with at least one preceding valid
utterance. Gap-m is active only when the utterance is valid, history exists,
and modality m is currently missing. Inactive and padding slots are zeroed with
`torch.where` before any biased layer and are masked again after encoding.

For any retained evidence subset S, define the shared Flat pre-norm function

\[
u_t(S)=S_L(L_t)+A([L_t,C_t(S)]),
\]

where `S_L`, `A`, and the evidence layout are the original trainable
`local_skip`, `emotion_adapter`, and fixed Base/Gap slots. The normal anchor is

\[
u_t^F=u_t(E_t),\qquad h_t^F=LN(u_t^F).
\]

The existing Flat parameters are not copied, frozen, or replaced. Both new
methods add a zero-initialized residual before the existing `emotion_norm`:

\[
h_t=LN(u_t^F+W_R r_t).
\]

Consequently, both enabled models exactly match their own Flat anchor at
initialization, while the default-disabled path remains byte-for-byte and
seed-for-seed unchanged.

## Shared-dropout evaluation of Flat views

Training-time subset differences must not be contaminated by independent
Dropout draws. All Flat views of the same utterance use the same dropout mask
inside the original adapter. The implementation evaluates the original Flat
layers explicitly as

`LayerNorm -> Linear -> GELU -> shared-view Dropout -> Linear`.

One Bernoulli mask is sampled for the full view and broadcast to every subset
view of that utterance. Its marginal distribution is identical to the original
Dropout. In evaluation mode, no mask is applied. The full view supplies the
actual anchor; it is not recomputed through a separate parameter copy.

Default-off code continues to call the untouched original adapter directly.
Tests must show that equal subset inputs produce equal outputs during training,
which proves that a reported marginal difference comes from evidence removal
rather than random Dropout.

## Experiment A: Counterfactual Evidence Marginal Readout

For each active evidence slot i, compute a leave-one-evidence-out Flat view:

\[
d_t^i=u_t^F-u_t(E_t\setminus\{i\}).
\]

The four fixed tokens are constructed as

\[
x_t^i=P_D(d_t^i)+e_i+P_A(a_t),
\]

where `P_D: 1600 -> 128`, `e_i` identifies Base/Gap-A/Gap-T/Gap-V, and
`P_A` embeds the three-bit current availability. Inactive tokens are zero and
masked from all attention keys, values, and pooling.

The token set is processed by a two-layer, four-head Transformer encoder with
model dimension 128 and FFN dimension 256. A masked active-count mean produces
`r_t` in 128 dimensions. `W_R: 128 -> 1600` is zero-initialized.

This branch consumes the original Flat function's conditional response to each
historical evidence slot. It does not interpret token norms as reliability or
claim to recover a missing modality.

The method requires one full and at most four omit views, vectorized as five
fixed views while safely masking inactive slots.

## Experiment B: Coalition Interaction Readout

CIR uses the same anchor and shared-view Flat evaluation, but explicitly keeps
first- and second-order evidence effects. Let the empty-history Flat view keep
Local and zero all four historical slots. For active i and active pairs i,j:

\[
I_t^i=u_t(\{i\})-u_t(\varnothing),
\]

\[
I_t^{ij}=u_t(\{i,j\})-u_t(\{i\})-u_t(\{j\})+u_t(\varnothing).
\]

Four singleton and six unordered pair slots are always allocated, with inactive
effects hard-zeroed. Each effect is projected `1600 -> 128`, receives a learned
singleton/pair identity embedding plus the same availability embedding, and is
processed by a two-layer, four-head Transformer encoder (128 model dimension,
256 FFN). Masked active-count mean pooling produces `r_t`; the final
`128 -> 1600` projection is zero-initialized.

CIR uses one empty, four singleton, six pair, and one full view: twelve fixed
Flat views per utterance, vectorized without rerunning Memory. It does not use
higher-order coalitions. This bounds compute and isolates the question of
whether explicit Base/Gap pair effects add value beyond leave-one-out marginals.

## Configuration and compatibility

Introduce one mutually exclusive option:

`--osram-counterfactual-readout {none,marginal,coalition}`

Default is `none`. It is legal only with the cfg84 Flat, causal, no-JEPA,
emotion-only configuration and cannot be combined with previous readout blocks.
Legacy checkpoints load strictly when the option is `none`. Enabled checkpoints
record the method, dimensions, view count, and parameter count.

The two methods use isolated module initialization so adding a disabled option
does not change downstream RNG. Neither method changes the task-head interface.
Padding hidden outputs remain exactly zero.

## Diagnostics

Record detached summaries only:

- active evidence count;
- mean marginal-token norm for A;
- mean singleton and pair-effect norms for B;
- residual-to-Flat-anchor norm ratio;
- number of evaluated Flat views;
- peak memory and epoch wall time.

These statistics are not called reliability, conflict, emotion shift, or
missing-modality reconstruction.

## Correctness tests

Before formal training, verify:

1. Default `none` reproduces legacy state, output, RNG, and optimizer grouping.
2. Zero-initialized enabled models reproduce the Flat prediction.
3. The full-view anchor equals the original Flat computation in evaluation.
4. Shared-view Dropout gives equal outputs for equal inputs during training.
5. Removing evidence changes only its designated fixed slot.
6. Inactive Gap, no-history Base/Gap, and padding cannot leak through bias or
   type embeddings.
7. A leave-one-out difference is zero when the selected evidence is zero.
8. B singleton and pair identities satisfy their numerical Mobius formulas.
9. The residual branch receives finite, nonzero gradients after optimization
   begins; the original Flat remains trainable.
10. Memory states and queries are identical with the option off/on under the
    same inputs and weights.
11. Legacy checkpoints remain strict-load compatible.
12. A real MOSI batch completes forward, backward, clipping, and optimizer step
    without non-finite values.

## Formal screening protocol

Run two jobs only:

- MOSI seed66, method A, 150 epochs.
- MOSI seed66, method B, 150 epochs.

Use the established schedule: epochs 1--100 at LR 0.001, then full-state resume
for epochs 101--150 at LR 0.0001. Preserve model, optimizer, RNG, masks, and best
checkpoint references across the resume. Use cyclic random-missing training and
per-rate Test-oracle selection at missing rates 0.0--0.7, matching the existing
Flat-150 reference.

After correctness and real-batch smoke pass, launch both independent jobs
concurrently on one healthy 32-GB biggpu device. Never use host GPU4. Admission
requires a measured two-process smoke or memory estimate leaving safe headroom;
otherwise run them sequentially on that same card without changing batch size
or protocol.

Report per-rate W-F1, eight-rate mean, high-missing mean, selected epochs,
parameter count, peak GPU memory, and wall time against Flat-150. Do not expand
either method to additional seeds automatically. A method advances only if its
seed66 aggregate evidence justifies a separate user decision.

## Interpretation boundary

These experiments test whether explicitly encoding the already-trained Flat
function's response to historical evidence subsets improves its readout. They
do not prove causal evidence value, missing-modality recovery, reliability
estimation, or semantic conflict detection. Test-oracle screening is internal
model selection and cannot be reported as deployable single-checkpoint
performance.
