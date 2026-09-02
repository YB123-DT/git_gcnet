# Matched Stratified Original GCNet Control

## Research question

Does the current Missing-M3 model outperform Original GCNet when both models
receive exactly one masked view per source conversation and use the same
stratified rate assignments, training batches, evaluation masks, epoch budget,
and checkpoint-selection rule?

The completed JEPA-versus-gradient-off experiment isolates the JEPA gradient
inside the Missing-M3 architecture. It does not compare that architecture with
Original GCNet, because disabling the JEPA coefficient leaves the observed-set
encoder, student projectors, and other Missing-M3 parameters in place.

## Control definitions considered

1. **Original GCNet with its masked reconstruction objective.** This is the
   selected control. It uses raw zero-filled concatenated features, the original
   pre-graph BiLSTM, temporal and speaker graphs, branch addition, classifier,
   linear reconstruction head, and classification-plus-masked-reconstruction
   loss.
2. **Original GCNet architecture with classification only.** This can isolate
   the reconstruction objective, but it is an ablation rather than the formal
   Original method. It is deferred unless the main control exposes a specific
   reconstruction confound.
3. **Missing-M3 with `jepa_weight=0`.** This is already complete and remains the
   JEPA-gradient-off arm. It cannot be renamed Original because its input
   encoder and parameterization differ from GCNet.

Only option 1 is required for the next decision. Running options 1 and 2
together would answer an extra question at twice the training cost and is not
needed to decide whether Missing-M3 exceeds Original.

## Architecture

The implementation lives in an independent `gcnet_original_stratified/`
package so the completed Missing-M3 path remains unchanged. Shared protocol
utilities are reused rather than copied.

```text
complete frozen A/T/V features
        -> matched conversation-level stratified mask
        -> raw zero-filled [A;T;V]
        -> Original pre-graph BiLSTM
        -> Original temporal GCNet branch
           + Original speaker GCNet branch
        -> Original classifier
           + Original linear reconstruction head
```

The model is the repository's package-safe `gcnet_modality_jepa.model.GraphModel`
with the following options fixed:

- `enable_reconstruction=True`;
- `graph_branch_mode="both"`;
- `recurrent_padding_mode="legacy"`;
- `postgraph_sequence_mode="independent"`;
- `graph_message_calibration="none"`;
- no student projector, teacher, MMoE, missing predictor, conditioned readout,
  or completion path.

A parameter-free adapter may normalize the call signature used by the shared
evaluation helpers. It must not add tensors to the state dict.

## Training objective

For each stratified batch, the model performs one forward, one backward, and
one optimizer update:

```text
L_original = L_task + L_masked_reconstruction
```

The reconstruction coefficient is exactly `1.0`, `reccls_flag` is false, and
the reconstructed feature is never fed into a second classification forward.
At requested rate 0.0 there are no missing targets, so reconstruction loss is
an exact differentiable zero.

The formal repository corrected `gcnet_modality_jepa.loss.MaskedReconLoss` is
used. The literal upstream implementation flattens `[L,B]` outputs against a
`[B,L]` mask and normalizes missing targets inconsistently; it is not the loss
used by the repository's prior formal Original sweeps. Provenance therefore
records `reconstruction_loss_variant=corrected-formal-repo`. A literal-upstream
loss, if ever needed, must be named as a separate diagnostic and cannot be
mixed into this control.

## Matched protocol

The control must match the completed Missing-M3 experiment on:

- IEMOCAPSix, fold 5, seeds 66--70;
- 100 epochs, batch size 32, Adam learning rate `1e-3`, weight decay `1e-5`;
- hidden size 200, windows 2/2, dropout 0.5, time attention disabled;
- one conversation-level stratified rate from 0.0--0.7 per conversation;
- 120 source conversations, 120 masked views, four forwards and four optimizer
  steps per epoch;
- deterministic training assignment and fixed validation/test mask schedules;
- checkpoint selection by mean validation weighted F1 over all eight rates;
- one selected checkpoint evaluated on all eight test rates.

Model parameters necessarily differ, and Original performs a reconstruction
loss that defines its published training method. These are method differences,
not protocol mismatches.

## Interfaces and artifacts

The independent trainer writes the same lightweight contract as Missing-M3:

- `config.json`, `history.json`, `metrics.json`, and `status.json`;
- `best.pt` while running remotely, excluded from the Git result copy;
- eight compressed prediction NPZ files;
- per-epoch rate assignments, raw missing/total counts, realized rates,
  source-view-forward-update counts, and assignment hash;
- validation/test mask hashes, source commit, and source-file SHA-256 values.

The shared bounded-GPU runner receives an explicit model-arm selector. Its
default remains Missing-M3 so historical commands do not change. The Original
arm selects the independent module and rejects GPU 4 as before.

## Verification

1. The adapter and underlying `GraphModel` produce identical logits,
   reconstruction tensors, hidden states, gradients, parameters, and state-dict
   keys for the same initialization and input.
2. The Original control contains none of the Missing-M3 projectors, teacher,
   MMoE, predictor, completion, or availability-conditioned readout parameters.
3. Corrected masked reconstruction is zero at rate 0.0 and equals an explicit
   missing-only modality-wise MSE calculation otherwise.
4. One stratified source batch performs one model forward and one optimizer
   update, and reports reconstruction loss separately from task loss.
5. Seed-matched Original and Missing-M3 runs have identical per-epoch
   assignment hashes and identical validation/test mask hashes.
6. Best epoch is selected only from the eight-rate validation mean; test
   metrics are produced after loading that checkpoint.
7. Existing Missing-M3 tests and outputs remain unchanged.

## Decision rule and claim boundary

Compare With-JEPA, JEPA-gradient-off, and matched Original using paired
seed-level eight-rate and high-missing-rate means. This control permits the
phrase **"Original GCNet architecture and reconstruction objective under the
matched stratified protocol."** It is not an exact reproduction of the
upstream per-rate training protocol, because equal-budget stratified training is
deliberately shared across all three arms.

Do not claim Missing-M3 superiority from the JEPA-gradient-off comparison alone.
Do not call the matched control an exact upstream reproduction, and do not call
the classification-only architecture Original without an ablation qualifier.
