# Minimal paired-history views (CoMM-style, not a paper reproduction)

User-specified experiment on original cfg84 no-JEPA Flat; seed66 only.
Two100-epoch runs: control(lambda0) and contrastive(lambda.1).
Original Flat seed66 reused, not retrained. Original optimizer/learning rate/
batch32/cyclic0–.7 schedule/task MSE/classifier/evaluation masks retained.

View1 uses original preparation unchanged. View2 drops each originally observed
modality with probability.2, restores one uniformly among original observed
modalities if all would be lost, and keeps padding zero. Actual drop fraction
is removed observed bits divided by View1 observed bits, not by all three slots.
No features unavailable in View1 are exposed to View2.

Contrast anchors: current masks equal AND a strictly earlier valid utterance
in the same conversation has different masks. No first-step/self difference,
future information, or cross-conversation history is used.

Two complete model forwards share parameters, not Memory state. OSRAM scan
allocates zero memory on each call. Model/backbone source is not modified.
Independent augmentation and View2 dropout RNG preserve original View1 RNG.
Dropout is still active: this is not a claim of history-only hidden differences.

Projector consumes post-Flat/pre-classifier hidden:1600->256->GELU->128.
Only anchor rows are projected. L2 normalization precedes symmetric cross-view
InfoNCE, temperature.1. Own paired row is positive; ALL other anchors from the
same conversation are excluded from negatives, including duplicate batch IDs.
No cross-conversation negatives: differentiablezero, task learning continues.
No queues, labels as contrast supervision, local/base/gap alignments or distillation.

Both runs execute projector/InfoNCE forward for comparable compute; control
omits that term from backward and its projector is unused by the task.
The main matched budget is two full OSRAM forwards, same batches/optimizer steps;
contrast backward adds overhead, so this is not exact wall-clock equality.
Projector initialization uses fork_rng. Inference does not call it. Checkpoints
retain its training state; offline backbone-only loaders must explicitly handle
the `history_contrast_projector.*` keys, not silently ignore arbitrary mismatches.

L_control=.5*(task1+task2); L_con=L_control+.1*InfoNCE.
Global clipping remains original1.0 over trainable parameters (including the
projector gradients in contrastive). No Gate, JEPA, completion, persistent masks,
freezing, architecture/readout changes or hyperparameter sweep.

History records cumulative anchors/counts per epoch, usable anchors, actualdrop,
InfoNCE/task1/task2 losses, skipped-negative batches and both mask digests.
Train W-F1 refers to View1, classification_loss to the two-task mean.
Counts across epochs are exposure counts, not unique utterances.

Server biggpu GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153; GPU4 forbidden.
Two staggered independent outputs under runs/control and runs/contrastive.
Per-rate BEST retained exactly as previous historical Test-oracle INTERNAL
diagnostic protocol, not formal validation-selected paper results.

Acceptance: subset/nonempty masks, strict-prior anchors, no sameconversation
negatives, independent Memory, unchanged default behavior/View1 masks, finite
gradients and projector updates, identical A/B mask digests, original eval masks.
