# Verification before full training

Initial remote regression32passed (one existing PyG deprecation warning).
Additional aggregate-count test follows red/green; runner3tests pass locally.
Independent code reviewer: no launch-blocking semantic defect.

Actual cfg84 GPU6 dimensions512/1024/1024, post-Flat hidden1600:
default-off current training versus pre-change train_gcnet from the preserved
memory-only experiment snapshot produces bit-identical metrics and model state
after two Adam updates. Projector1600->256->128 gets finite gradients and updates
under paired contrast training using the real optimizer-group builder.

Unit tests verify subset/nonempty masks, strictprior anchors, padding exclusion,
crossconversation-only negatives including duplicate IDs, differentiablezero
when no eligible negatives, preserved View1 masks/logits/RNG, and independent
Memory across A/B/A forwards. No OSRAM/model.py changes in this experiment.

Both one-epoch real-data smoke arms finished with valid8best checkpoints and
original canonical evaluation masks. Each saw1284valid utterances,3710observed
bits,720removed bits (19.407008%),644anchors and644usable anchors; no batch had
to skip InfoNCE. View1 hashes identical between A/B; View2 hashes also identical.
Smoke is isolated at remote sibling smoke/, excluded from final experiment.

Scope: implementation and smoke checks do not demonstrate performance gains.
Full training uses fixed lambda.1/temp.1/drop.2, no tuning or early stopping.
