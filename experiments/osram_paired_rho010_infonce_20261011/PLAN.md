# Single-seed rho.10 + InfoNCE

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

Train one MOSI seed66 model from scratch for100epochs:
L=.9 task(View1)+.1 task(View2)+.1 symmetric InfoNCE.
Reuse completed rho.10 task-only C02 (81.230699%, high76.202874%) as control.
Do not retrain control, add a grid or automatically expand seeds.

Historical backbone snapshot6e2a998 and recorded config retained exactly; only
history_contrast_weight changes0→.1. Verify four source hashes against old C02
PROVENANCE before training. View2 observed-bit drop.2; contrast temperature.1.
Original MSE task, Adam.001 constant, batch32, dropout.5, clip1, weight decay1e-5,
eight cyclic random missing rates, noJEPA Flat1600 retained.
Only same-current/different-prior-history anchors aligned; other utterances
same conversation excluded from negatives. Independent zero Memory for views.
Single-view inference; projector and contrastive loss are training-only.

Server biggpu physical GPU6, UUID pinned, bad GPU4 excluded. Isolated historical
code snapshot and new output; no changes to other running experiments.
Keep per-rate BEST and original mask protocol. Verify final evaluation masks
and every epoch's View1/View2 mask hash against C02; report failures, not silently
change protocol. Save per-rate W-F1, eight/high means and training exposures.
