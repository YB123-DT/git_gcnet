# Gap-only Evidence Gate

INTERNAL DIAGNOSTIC ONLY

Status: implemented; seed66 authorized, GPU6 queue admission pending deployment.

One-stage from-scratch MOSI, 100 epochs, unchanged cfg84 causal no-JEPA Flat,
cyclic random missing 0.0–0.7, original per-rate BEST Test-oracle protocol.
Reference: osram_mosi_memory_gap_ablation_20260920/full/seed_66.
Prior four-slot Gate: osram_local_evidence_gate_20260930/results/seed_66.

Only intervention: valid Base gate fixed to 1, inactive Base/padding 0;
three active missing-modality Gap gates retain 1+0.2*tanh(shared MLP).
L2 coefficient 0.001, averaged over active-valid Gap slots only; no active Gap
means zero penalty. Gate parameter shapes and initialization unchanged.
No Memory write/read/query, Local, Flat, task head or auxiliary objective changes.
Base is not modulated and remains in the original Flat input. Joint training
can still change backbone and full-modality results.

Default flag false preserves the prior four-slot behavior and checkpoint keys.
New flag: --osram-local-evidence-gate-gap-only (requires evidence gate).

Focused CPU tests cover initialization, Base identity, masks, active-Gap
regularization normalization, no-Gap zero penalty, finite gradients and unchanged
Gap values for identical weights. No new GPU smoke or parameter sweep.
Verified on biggpu CPU with the existing s0 environment: 2 tests passed (2.99s).
Local default Python has no torch; no dependencies were installed.

GPU6 only; wait for existing core20 exclusive queue to drain. Do not interrupt it.
Remote run will save effective config, code/source hashes, reference hashes,
status, predictions and eight BEST checkpoints. No scores claimed before completion.
