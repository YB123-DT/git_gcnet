# Two-level Memory evidence gate

## Scope

Optional module implementation and verification, not an authorized full training
run. Keep cfg84 no-JEPA Flat as the readout, retain Local skip and jointly
trainable original modules. No previous evidence-gate regularizer, prediction,
JEPA, persistent50/50, or frozen Stage2 behavior is enabled by this flag.

## Architecture

For the four fixed evidence slots Base, Gap-A, Gap-T, Gap-V:

1. **Feature level:** learn context_dim coefficients per evidence (1024 in the
   existing cfg84 Flat interface: two 512-dimensional directional slots).
   `F_i = 2 * sigmoid(z_i)`, `filtered_i = F_i * C_i`. This is per-channel, not
   one coefficient per attention head. Coefficients can suppress or amplify.
   Forward-only execution preserves the legacy second slot (zero or reused
   forward read according to the existing configuration); this module does not
   change that protocol or truncate the original Flat input.
2. **Evidence level:** compute one score from each filtered evidence and its
   conditioning; masked softmax over active evidence, `r_i = K * alpha_i`.
   `C_i_final = r_i * filtered_i`. Local is not part of the softmax competition.
3. Feed `[Local, Base_final, Gap-A_final, Gap-T_final, Gap-V_final]` into the
   unchanged Flat adapter and retain original Local skip and emotion_norm/head.

Shared small networks use Local, evidence, evidence type and availability as
conditioning. Local and evidence can be projected for gate computation only;
the final coefficients scale the original memory contexts. Level2 explicitly
sees Level1's filtered contexts, rather than an independent copy of raw contexts.

Base active on every valid utterance; Gap active iff corresponding modality is
missing. All inactive evidence and padding are safely zeroed before projection.
All-padding rows must avoid all-negative-infinity softmax NaNs.

Example observed A only: availability[1,0,0], active Base/Gap-T/Gap-V, K3.
If active alpha=[.2,.6,.2], r=[.6,1.8,.6], with Gap-A exactly0.
With all modalities present, K1 and Base evidence-level reweight is exactly1;
its feature-level filter may still change.

Zero-initialize only final feature-logit and evidence-score outputs. Initially
F1 and uniform active alpha make Kalpha1: recover original Flat computation.
Preserve shared initialization RNG; default flagoff retains old behavior.

## Interpretation limits

Normalization fixes the average active evidence-level reweight at1, NOT each
coefficient, and NOT the norm of the resulting representation. Feature-level
filters are not constrained to average1. The module neither guarantees stable
near-identity training nor proves which missing modality is actually recovered.
Gap-T preference means prioritizing that readout, not verified Text completion.

## Plan / acceptance

Add failing tests, implement optional model/config/CLI path, then verify identity,
off-path preservation, controlled softmax arithmetic, K1/K3/K4 behavior, padding
and inactive NaN masking/gradients, ablation isolation, filtered Level2 inputs,
joint gradients and diagnostics. Run actual cfg84 dimensions on biggpu GPU0 and
related regression suite. Commit/push implementation and evidence; no full run
or new performance scores are implied by these checks.
