# MELD B1/B2 implementation plan

User-approved design: full forward Base/Gap slots, B1 input2435 and B2 input2564.
No compression or pooling of those slots. Existing Flat and Local Skip remain intact.

1. Implement and test a standalone window8 causal Local-history attention residual.
   Shared attention width128; B1 task MLP2435→128→1600; B2 MLP2564→128→1600.
   Auxiliary S predictor uses detached original representations for its BCE only.
2. Add opt-in pre-LayerNorm hooks around the archived numerical Flat implementation.
   Assert baseline initialization/RNG/output parity and original optimizer ownership.
   Test no-history, missing Gap, padding, strict causality, auxiliary gradient isolation.
3. Add an isolated two-job runner using the already verified O6 configuration:
   MELD seed66,50epochs,batch32,Adam lr1e-5,weight_decay2e-4,constantLR,
   gcnet-random-mask-v1,per-rate Test-best ACC with same-forward UA/W-F1.
   Run from scratch with original parameter initialization preserved, not a weights-only
   continuation of O6. B2 adds coefficient0.05; no other objectives or modules.
4. Verify finite real-batch GPU forward/backward/update and recoverable snapshots.
   Capture exact selected predictions/masks. Record all default metadata encodings.
5. Commit/push code, deploy immutable source, launch B1/B2 concurrently on a healthy
   UUID-bound GPU, preserve unrelated jobs, verify live progress and saved artifacts.

All results are INTERNAL TEST-ORACLE SCREEN, not independent test claims.
