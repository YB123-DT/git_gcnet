# Paired cosine learning-rate screen

INTERNAL DIAGNOSTIC ONLY; inherited Test-oracle per-rate BEST protocol.

Authorized: directly launch after configuration change. Two new from-scratch
MOSI seed66 runs: Flat and original Nested;100epochs, eight cyclic missing rates.
Reuse completed constant runs in osram_nested_training_gradients_20261010; no
repeat constant training, no other seeds or model changes in this first screen.

Change only lr_schedule=cosine. Existing warmup_ratio=.05 becomes active:
epochs1–5 linear warmup from.0002 to.001, epochs6–100 cosine to0.
This tests the existing WARMUP+COSINE package, not an isolated late-decay effect.
Adam, peakLR.001, weight decay1e-5, batch32, clip1, MSE sample-mean, emotion-only,
no-JEPA, cfg84 output1600, heads8/key64/value64 and random-missing masks unchanged.
All parameter groups retain their existing multiplier1. Nested has no separateLR.
Do not silently change global defaults or old experiment files/checkpoints.

Reuse sealed original source_ad211c0 and the previously verified passive gradient
wrapper; add an optional schedule flag, defaultconstant, and per-epoch actualLR
records. Validate configuration differs only by schedule/known Nested switch;
validate actual existing schedule's endpoints and group ratios before launch.
Persist independently under /data1 on biggpu healthyGPU7, two parallel jobs;
save per-rate BEST and last_training.pt with original model/optimizer/RNG state.

Compare both against their own seed66 constant scores, then compare
(Nested_cosine−Flat_cosine) with (Nested_constant−Flat_constant). Report all rates,
mean8, high-missing, BEST epochs and actualLR traces. No claim of improvement
before completed results; no automatic multi-seed expansion.
