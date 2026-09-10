# Causal eta=.6 classification readout ablation

User approved retraining three readout variants on MOSI; Full is inherited.

1. Add `osram_emotion_ablation` (`full`, `local-only`, `local-base`, `local-gap`), default full. Keep legacy `osram_ablation` unchanged. Only emotion fusion slots are masked; returned predictor contexts and scan remain unchanged. Reject combining non-full legacy and new masking.
2. Verify default parity, exact per-step memory equality, unchanged returned contexts and structured predictor outputs, intended fusion slots, config/CLI, parameter/RNG identity and backward. Use existing remote Python, no repeated environment smoke.
3. Clone each Full seed config, changing only new readout flag. From-scratch 100 epochs, CMUMOSI, seeds66–70, cyclic, forward-only, write_step=.6, H8/K32/V32/output700, batch32, LR.001. No B2, no resume, no new losses.
4. Train local-only/local-base/local-gap (15 jobs); inherit Full histories. Evaluate test all eight rates every epoch. Reporting independently maximizes weighted-F1 for each seed/rate, earliest epoch on tie. Do not use eight-rate-mean selected checkpoint scores for comparison. Existing single-best checkpoint saver may remain unchanged because it does not affect training; explicitly distinguish stored checkpoint policy from reporting policy.
5. Save per-seed/rate epochs and metrics plus source histories/provenance. INTERNAL TEST-ORACLE DIAGNOSTIC ONLY; no formal generalization claims. Full retraining is not authorized or needed. Compare masks at aggregation.

Readout ablation does not remove memory from auxiliary JEPA learning. Parameters evolve differently across retrained groups; scan equality applies at fixed weights/inputs, not across independently trained models.

GPU placement: inspect current availability once; avoid busy 0/1 and unhealthy4. Target three simultaneous jobs per available GPU. No unrelated process interruption. Model code verification precedes launch.
