# Verification before full one-stage launch

- TDD: initial new assertions failed on absent L1 penalty/config field.
- Local18 tests passed; expanded final remote focused regression28 passed,
  one existing PyG deprecation warning (9.16s). Tests cover active-only L1
  value/gradient, defaultL2 behavior, once-weighted actual train_epoch loss,
  joint Flat updates, exact launcher config delta and forbiddenGPU4.
- Remote seed66 one-epoch smoke completed and passed canonical-mask/checkpoint
  validation for all eight evaluation rates; provenance status complete.
- Training diagnostics explicitly record regularization type `l1`. First-epoch
  logged penalty0 (identity initialization), not evidence lambda is strong enough.
- After the epoch, miss0.7 evaluation gate means: Base0.998442, A0.999929,
  T0.998925, V1.000095. Every active gate within0.01 of1, zero saturation.
- At equal lambda, for nonzero deviations the L1 restoring gradient w.r.t. g is
  stronger than L2 by1/(2*abs(g-1)); this is an analytic scale check, not a
  measurement of balance against task gradients. Lambda0.001 is fixed for a
  one-factor L1-vs-L2 comparison, not asserted optimal.

Keep original Flat and all backbone parameters trainable. No checkpoint loading
or freezing, no new loss besides replacement of the existing Gate penalty.
Smoke metrics are implementation evidence only, not final accuracy evidence.
