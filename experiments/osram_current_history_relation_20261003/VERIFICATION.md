# Correctness verification

Implementation commit: `7777d2d`. No inference/training performance claim here.

Before implementation, new-module tests failed because the module was absent.
Runner tests likewise failed before its implementation; then passed.

On biggpu GPU5 / V100, torch2.2.2 CUDA12.1:

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -m unittest \
tests.test_current_history_relation tests.test_relation_runner \
tests.test_memory_shift_residual tests.test_memory_shift_integration -v
```

**18 tests passed**, including both CUDA cfg84 tests; none skipped.

Local CPU:

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
-m unittest tests.test_relation_legacy tests.test_relation_audit \
tests.test_relation_runner -v
```

**7 tests passed**. `git diff --check` passed.

The real saved Flat `best_miss_0p0.pt` from the reference seed66 run was also
loaded into the current flag-off full model with `strict=True`: all keys matched
(13,509,793 parameters). This was a load check, not a baseline rerun.

Coverage:

- Real source at pre-change `7d56881` vs current flag-off: identical state keys,
  initial parameter values, RNG state, train/eval hidden output; strict load works.
- Flag-on zero-init: full-model logits exactly equal Flat in train and eval,
  CPU small model and CUDA actual cfg84 dimensions. Dropout stream preserved.
- First valid utterance and padding remain zero even after nonzero output bias.
- Inactive Gap and unused backward half tolerate NaN poison without output or
  gradient leakage; pairwise equation matches independent computation.
- Three Adam steps: finite gradients and updates in relation branch, original
  adapter, Local Skip and query projector; no freezing.
- Changed residual leaves Memory reads/writes equal at identical upstream weights;
  branch receives post-ablation Base/Gap rather than raw contexts.
- Pairwise/control added parameter counts differ by only114 (<0.1%).
- Fixed offline audit baseline-as-new produces zero corrections/harms/deltas,
  reproduces 81.06809539495711% eight-rate W-F1 and4824 fixed-group exposures.

These tests establish implementation properties, not the research hypothesis.

After the completed run, the combined local relation/legacy/runner/audit/shift
suite ran19 tests:18 passed,1 CUDA-only test skipped locally (that test passed on
the target V100). Saved records were additionally checked for100 sequential epochs,
200 optimizer steps and200 forwards, zero JEPA/EMA steps, exact preservation of
every original config field, eight matching mask hashes,686 valid/655
history-supported test positions per rate, and matching implementation SHA256s.
All13,789,441 model parameters are trainable. The eight saved BEST checkpoints
exist remotely and their collected predictions reproduce saved W-F1 metrics.
