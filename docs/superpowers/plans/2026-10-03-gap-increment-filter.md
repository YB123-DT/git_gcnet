# Gap increment filter implementation plan

> Execute with subagent-driven-development; approved placement is recorded in the
> adjacent specification. Stay in existing OSRAM worktree; preserve unrelated edits.

Goal: filter the original Flat Gap increment without modifying Memory or losses.
Architecture: single scalar identity-initialized gate around task-space Gap delta.
Stack: existing PyTorch and unittest; no dependencies.

- [ ] Write failing tests in tests/test_gap_increment_filter.py for default-off,
  identity eval/train, exact shared RNG/dropout, mask safety and one scan.
  Run with multimodalerc310 python -m unittest tests.test_gap_increment_filter -v.
- [ ] Implement GapIncrementFilter and helper in gcnet_missing_m3/osram.py;
  propagate default-off flag in model.py/train_gcnet.py and evaluation builder.
  Use original adapter/norm/head, `uF + (g-1)*(adapterF-adapterB)`; zero output
  layer and fork initialization; fork base replay RNG; do not detach gradients.
- [ ] Test incompatibility guards, CLI/config wiring, gradient finite and actual
  forward/backward updates; add diagnostics to existing epoch/eval accumulators.
- [ ] Review specification conformity then code quality. Run old Flat regressions
  and actualcfg84 CUDA tests in isolated biggpu snapshot before formal launch.
- [ ] Add experiments/osram_gap_increment_filter_20261003/run.py using raw original
  Flat seed66 config plus osram_gap_increment_filter=True only. Record source hashes,
  environment/config/parameters; validate all8 canonical evaluation masks at finish.
- [ ] Commit/push verified implementation; launch one100epoch seed66 run on GPU5,
  track PID/log and report actual running state, then collect results if completed.
  Never conflate launched with complete or expand seeds automatically.
