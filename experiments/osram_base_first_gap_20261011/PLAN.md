# Base-first Gap-correction: approved single-seed plan

User approved the displayed two-stage hidden-representation design, MOSI seed66,
150 epochs and per-rate Test-oracle selection. INTERNAL DIAGNOSTIC ONLY.
No new candidates, multi-seed expansion, MC-FWC restart, or GitHub push.

u = original_skip(L) + F_B([L,B]); delta = F_G([LN(u),masked G_A/G_T/G_V]);
h = original_emotion_norm(u + delta), followed by unchanged task head.
Both MLPs use LN(input), Linear, GELU, original dropout, Linear(output_dim).
Their common hidden width is chosen by nearest parameter-count match to the
removed original adapter, not by a score sweep. Both final layers start at zero
(F_B follows original adapter initialization; F_G starts with zero correction).
Keep context_dim interface and fixed Gap slots; do not alter CHM reads/writes.
No history or no active Gap forces delta=0 after output bias, including after
training. Padding and inactive inputs use safe torch.where. This is one-stage
joint training and one task loss, not hierarchical decision supervision.

- [ ] Tests-first module and config/default-off integration tests.
- [ ] Implement separate module and boolean flag, preserve old path/RNG.
- [ ] Reuse existing audited 100@.001 +50@.0001 full-state runner; verify cfg84
  reference, hashes, original test masks, one seed and no extra modifications.
- [ ] Run targeted CPU tests and one real MOSI GPU batch forward/backward/update.
- [ ] Commit only scoped files; seal deployed source, launch one job on healthy
  biggpu GPU1 with UUID verification, independent logs and checkpoints.
- [ ] Verify first epochs and record status; final summary generated only after
  full150 completion and eight per-rate BEST/prediction checks.

Owned files: gcnet_missing_m3/base_first_gap.py and integration in osram.py,
model.py, train_gcnet.py; tests/test_base_first_gap.py; this experiment directory
and tests/test_base_first_gap_runner.py. Preserve unrelated gcnet/model.py edit.
