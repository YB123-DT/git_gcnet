# Evidence-centered decision correction implementation plan

> Execute with subagent-driven-development and TDD in this existing worktree.
> User authorized implementation and launch with “开始跑”. Main owns verification,
> source snapshots, commits/pushes and launch; implementation agent owns core only.

Goal: replace Flat with the specified centered L -> LB -> LBG task outputs.
Stack: existing PyTorch/unittest; no new dependencies.

Fixed implementation: default-off osram_decision_correction boolean. New
EvidenceCenteredDecisionHead uses LayerNorm(input) -> Linear(input,128) -> GELU
-> Linear(128,n_classes) for H_L, D_B, D_G. No dropout or BatchNorm. D_B/D_G final
weights zero-init and final bias=False (a shared final bias always cancels).
H_L normal init. D_B input [L,Bf], D_G input [L,Bf,Gf_A,Gf_T,Gf_V,a]; f is the
causal forward half (num_heads*value_dim). Both centered calls differentiable.
Keep original read/write/local path and one scan, bypass original Flat/head.
Avoid unused parameters in optimizer and report retained vs active counts if any.

## Task 1: core and tests

Files: new gcnet_missing_m3/decision_correction.py; modify osram.py, model.py,
train_gcnet.py and experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py;
new tests/test_decision_correction.py and tests/test_decision_correction_training.py.

- [ ] Write failing tests for new flag and centered equations before implementation.
  Tests must assert exact dB=0 for B=0, dG=0 for G=0 with nonzero learned heads
  in train/eval, poison inactive Gap/padding, first-valid history, logits C=1/6.
- [ ] Implement the head and default-off model path, expose
  model.last_decision_outputs={'local':sL,'base':sLB,'full':sLBG,
  'delta_base':dB,'delta_gap':dG}. These are [L,B,C], with gradients attached.
  Compute one Memory trajectory; use emotion-ablation-aware contexts. Preserve
  default-off state_dict/output/RNG and reject incompatible modules/protocols.
- [ ] Route model output directly to sLBG; do not call old adapter/norm/head.
  Use safe padding and inactive Gap where masks and valid-prefix history masks.
- [ ] Train with exactly (_task_loss(sL)+_task_loss(sLB)+_task_loss(sLBG))/3,
  original task contracts and same valid samples. Inference uses full only.
  Record per-exit task losses/W-F1 and delta magnitudes, not reliability scores.
- [ ] Run finite-gradient/optimizer-update and strict checkpoint tests, legacy
  Flat exact regression, one-scan/no-Flat hooks and actual cfg84 CUDA smoke.
  Command: multimodalerc310/bin/python -m unittest tests.test_decision_correction
  tests.test_decision_correction_training -v. Expect all PASS.

## Task 2: runner and remote verification

Files: experiments/osram_decision_correction_20261003/run.py, RESULT.md;
tests/test_decision_correction_runner.py. Main owns these files.

- [ ] Add runner tests first: each seed's raw Flat config differs only by the new
  boolean; reject existing Gate/Relation/completion, wrong seeds/protocols.
- [ ] Reuse existing runner provenance/GPU checks; verify canonical test mask
  hashes, 100 epochs, eight BEST checkpoints. No metrics-based retrial/search.
- [ ] Create immutable remote code snapshot on biggpu. Exclude GPU4 always;
  prefer three concurrent seeds on free GPU0 after memory checks. GPU5 is a
  fallback healthy card. Preserve unrelated jobs and existing run outputs.
- [ ] Run new and old relevant tests on target V100; perform spec then quality
  review and resolve blocking findings before launch. Record test evidence.

## Task 3: authorized experiments and handoff

- [ ] Commit/push verified code with Lore trailers before launch. Validate remote
  source hashes against commit. Use raw original cfg84 configs for seeds66/67/68,
  each100 epochs; original cyclic random masks/loss primitive/optimizer/batch/
  per-rate BEST Test-oracle. Only new decision head and three-exit objective differ.
- [ ] Launch independent logs/output dirs with persistent processes; verify PID,
  first epochs and checkpoints. Record config/env/data/reference hashes/code/GPU.
- [ ] Commit/push launch records. Report running vs completed honestly. Completed
  results: per-rate W-F1, eight-rate and high(.5/.6/.7) means vs original Flat;
  all INTERNAL DIAGNOSTIC ONLY, not formal validation-selected paper scores.

Unit sketch (required assertions, not a substitute for executable tests):
```python
out = head(local, base, gap, availability, umask)
assert torch.equal(out['full'], out['base'] + out['delta_gap'])
assert torch.count_nonzero(out['delta_gap'][all_observed]) == 0
assert torch.count_nonzero(out['delta_base'][first_valid]) == 0
assert torch.count_nonzero(out['full'][padding]) == 0
loss = sum(task(out[k], labels, umask) for k in ('local','base','full')) / 3
loss.backward()
```
