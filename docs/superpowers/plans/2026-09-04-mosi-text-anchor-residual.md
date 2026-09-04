# MOSI Text-Anchor Residual Fusion Implementation Plan

> **面向 AI 代理的工作者：** 使用 TDD 在当前隔离 worktree 内执行。

**目标：** 实现并运行 Text-anchor bounded-residual 对 Slot Missing-M3 的五种子 MOSI 判别实验。

**架构：** 保留原 Slot fusion 作为公共计算和 Text 缺失时的 fallback；Text 存在时用同一 fusion 权重产生 T-only anchor，并加入零初始化、相对范数有界的 residual。训练器只新增一个 `fusion_type` 选项。

**技术栈：** PyTorch、现有 GCNet/Missing-M3 trainer、pytest、biggpu V100。

---

### Task 1: Lock fusion behavior with failing tests

**Files:**
- Modify: `tests/test_missing_m3.py`

- [ ] Add a test that requests `text-anchor-residual` and proves the current
  implementation rejects it.
- [ ] Add expected-behavior tests for exact T preservation, unchanged AV
  fallback, missing-value isolation, and the residual norm cap.
- [ ] Run only these tests remotely and confirm failure is caused by the
  missing fusion type.

### Task 2: Implement the minimum fusion variant

**Files:**
- Modify: `gcnet_missing_m3/model.py`
- Modify: `gcnet_missing_m3/train_gcnet.py`

- [ ] Extend `ObservedSetEncoder` with `text-anchor-residual`.
- [ ] Reuse the existing projectors, embeddings, and Slot fusion weights.
- [ ] Add one zero-initialized residual MLP under a forked RNG context.
- [ ] Preserve current outputs and state keys for `mean` and `slot`.
- [ ] Expose the new CLI choice and run the focused tests to green.

### Task 3: Run the locked five-seed screen

**Files:**
- Create: `experiments/missing_m3_mosi_text_anchor_residual_20260904/RESULT.md`
- Create: `experiments/missing_m3_mosi_text_anchor_residual_20260904/results/`

- [ ] Synchronize only the changed source and tests to the existing remote
  mirror.
- [ ] Run seeds 66--70 on healthy GPUs with the locked configuration.
- [ ] Pull metrics and prediction artifacts, recompute W-F1, and compare with
  inherited Slot results.
- [ ] Record the causal verdict, verification evidence, runtime, and known
  limitations; commit and push using the Lore protocol.

