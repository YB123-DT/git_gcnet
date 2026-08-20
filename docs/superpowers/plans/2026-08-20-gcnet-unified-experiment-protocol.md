# GCNet Unified Experiment Protocol Implementation Plan

> **For AI agents:** Required sub-skill: use superpowers:subagent-driven-development or superpowers:executing-plans to implement each task. Track progress with these checkboxes.

**Goal:** Make Baseline and JEPA experiments leakage-free, paired, reproducible, and auditable across IEMOCAP-Four, IEMOCAP-Six, CMU-MOSI, and CMU-MOSEI.

**Architecture:** Keep the legacy GCNet model and feature loaders, but move seed/order control, conversation-keyed masks, splits, shared-state hashing, and manifests into focused modules. The trainer consumes those interfaces, validates paired invariants, selects checkpoints only on validation, and evaluates test once.

**Technical stack:** Python 3.8+, PyTorch 1.8, PyG 2.0.1, NumPy, scikit-learn, unittest, JSON/YAML environment manifests.

---

## File responsibilities

- Create `gcnet_modality_jepa/protocol.py`: seed bundle, stable hashes, epoch-aware sampler.
- Create `gcnet_modality_jepa/mask_schedule.py`: conversation-keyed deterministic masks and realized-rate statistics.
- Create `gcnet_modality_jepa/splits.py`: official MOSI/MOSEI splits and leakage-free IEMOCAP train/validation/test indices.
- Create `gcnet_modality_jepa/run_manifest.py`: environment, feature, split, mask, sampler, and initialization evidence.
- Create `gcnet_modality_jepa/shared_state.py`: shared parameter selection, checkpoint loading, and hashing.
- Modify `gcnet_modality_jepa/loss.py`: missing-count normalized reconstruction.
- Modify `gcnet_modality_jepa/model.py`: common training-only stability decoder.
- Modify `gcnet_jepa_replacement/model.py`: expose the same stability decoder without restoring original GCNet reconstruction.
- Modify `gcnet_modality_jepa/train_gcnet.py`: consume the unified protocol and test once.
- Create focused tests under `tests/` for every protocol boundary.
- Create `scripts/audit_paired_runs.py`: reject unfair or mismatched paired manifests.
- Update `environments/gcnet-official/environment.yml` and experiment documentation.

### Task 1: Isolate random streams and sample order

**Files:**
- Create: `gcnet_modality_jepa/protocol.py`
- Create: `tests/test_protocol_rng.py`

- [ ] Write tests that instantiate identical `SeedBundle` objects and assert equal component seeds, different component names produce different seeds, and `EpochSeededSubsetSampler(seed=66)` returns identical epoch-3 orders after unrelated `torch.rand` calls.
- [ ] Run `python -m unittest tests.test_protocol_rng -v`; expect import failure for the missing module.
- [ ] Implement SHA-256-derived 31-bit component seeds and a sampler whose `__iter__` creates a local `torch.Generator` from `seed + epoch`.
- [ ] Run `python -m unittest tests.test_protocol_rng -v`; expect all tests to pass.

### Task 2: Generate conversation-keyed mask schedules

**Files:**
- Create: `gcnet_modality_jepa/mask_schedule.py`
- Create: `tests/test_mask_schedule.py`

- [ ] Write tests proving the same conversation/rate/split/epoch produces identical masks independent of batch order; train epoch 1 differs from epoch 2; validation/test ignore the requested epoch and remain fixed; every real utterance keeps at least one modality; padding is marked separately; requested and realized rates are both reported.
- [ ] Run `python -m unittest tests.test_mask_schedule -v`; expect import failure.
- [ ] Implement `ConversationMaskSchedule` using local NumPy generators derived from stable conversation keys, with separate host/guest keys and no global RNG mutation.
- [ ] Run `python -m unittest tests.test_mask_schedule -v`; expect all tests to pass.

### Task 3: Make dataset splits leakage-free

**Files:**
- Create: `gcnet_modality_jepa/splits.py`
- Create: `tests/test_protocol_splits.py`
- Modify: `gcnet_modality_jepa/train_gcnet.py`

- [ ] Write synthetic five-session tests proving the held-out IEMOCAP session appears only in test, validation comes from non-test conversations, all splits are nonempty and disjoint, and repeated seeds reproduce the same split hash. Add MOSI/MOSEI tests that preserve supplied official sets exactly.
- [ ] Run `python -m unittest tests.test_protocol_splits -v`; expect import failure.
- [ ] Implement deterministic conversation-level IEMOCAP validation allocation with label-distribution-aware greedy assignment and explicit overlap checks; implement direct official split mapping for MOSI/MOSEI.
- [ ] Replace `return train_loaders, test_loaders, test_loaders` with loaders built from the three index sets and `EpochSeededSubsetSampler`.
- [ ] Run split tests plus `tests.test_no_leakage` if present; expect all to pass.

### Task 4: Normalize missing reconstruction by supervised elements

**Files:**
- Modify: `gcnet_modality_jepa/loss.py`
- Create: `tests/test_reconstruction_normalization.py`

- [ ] Write tests where one missing error and two duplicated equal missing errors return the same loss; observed and padded values do not contribute; no missing elements returns finite graph-connected zero with zero gradient.
- [ ] Run `python -m unittest tests.test_reconstruction_normalization -v`; expect failures under the old all-utterance denominator.
- [ ] Implement per-modality sums divided by the count of missing real feature elements, then average only modalities with a nonempty selection.
- [ ] Run the new tests and existing loss tests; expect all to pass.

### Task 5: Give both variants the same stability decoder

**Files:**
- Modify: `gcnet_modality_jepa/model.py`
- Modify: `gcnet_jepa_replacement/model.py`
- Modify: `gcnet_modality_jepa/train_gcnet.py`
- Create: `tests/test_common_stability_path.py`

- [ ] Write tests asserting add-on and replacement variants instantiate an equal-shaped `stability_rec_head` only when enabled, return equal output shapes from identical hidden tensors, and do not call the stability path in evaluation.
- [ ] Run `python -m unittest tests.test_common_stability_path -v`; expect failure because replacement currently rejects the stabilizer.
- [ ] Add an optional stability decoder to the shared `GraphModel`; use it only on the deterministic auxiliary masked view during training. Keep replacement's original `linear_rec` absent.
- [ ] Remove the parser rejection for replacement plus stability and replace it with an assertion that paired variants use equal rate and weight.
- [ ] Run stability, replacement, and miss-zero parity tests; expect all to pass.

### Task 6: Prove shared initialization parity

**Files:**
- Create: `gcnet_modality_jepa/shared_state.py`
- Create: `tests/test_shared_state.py`
- Modify: `gcnet_modality_jepa/train_gcnet.py`

- [ ] Write tests proving encoder/classifier keys are shared, reconstruction/predictor/stability heads are excluded, equal shared tensors yield equal SHA-256 hashes, and a mismatched tensor raises a parity error.
- [ ] Run `python -m unittest tests.test_shared_state -v`; expect import failure.
- [ ] Implement shared-state extraction, atomic checkpoint save/load, and manifest hash generation. Add `--shared-init-checkpoint` and `--require-shared-init-hash` CLI options.
- [ ] Run shared-state and existing miss-zero parity tests; expect all to pass.

### Task 7: Evaluate test exactly once

**Files:**
- Modify: `gcnet_modality_jepa/train_gcnet.py`
- Create: `tests/test_evaluation_lifecycle.py`

- [ ] Write a fake-loop test counting train, validation, and test calls and asserting `epochs` train calls, `epochs` validation calls, and one post-restore test call. Assert no test metric is available to checkpoint selection.
- [ ] Run `python -m unittest tests.test_evaluation_lifecycle -v`; expect failure because the current loop tests every epoch.
- [ ] Extract a `run_training_fold` lifecycle that stores the best validation checkpoint, restores it, and invokes test once with the fixed evaluation mask schedule.
- [ ] Run lifecycle and best-checkpoint replay tests; expect all to pass.

### Task 8: Record and audit complete run manifests

**Files:**
- Create: `gcnet_modality_jepa/run_manifest.py`
- Create: `scripts/audit_paired_runs.py`
- Create: `tests/test_run_manifest.py`
- Modify: `gcnet_modality_jepa/train_gcnet.py`

- [ ] Write tests asserting manifests include environment versions, GPU/driver, command, git revision, feature hashes, split/mask/order/init hashes, seed bundle, requested/realized rates, model variant, stabilizer configuration, and test-call count. Write paired-audit tests that reject any invariant mismatch.
- [ ] Run `python -m unittest tests.test_run_manifest -v`; expect import failure.
- [ ] Implement manifest collection with streaming SHA-256 feature-directory metadata hashes and a paired-audit CLI returning nonzero on mismatch.
- [ ] Run manifest tests and a synthetic paired audit; expect all to pass.

### Task 9: Lock the formal environment and protocol commands

**Files:**
- Modify: `environments/gcnet-official/environment.yml`
- Modify: `environments/gcnet-official/README.md`
- Create: `configs/unified_protocol.json`
- Create: `docs/experiments/UNIFIED_PROTOCOL.md`

- [ ] Record exact Python/PyTorch/CUDA/PyG/NumPy/scikit-learn versions and the known nondeterministic CUDA scatter limitation.
- [ ] Add one canonical configuration containing seed derivation, validation ratio, stability rate/weight, checkpoint metric, requested rates, and collapse threshold.
- [ ] Document single-fold, full-fold, and paired Baseline/JEPA commands without environment-dependent defaults.
- [ ] Validate JSON/YAML parsing and scan documentation for conflicting protocols.

### Task 10: Four-dataset protocol smoke and final regression

**Files:**
- Create: `scripts/run_unified_protocol_smoke.py`
- Create: `tests/test_unified_smoke_matrix.py`
- Update: `docs/experiments/UNIFIED_PROTOCOL.md`

- [ ] Write a matrix test requiring one Baseline and one JEPA smoke job for each of IEMOCAP-Four, IEMOCAP-Six, CMU-MOSI, and CMU-MOSEI, with at most three jobs per GPU and GPU4 excluded.
- [ ] Run the matrix test; expect import failure for the missing runner.
- [ ] Implement resumable smoke orchestration, isolated output directories, command/status manifests, and paired-audit invocation after each pair.
- [ ] Run all unit tests in the canonical environment.
- [ ] Run two-epoch smokes for all eight jobs; require finite losses, disjoint split hashes, paired mask/order/init hashes, and one test call.
- [ ] Run `git diff --check` and inspect all generated manifests before reporting completion.
