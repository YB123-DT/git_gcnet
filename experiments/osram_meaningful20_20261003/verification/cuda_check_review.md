# Independent CUDA checker review

Date: 2026-10-03. Final verdict after independent rereview: **APPROVED for the
authorized CUDA feasibility run; all three initial findings resolved**.
This is approval of the checker, not a claim that CUDA feasibility has passed.
Read-only review of checker, tests, and directly relevant loader/config/optimizer/
mask/initialization helpers. Only this audit was written; no CUDA allocation,
training, core edit, or Git action was performed by the reviewer.

Initial reviewed hashes (findings below refer to this version):

- `cuda_check.py`: `fc9520882787968b39ea1769fca647ef56b05ce40424712173af8f9038eb0a71`
- `test_meaningful_cuda_check.py`: `6cff38db8985fc34f4de294bc94fe38e1f98a50d11c7b026fab646eee594ff60`

## Initial findings sent immediately to coordinator (now resolved)

1. **HIGH — label/split root is not bound to the supplied manifest before import.**
   `cuda_check.py:45` imports Torch, then trainer/config at lines 46–49; only line 52
   validates feature roots, and no dataset-root environment variable is set.
   `gcnet_modality_jepa/train_gcnet.py:180` reads labels/splits through
   `config.PATH_TO_LABEL`, which is fixed at import from `GCNET_DATASET_ROOT` or
   the snapshot's default `dataset/` directory (`config.py:19`). Supplied feature
   roots do not override this. An otherwise valid standalone invocation can fail
   in a snapshot or mix hashed features with unrelated labels/splits. Bind and
   validate the dataset root before any trainer/config import, and cover the
   actual label/split file in the hashed manifest. An externally guaranteed exact
   environment can avoid failure, but the checker currently neither requires nor
   verifies that guarantee.

2. **MEDIUM — runtime NODE initialization is not explicitly verified.**
   Lines 107–117 check finite loss and any present gradients but never assert
   the three data-initialization buffers become true or that initialized threshold/
   log-temperature tensors are finite. NODE only initializes in training mode
   (`meaningful_blocks_feature_reasoning.py:152`); the preceding identity check is
   eval-only. Require post-first-forward initialized-state/finite-state assertions
   before reporting this readiness requirement as satisfied. Current helper tests
   do not protect this requirement.

3. **MEDIUM — baseline scan mock survives the profiling reset.**
   `bscan` at line 84 retains its wrapped bound method after its patch context ends.
   `del baseline` at line 102 therefore does not release the baseline OSRAM owner
   and its tensors before line 103 resets peak counters. The unused baseline
   footprint is counted as candidate train/eval memory. Delete temporary scan
   mocks before collection/reset. This is conservative rather than an unsafe
   underestimate, but can cause false resource rejection and invalid comparison.

   A CPU weak-reference probe reproduced the lifetime issue: after deleting the
   owner, `weakref()` remained non-null; after deleting the mock it became null.

## Checks that passed inspection

- The host index and UUID mapping are queried and checked before importing Torch.
  `validate_gpu` rejects physical index 4, its UUID under another index, unknown
  indices, and mapping mismatch; visibility is restricted to exactly the audited
  UUID before the first CUDA availability/allocation operation.
- `candidate_config` checks baseline batch32, Adam, learning rate, weight decay,
  constant schedule, official masks, and the permitted architecture settings.
  The loader receives batch32 and verifies 32 conversation IDs. Adam parameter
  groups come from the same trainer helper; weight decay and gradient clipping
  retain the effective config. No formal config is mutated to make smoke cheaper.
- Three updates on a fixed .7 training mask are explicitly disposable feasibility
  steps, not a replacement for the cyclic 100-epoch run. The checker writes only
  to a newly created output directory. It does not launch the formal runner.
- Zero bridge predictions are compared exactly; one `_scan` call is asserted for
  baseline and candidate. Shared raw gradients are compared before optimizer or
  clipping. The gradient comparison uses a documented numerical tolerance, not
  bitwise equality.
- Training and all eight evaluation masks are inside the peak measurement window,
  and CUDA synchronization precedes the peak read. Strict reload is checked
  afterward, so the profile is a train/eval footprint, not a checkpoint-load peak.
  It covers one real batch, not the maximum over every conversation length.
- EA evaluation under both no_grad and inference_mode is actually invoked; its
  implementation locally disables inference mode and clones temporary tensors.
- Evaluation masks are applied to training conversations; the output explicitly
  calls this feasibility only and excludes task scores/full epoch/full resume.

## Verification executed

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_cuda_check.py -v
```

Result: **3 tests passed in 0.056 s**. These cover parser identity arguments,
GPU4/UUID rejection helper, and retained-artifact budget only. They do not exercise
`profile()` or establish actual CUDA feasibility. No GPU execution was performed
in this review. The subsequent correction review is recorded below.

## Correction rereview and final decision

Independently reread the actual updated implementation and its newly landed
`run.bind_data_environment` dependency; did not rely on the fix author's summary.

1. **Root binding fixed.** `cuda_check.py:65–66` invokes the stdlib-only helper
   before importing Torch/trainer/config. `run.py:85–108` now requires an explicit
   absolute dataset root, feature roots inside it, the canonical CMUMOSI
   label/split pickle in `split_files`, and matching hashes for that file and all
   feature files. Lines 112–125 bind the root and fail closed if an already-imported
   config points to a different label file. Thus stale inherited environment does
   not silently control labels/splits.
2. **NODE initialization/finite state fixed.** `check_finite_state` checks every
   floating/complex state-dict tensor, plus exactly three initialized NODE tree
   layers. The checker invokes it after each training forward and update and after
   strict reload. It is not incorrectly invoked on uninitialized eval-only NODE
   before the first training forward.
3. **Peak-window retention fixed.** Both scan mocks are explicitly deleted before
   baseline deletion, collection, CUDA cache release, and peak-stat reset.

Fresh verification:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_cuda_check.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful20_infrastructure.py -v
```

Results: **6 checker tests passed (1.392 s)** and **17 infrastructure tests passed
(0.073 s)**. New regressions cover import ordering, missing/uninitialized/nonfinite
NODE state, mock deletion ordering, wrong inherited dataset root, missing canonical
split declaration, and changed label-file content. The earlier numerical/core
checks and feasibility-only reporting boundaries remain unchanged.

Final rereviewed hashes:

- `cuda_check.py`: `2231da7c63272706b81e8a2e5cbe7bf4e041a6d4cc3191f16293297692600717`
- `test_meaningful_cuda_check.py`: `6ee54833bc3c0ad505e78204de4815b957e253c0d312c0a7cf79e3698c98c262`
- `run.py`: `43cad35f3b9974ba686cfedd1b7140a14d96460869a5800a9cb49bf81a50c515`

No remaining blocker found in this bounded review. Actual CUDA execution, measured
GPU memory, full-run stability, and task scores are still outside this review's
evidence; the checker must run successfully on the approved healthy GPU to establish
CUDA readiness.
