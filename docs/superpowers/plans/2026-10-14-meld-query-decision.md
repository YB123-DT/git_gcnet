# MELD frozen Query versus Decision implementation plan

> AI implementers: use tests-first and bounded subagent implementation followed
> by separate specification and code-quality review. The user approved this
> diagnostic; proceed without another approval round.

Goal: fit lightweight decision heads on original reads, then freeze each fitted
head and train only a query corrector, with Local and cross-conversation donor
controls. Original CHM and its source weights never train.

Architecture: freeze seed66 original1600 Flat eight-rate checkpoints. Capture
pre-target-write state, original Local, queries, observed keys and hard masks.
Read operations for corrected queries remain differentiable; frozen state and
keys do not. Do not reuse the no_grad evaluation read helper for training.

Stack: existing PyTorch1.13.1, NumPy and sklearn in biggpu cmerc-repro; no new
dependency, no Optuna search. All new source files live in
`experiments/osram_meld_query_decision_20261014/`; no backbone edits.

Execution record2026-10-14: all planned implementation, tests, bounded GPU
smoke, push/immutable deployment and eight-rate runs completed. The checklist
below preserves the initial plan; completion evidence is in VERIFICATION.md,
LAUNCH.json, runs/*.json and INTERPRETATION.md. Forty fits completed, all frozen
hash checks passed. No real-history advantage beyond Local was established;
no automatic multi-seed or D-control expansion was performed.

## Fixed protocol

- Source model remains original GCNet-mask cyclic-trained no-JEPA Flat; this
  experiment is an OFFLINE FROZEN PROBE, not another50epoch backbone run.
- For each checkpoint/rate, construct one fixed training view using the exact
  GCNet random_mask algorithm (host/guest padded rectangle and original speaker
  selector). Keep the same view for all diagnostic fits. This differs from
  end-to-end cyclic training and is recorded explicitly, not hidden.
- Test uses exact winning-epoch saved masks in `exact/source_0`, never new
  samples or the baseline root's different final evaluation masks.
- Head: input Local256 + Base512 +3Gap512 +availability3; LayerNorm,
  Linear64, GELU, Linear7. Fit train-only normalizer; same head dimensions,
  initialization seed, sample order and budget for real/Local/donor variants.
- Query correction: shared MLP128, type/head embeddings16, normalized
  original query + zero-initialized delta; condition only on current Local,
  original query, availability and role/head identity. No state input to MLP.
  Preserve original Gap residualization/ridge and exact inactive masks.
- B_real, B_local (zero history), B_donor. Then C_real with frozen B_real
  decoder and C_donor with frozen B_donor decoder. A is untrained reference.
  Donors use current availability/history-presence matches from a different
  conversation within the SAME split; matching never uses labels. Nonmatches
  are counted and all comparisons use the same eligible population.
- Thirty epochs each stage, Adam1e-3, weight_decay1e-5, batch256, clip1,
  source/probe seed66. Task loss only CE. Test-best ACC selection is internal;
  UA/W-F1 evaluated at that same point, without threshold search.
- Report per-rate ACC/UA/W-F1/CE, A→B and B→C changes, corrections/harms,
  current availability, donor coverage, Query gradient/perturbation diagnostics,
  frozen source/decoder hashes, wall time and parameter counts.
- Cache full states only as run-owned temporary files when needed. Preserve
  source weights, exact train/test masks, donor identities, probe/query weights,
  predictions and provenance. Can reconstruct states from source weights and
  recorded masks; do not retain13GB of temporary state caches indefinitely.

## Files and interfaces

- `learning.py`: `forward_reads(memory,queries,keys,availability,read_ridge,
  residual_strength) -> (base,gap)`; `QueryCorrector.forward(local,queries,a)`;
  `LightReadout.forward(local,base,gap,a)` and train-only normalizer.
- `trace.py`: `extract_split(model,loader,config,dimensions,rate,split,
  saved=None,mask_seed=66,max_batches=None) -> (arrays,checks)`; snapshots are
  original pre-write states. Output fields: local, memory, queries, keys,
  availability, labels, conversation_ids, utterance_indices, has_history,
  history_seen, logits, base, gap. Donor matcher returns indices and included.
- `run.py`: one immutable rate job, stage fitting, predictions, checkpoints,
  source restoration/mask parity, per-job SUMMARY and failures.
- `analyze.py`: assemble only complete jobs; distinguish full A reference from
  any common donor-matched subset; no comparison to three-seed60.693 as though
  it were single-seed66 performance.
- `tests/test_meld_query_decision_learning.py`,
  `tests/test_meld_query_decision_trace.py`,
  `tests/test_meld_query_decision_runner.py`: scoped regression tests.

## Task1: differentiable read and trainable modules

- [ ] Write a test comparing `forward_reads` against actual original `_scan`
  captured outputs, including residualized Gap and original masked keys.
- [ ] Run it and observe missing-function failure before implementation.
- [ ] Implement only read algebra, MLP query correction and lightweight head.
- [ ] Test zero-delta start, source memory/key detachment, finite nonzero query
  gradients on nonempty history, zero influence for empty history/inactive Gap.
- [ ] Fit head normalizer from training arrays only; verify decoder freezing
  still allows differentiation with respect to its read inputs.

## Task2: frozen trace and donor controls

- [ ] Write tests for batch-major flattening, strict pre-write state parity,
  original GCNet host/guest RNG agreement and no valid all-missing rows.
- [ ] Implement captures using existing WriteCapture/TaskCapture.
- [ ] Verify state/parameter hashes unchanged and test prediction/label parity.
- [ ] Match donors without labels and never within the same conversation;
  preserve availability/history presence and count any unmatched samples.

## Task3: runner and experiment orchestration

- [ ] Test optimizer ownership, fixed B decoder in C, paired population and
  failure when test masks/weights do not match the canonical source.
- [ ] Implement five fits per rate (three heads, two query-only fits) with
  identical stage budgets and only training-label gradients.
- [ ] Save best and latest stage state, normalized inputs/decoder configuration,
  optimizer/RNG and predictions; log valid query gradient coverage.
- [ ] Keep A→B and B→C as paired comparisons on each job's exact population.
- [ ] Run bounded real rate0.7 train/test extraction and both-stage GPU smoke.
- [ ] Review source/mask semantics first, then code quality; fix concrete issues.
- [ ] Commit/push to YB repository using scoped YB credentials, create immutable
  biggpu worktree, and launch eight independent rate jobs on healthy GPU(s),
  conservatively parallel, preserving existing unrelated processes.
- [ ] Assemble results and keep failures/missing jobs visible. No automatic
  multi-seed extension, D raw-history fit or new architecture search.
