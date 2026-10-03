# Test specification: meaningful twenty-block loop

Status: sequential Architect/Critic review passed; read together with `prd-osram-meaningful20-loop.md`. Tests below are required checks, **not claims of passing**. Each candidate's verified evidence gates training. No new test or runtime dependency is authorized.

## Test fixtures and evidence

Use the historical Flat reference in `experiments/osram_current_history_relation_20261003/reference/`. Keep its raw JSON immutable. Build common deterministic synthetic fixtures plus one existing real MOSI batch with the baseline batch size/protocol, using a healthy biggpu UUID only. Synthetic fixtures cover all seven nonempty availability patterns, leading/interior/trailing padding, length-one conversations, empty padded columns, zero features, identical head values and large finite values. Test noncontiguous tensors because Flat GEMM layout previously required explicit contiguous handling (`osram.py:1678`).

Evidence lives in the new experiment directory: source/design acceptance ledger, `VERIFICATION.json`, per-candidate correctness/source-audit records, CPU/CUDA parity summaries, parameter/resource profiles, frozen round manifest, per-run provenance and source/config hashes, queue event/state records, attempt/failure ledger, all-rate result tables and fixed promotion decisions. Do not store tensors, datasets or credentials in Git.

## A. Source and registry acceptance

1. Parse the heterogeneous card schemas (`accepted`, `candidates`, `cards`, `papers[].accept_or_reject`) without treating `rejected` papers as accepted. Exactly twenty unique IDs, distinct full core chains, non-MSA/MERC domains and source mappings must be accepted before first training dispatch. Require inspected primary-code pins and explicit source adaptations/license handling; Equilibrium Aggregation has a documented primary publisher-code-listing exception and no fabricated Git pin.
2. Duplicate test: generic Set Transformer/PMA/AllSet aliases, renamed basic gates, width/depth variants and an earlier round's mechanism must fail the quota validator. Rejected DDN/iSQRT and all withdrawn basic-operator IDs are not eligible.
3. Every accepted ID resolves to one complete factory/core and matching fixed design hash; unknown IDs fail immediately. An accepted card lacking implementation remains awaiting-implementation, not ready. A code-ready candidate cannot launch while the twenty-card gate is incomplete.
4. API contract tests instantiate `build_<family>(method, latent_dim, num_heads, value_dim)`, require integer `output_dim`, and call `forward(local[N,D], evidence[N,4,H*V], active[N,4], availability[N,3])`. The wrapper removes padding/no-history and safely zeros inactive evidence before the call; each family owns projections and respects Base/Gap-A/Gap-T/Gap-V order. Assert TabNet output64 and NODE192, grouping64 processing and other card-specific128 interfaces; do not replace all tokenizers with one shared projection. Empty eligible batches bypass the family.
5. Audit distinguishing equations with small independently computed fixtures, not merely output shape. Examples: RRN shared iterative state; EGT node and edge updates; hypergraph two-direction incidence operations; sheaf pinned author-code operator; CRATE tied MSSA plus ISTA; Hamburger alternating NMF plus final differentiable update; TabNet prior/masked sparse selection; NODE split-path products/dense tree connections; Equilibrium Aggregation's nonnegative potential/cardinality scaling/Nesterov lookahead and per-example threshold. Each remaining card supplies its own equivalent invariant before ready status.

## B. Default-off and zero-start regression

Proposed suite: `tests/test_meaningful_blocks.py` and `tests/test_meaningful_block_integration.py`; extend focused existing tests rather than duplicating fixtures.

1. With flag absent or `none`, same seed produces identical original state-dict key set/tensors, parameter count, next Python/NumPy/Torch random draws and legacy outputs. Compare against a frozen pre-change implementation fixture/source snapshot; comparing two copies of modified code is insufficient.
2. Instantiate each of twenty branches under RNG isolation. Its original shared parameters and subsequent RNG streams must equal `none` exactly. Check CPU and every used CUDA generator. Original dropout consumption is unchanged in training mode.
3. At zero bridge initialization, on finite valid input assert exact original Flat/head output equality (`torch.equal`) on CPU/CUDA for matched dtype/layout. If an actual backend creates unavoidable rounding differences, stop and document the cause for Architect review before substituting a tolerance; do not silently relax parity. Original-parameter **raw gradients before clipping** must match (CPU float64 `atol=1e-10, rtol=1e-8`; CUDA float32 `atol=1e-6, rtol=1e-5`). Optimizer-update parity applies to the OFF path, an explicitly unclipped toy shared-parameter step, and full-resume controls, not enabled training against Flat: baseline global `clip_norm=1` includes the new bridge gradients, so its norm and resulting original-parameter updates may legitimately differ. Preserve the existing clipping rule; this change in the global gradient norm is an allowed training effect, not justification to alter the protocol.
4. Store/reload default-off checkpoints with strict loading; enabled checkpoints round-trip every core parameter/buffer and initialization flag. Missing/unexpected branch keys fail rather than being swallowed. A bare legacy checkpoint is valid only for `none`, unless an explicit initialization operation is separately identified; no new training is secretly warm-started.
5. Configuration tests assert rejection of all incompatible readout/fusion/paired-view/completion/auxiliary/query/write interventions and noncausal/head-layout variants. CLI, TrainConfig and direct model construction enforce the same constraints. Old withdrawn runner still rejects launch.
6. Rebuild one enabled saved checkpoint through `experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py::_build_model`, verifying the new flag reaches the model and strict loading succeeds. The root owns only that keyword-plumbing edit. Assert the new queue never imports/inherits this runner's historical `GPUS=(4,5,7)` policy.

## C. Mask, causal and state correctness for every block

1. For seven availability patterns, perturb inactive Gap values with finite extremes, NaN and Inf; finite outputs/gradients on active evidence must be identical to safely zeroed controls. Repeat poison injection into padding and unused backward halves. Mask before normalization/projection; multiplication by zero after NaN is not sufficient.
2. Residual is exact zero on all padded rows and the **first valid** utterance after any leading padding, even after bridge biases learn nonzero values. Forward hooks verify the core is not invoked for those rows. All-valid length-one batch skips the core completely.
3. At rate zero, all Gap branches are excluded; Base/Local may still be processed after first history. Inactive Gap output and gradients remain zero through biased layers, embeddings, normalization, graph/hyperedge reductions and sparse selectors. Valid empty-history/empty-mask cases produce no all-masked softmax, divide-by-zero or NaN.
4. Count the original OSRAM `_scan` and memory read/write/query calls with hooks before/after; counts and returned Base/Gap values must match exactly for identical upstream states. No additional call/cache/state is introduced. Static inspection confirms branch inputs contain no labels or future data.
5. Perturb only future utterances and check earlier representations/predictions unchanged; compare one conversation alone and concatenated with unrelated conversations in evaluation mode. Cores have no cross-utterance persistent state or BatchNorm/normalization across examples. Do not claim training-mode dropout invariance under arbitrary batch reorder.
6. Track named buffers/state before and after repeated eval calls. Parameters/buffers stay unchanged except explicit read-only diagnostic outputs. Hamburger's basis buffer is immutable and inferred factors reset; NODE starts with finite thresholds0/log-temperatures0 and `initialized=False`, permits fresh zero-bridge eval with no mutation, and never relies on NaN placeholders times zero. Its first eligible normal training batch replaces provisional values; eval/test cannot initialize it, and no eligible row defers initialization. Save the initialized flag. NODE construction/initialization preserves enclosing RNG and uses no extra OSRAM pass or optimizer step.
7. Slot Attention training uses only a dedicated CPU generator whose experiment-seed-derived state is a persistent model buffer. Assert consecutive training epsilons differ as specified, global CPU/CUDA RNG snapshots remain identical across branch draws, and save/reload reproduces the exact next epsilon. Evaluation reuses the fixed seed1729 epsilon buffer: repeat eval under both grad-enabled and no-grad modes and assert identical epsilon/output and no model/generator-buffer mutation. Changing batching cannot feed later evidence into earlier slots. Verify generator state stays valid after CPU/CUDA module transfer; never seed or draw from a global generator to recover it.

## D. Gradients and numeric mechanism tests

1. Test each core directly with a nontrivial safe objective so zero bridge output does not mask broken internal gradients. Assert finite loss/outputs and finite gradients for every parameter group expected to participate in the selected availability pattern; distinguish intentionally inactive groups.
2. In the integrated branch, step 1 should update the zero bridge while core gradients may be zero. Then run at least two more nondegenerate real task-loss steps and assert bridge and expected core parameter groups receive finite nonzero gradients and at least one tensor per expected group changes. Do not require all individual scalar entries to change.
3. Check CPU float64 finite differences/gradcheck where the core is smooth. For sparsemax/entmax, min/max, ReLU and branch boundaries, use fixtures away from ties/kinks and compare analytic paths only; no false claim of differentiability at a discontinuity. Include extreme-value/zero-input tests for multiplicative updates, sparse selectors, tree temperatures and normalization denominators.
4. CUDA forward/backward on actual protocol batch shape must pass with finite task loss. Profile train and evaluation peak allocated/reserved memory and wall time; gradients/optimizer state must exist during the peak measurement. Test every candidate, not one proxy family, before its training admission.
5. Equilibrium Aggregation: training derivatives propagate through all ten inner updates (`create_graph=True`) to the potential and learned inner scalars. Compare a tiny smooth finite-difference control away from the stopping threshold. Under both `no_grad` and `inference_mode`, evaluate through local `torch.inference_mode(False)` plus `torch.enable_grad()` and clone inference-created inputs inside that scope into normal tensors; assert finite output, no parameter `.grad` mutation and no retained outer graphs. Include inputs actually created inside inference mode, not only normal tensors passed through it. Verify no cross-example stopping decision, no persistent inner optimizer state, no source auxiliary term in the outer task loss, and honest peak-memory profiling of higher-order training.

## E. Config, data and immutable source audit

Proposed suites: `tests/test_meaningful20_runner.py`, `tests/test_meaningful20_summary.py`.

1. Raw config equality: `candidate_raw - {'osram_meaningful_block'}` exactly equals the same-seed baseline raw JSON. Do not drop keys, reconstruct a reduced subset or override baseline defaults. Effective-schema comparison allows only the block field; baseline-absent defaults are listed separately. Seeds 67/68 change only seed plus the block field relative to the registered seed-66 protocol.
2. Data paths/version/hash, train/test split, preprocessing, feature dimensions, task loss, mask generation, training-rate schedule, batch32, epochs100, Adam .001/wd1e-5, selection policy and expected test sample counts match baseline. Compare canonical mask-row multisets/hashes using the existing project's established validation; retain ratewise evidence, not only a boolean.
3. Mutating any source import, design card, manifest or config after snapshot creation causes launch rejection. Changing the active worktree after launch cannot change the child's snapshot source path or hash. No symlink into a mutable source tree is allowed as the execution snapshot.
4. Run provenance includes model/core parameter counts, code commit and scoped diff, imported source hashes, raw/effective config, design, data, environment, seed, GPU UUID/host index, command, process identity, timestamps and isolated logs/artifacts. Large artifacts are referenced/hashes only in Git.

## F. Queue, admission and recovery

Proposed suite: `tests/test_meaningful20_queue.py`; unit tests use mocked process/GPU/filesystem probes without training.

1. Concurrent coordinator starts: exactly one obtains the durable lock. Repeated same-run enqueue is idempotent. Same PID with changed start ticks or boot ID is not treated as the original process.
2. Inject crashes before/after launch-intent, Popen, child identity persistence and completion write. Restart reconciles live command/output identity and attaches; it never blindly duplicates a process or overwrites existing output. Ambiguous state remains inspection-pending until safely resolved.
3. GPU index4, its recorded UUID, UUID/index mismatch, unknown/stale whitelist and unhealthy device all fail closed. The child verifies its visible UUID. Probe immediately before **each** smoke/training launch, and test changed occupancy between previous poll and launch.
4. Underestimated memory, high compute occupancy, insufficient disk, CPU saturation and throughput regression prevent admission or reduce future concurrency; they never rewrite batch size/epochs/protocol. Test multiple jobs on one healthy GPU when measured free memory/throughput permits. Preserve a 16 GiB disk reserve plus projected artifacts and the PRD's explicit GPU memory margin.
5. Structured failure types cover CUDA OOM, nonfinite output, source/config mismatch, device mismatch, disk shortage, external interruption, missing artifacts and nonzero exit. Same unresolved failure cannot loop automatically. A fixed new attempt has a distinct identity/source hash and preserves original outputs. Resource-wait is recoverable and does not mark research complete.
6. Test the new queue independently from the withdrawn runner. The old `SCREEN_STATUS.json` remains unchanged/withdrawn and its runtime guard still rejects all twenty old IDs.

## G. Complete-state resume

Existing inference BEST semantics must not change. Implement/test a separate optional `last_training.pt` at safe epoch boundaries.

1. Required fields: model/buffers including Slot Attention's private training RNG state, optimizer, next epoch/step, Python/NumPy/Torch CPU/CUDA RNG, sampler and missing-rate schedule state, scheduler/scaler when used, history/per-rate BEST tracking and matching selected-checkpoint hashes, source/design/config/data hashes. Resume Slot Attention and assert the very next private epsilon matches the uninterrupted run, as well as the global streams.
2. Compare an uninterrupted deterministic CPU two-epoch control with stop-after-epoch1/save/restart/epoch2: exact next-batch masks/order/RNG, selected epochs, model/optimizer state and reported metrics match. CUDA check requires matching next-batch order/masks and the fixed float32 gradient/state tolerances in B; log any backend nondeterminism without relabeling weights-only loading as resume.
3. Restore occurs before next stochastic data/model action. Saved RNG is after epoch evaluation/checkpoint selection. Missing scheduler/scaler state is acceptable only when that component is demonstrably unused. Reject corrupted files, missing mandatory fields, architecture/protocol/data mismatch and BEST-only files.
4. Transactional BEST: each improving epoch writes one immutable full model snapshot, shared by every improved rate's versioned selection metadata. Last-training stores committed snapshot paths/SHA256, all per-rate epoch/score records and committed history. Canonical `best_miss_*.pt` updates use atomic replacement only; earlier immutable references remain available. Verify corruption/missing versions fails closed and no automatic garbage collection deletes any version.
5. Crash injection covers after immutable snapshot creation, after each individual canonical BEST update in a multi-rate improvement, after history write, and immediately before/after atomic last-training commit. Resume restores canonical BEST and history from the **last committed** version references, never mixes the new partial selections with old training state, and removes/reconciles the uncommitted tail from effective history while preserving audit files. Compare next epoch/model/optimizer/RNG and all selected metrics with the uninterrupted control; no duplicate epoch rows or loss of earlier BEST records.
6. Disk admission accounts for all retained model/selection versions and the full last-training checkpoint, not just eight canonical BEST files. A run without compatible complete state starts a new from-scratch attempt after root-cause handling; BEST-only cannot pass this recovery check.

## H. Screening/promotion/continuation tests

1. Completeness requires normal process termination plus 100 sequential epochs, eight rates, finite selected metrics, matching per-rate selected checkpoint metadata/nonempty files, ratewise prediction/mask evidence and complete provenance. Existing partial metrics files do not constitute a completed run.
2. Recompute seed-66 Flat mean exactly from JSON: `0.8106809539495711`; verify high mean `0.7635225088637502`. Malformed/missing/duplicate rate entries fail. Means are unweighted over rates, then unweighted over seeds; do not pool repeated utterance exposures as independent samples.
3. Synthetic result fixtures test: none positive; exact tie; rounded-to-equal but truly positive; >3 positives; exact score ties requiring ID tie-break; positive high mean but negative primary mean; an unfinished twentieth method; failed result with an attractive partial score. Only strict primary-positive complete methods qualify and only after all twenty valid results exist.
4. Once ranked, promotion IDs are immutable. Only missing seeds67/68 for the frozen up-to-three list are queued; seed66 is reused. Mismatched/missing Flat replication reference blocks comparison until audited/reproduced; it cannot be silently replaced by a rounded average or different protocol.
5. Three-seed primary delta positive counts as operational improvement even if high-missing or one seed is negative; report those facts. Nonpositive mean for every promoted method triggers a new twenty-design intake. Do not change this rule after outcomes. No new round trains until its own complete twenty-card gate passes.
6. Reports enumerate every method/attempt, all seed/rate values, failures/replacements, parameter/runtime costs and test-oracle/adaptive-selection limitations. Negative seed66 methods are not expanded. No finite seed SD is presented as a significance conclusion.

## Verification sequence and commands

Use the repository's existing environment and standard-library unittest runner; do not install pytest or any new dependency. Commands are templates whose proposed test files must exist first:

```text
python -m unittest discover -s tests -p 'test_meaningful*.py'
python -m unittest discover -s tests -p 'test_osram.py'
python -m unittest discover -s tests -p 'test_osram_emotion_ablation.py'
python -m unittest discover -s tests -p 'test_miss0_parity.py'
python -m unittest discover -s tests -p 'test_osram_per_rate_selection.py'
python -m unittest discover -s tests -p 'test_readout_candidates.py'
python -m unittest discover -s tests -p 'test_readout20_runner.py'
python -m compileall -q gcnet_missing_m3 <new experiment scripts>
git diff --check
```

Run configured lint/type/static checks if present; do not introduce dependencies solely for an absent checker. Record unavailable/nonapplicable tools honestly. Then run each candidate's real-shape CUDA preflight on an admitted healthy biggpu UUID, the tiny uninterrupted/resume control, queue restart/crash simulation and a CLI config-only audit. Only then mark that candidate ready. Collect complete-result and ranking evidence after training; implementation checks alone never prove a gain.

Completion review is by an independent verifier after the leader integrates results. Architect review precedes Critic review; family authors do not approve their own mechanism audit. Commit/push checks stage only explicit task paths and confirm the remote branch actually contains the scoped Lore commit.
