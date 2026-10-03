# PRD: meaningful Local/Base/Gap search and experiment loop

User override, 2026-10-03: hard cap of **60 distinct methods / three rounds of
20**, superseding unlimited continuation below. Count a distinct method once
even when repeated or replicated across seeds; failed attempted methods must
not be hidden to circumvent the cap. No automatic fourth round. If the cap is
reached, finish authorized replication and report all outcomes, even if none
improves. Use the shared minimal verification template; no additional generic
review rounds before dispatch.

Status: sequential Architect and independent Critic APPROVED on 2026-10-03. Implementation may proceed under the user's continuous authorization; training still requires each candidate's correctness/resource gates. Context: `.omx/context/osram-meaningful20-loop-20261003T000000Z.md`.

## Outcome and scope

Find a reproducibly better internal OSRAM readout by repeatedly sourcing, implementing and running batches of exactly twenty distinct substantive non-MSA/non-MERC blocks on `ssh biggpu`. Preserve every negative and failed attempt. Complete each seed-66 batch, promote at most three improving methods under a fixed ranking, and check seeds 66/67/68. If none of the promoted methods improves the three-seed primary mean, automatically source the next twenty genuinely new designs. Do not stop between rounds to request the permission already given.

All results remain internal diagnostics: historical checkpoint selection is per-rate **test-oracle**, and repeated adaptive method search adds further test-selection bias. A positive result meets this operational search goal; it is not a paper-ready generalization or significance claim.

## Grounded facts and implementation boundaries

| Fact | Evidence |
| --- | --- |
| Local is one 256-vector; each forward memory read is eight genuine 64d heads; backward slots are zero in the causal configuration | `gcnet_missing_m3/osram.py:1550`, `:1560`, `:1594`; graph cards' producer references |
| Readout inputs must already reflect both existing ablation controls | `gcnet_missing_m3/osram.py:1563` through the emotion-ablation application at `:1584` |
| The current Flat anchor is `local_skip(local) + emotion_adapter(emotion_input)`, followed by the existing normalization | `gcnet_missing_m3/osram.py:1696`, `:1745` |
| Existing candidate plumbing is separate and must remain withdrawn | `gcnet_missing_m3/train_gcnet.py:140`, `experiments/osram_readout20_20261003/run.py::require_active_screen`, its `SCREEN_STATUS.json` |
| Per-rate BEST is updated on strict WF1 improvement and records test selection | `gcnet_missing_m3/train_gcnet.py:3401` |
| BEST files contain model/config/epoch/selection, but no optimizer or RNG state | `gcnet_missing_m3/train_gcnet.py:638`, `:3414`; they are inference/selection artifacts, not resumable training state |
| Seed-66 Flat protocol and exact metrics exist locally | `experiments/osram_current_history_relation_20261003/reference/{config,metrics,PROVENANCE}.json` |
| Existing queue supplies useful lock/process-identity/source-integrity patterns, but its authorized manifest is withdrawn | `experiments/osram_readout20_20261003/queue.py::process_identity`, `::coordinate`; inspect as reference, do not enable/import its launch policy |

Do not touch the unrelated dirty `gcnet/model.py`, dataset cache, other projects, or historical experiment outputs. No new dependencies. Project ownership stays on biggpu; GPU 4 is prohibited regardless of free memory. No checkpoint/data files enter Git.

The first round now has twenty accepted source/design cards across six families (acceptance is for implementation/testing, not performance): graph4, hypergraph4, grouping4, set/context3, optimization3, feature reasoning2. The complete frozen-ID manifest must be generated from accepted entries only; the grouping file uses `accepted`, optimization uses `papers[].accept_or_reject`, and other files use `candidates` or `cards`. Rejected entries in the same files are not counted. Equilibrium Aggregation is the twentieth source-resolved design; its primary author-code evidence is the publisher's code listing with documented omissions, so the manifest must record that explicit source exception instead of inventing a repository commit.

| Card family | Accepted first-round IDs | Preserve this family boundary |
| --- | --- | --- |
| `graph.json` (4) | `rrn_evidence`, `egt_evidence`, `residual_gated_graph_evidence`, `pna_evidence` | Per-head128 typed graph embeddings and complete graph cores |
| `hypergraph.json` (4) | `allset_transformer`, `ed_hnn`, `hyper_sagnn`, `sheaf_hypergnn_diag` | Per-head128 typed incidence structures and method-specific hypergraph readout |
| `grouping.json` (4) | `capsule_dynamic_routing`, `slot_attention`, `otke`, `capsule_variational_bayes` | 64d typed evidence / Local-conditioned grouping as resolved in each card |
| `set_context.json` (3) | `perceiver_io`, `dgcnn_dynamic_edgeconv`, `graph_multiset_transformer` | Per-head128 evidence with distinct encode/process/decode, dynamic-graph or graph-multiset mechanisms |
| `optimization.json` (3) | `hamburger_nmf_full`, `crate_mssa_ista_full`, `equilibrium_aggregation` | Shared-head128 tokenizer; Hamburger/CRATE typed transformed-token pooling versus EA aggregate output |
| `feature_reasoning.json` (2) | `tabnet`, `node` | Fixed579 vector (Local64 + four history128 + availability3); TabNet64 versus NODE192 final representation |

These card-defined adapters are intentionally different. The common wrapper standardizes safe evidence access, RNG isolation, zero residual injection and provenance, not a universal128d tokenizer or a pooled representation that would erase a core mechanism. Card definitions remain the authoritative source for exact layer widths/ordering.

## RALPLAN-DR

Principles:

1. Test a complete distinguishing source mechanism, not renamed gates, primitive pooling aliases, or arbitrary width/depth changes.
2. Keep information access and training protocol fixed: only a readout residual may differ.
3. Make the off path and zero-start path verifiably equivalent to the historical Flat, including RNG and memory calls.
4. Freeze source/design and comparison rules before results; preserve immutable evidence per run.
5. Continue autonomously through recoverable failures and negative rounds while making resource decisions from live measurements.

Top drivers: causal/protocol validity; meaningful architectural diversity; sustainable throughput with reproducible recovery.

| Option | Advantages | Costs / decision |
| --- | --- | --- |
| A. Independent meaningful-block registry, shared safe residual boundary, family implementations, per-run snapshots and durable staged queue | Isolates the withdrawn screen; supports twenty substantive mechanisms, independent coding and continuous batches | Requires wrapper/config/queue integration and explicit source audit; **chosen** |
| B. Twenty standalone experiment/model forks with separate runners | Strong implementation isolation; allows source-specific structures without a shared API | Duplicates memory/protocol logic, multiplies parity audits, complicates recovery and comparison; viable but inferior to A |
| C. Extend the withdrawn `osram_readout_candidate` manifest | Reuses existing plumbing quickly | Conflates rejected basic operators with accepted source blocks and risks enabling an explicitly withdrawn queue; invalid under current scope |

## Fixed comparison and continuation rule

Let `S(c,s)` be the arithmetic mean of the eight stored selected-test WF1 values for method c and seed s, using rates 0.0–0.7. Let `H(c,s)` be the mean at 0.5/0.6/0.7. Use full-precision JSON values, not printed rounded percentages.

The local Flat seed-66 reference is `S(flat,66)=0.8106809539495711` and `H(flat,66)=0.7635225088637502`. Before dispatch, verify that the remote source `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66` matches the local reference. Hash the actual source files; the nested historical provenance's older `reference_sha256` is not a substitute for hashing these copies. Current local file SHA256: config `65ba11e17dff20264d05f60932c00ae389f6efa7c3f0acf10720db659464a2ba`; metrics `e94e43c7623522e3fc5a9bd0d28574bc7ec30e7d2ec699f992f03f1176f91a64`.

1. Freeze twenty accepted source/design cards and their ordered IDs before any training in the round. Every card needs primary paper, actual author-code permalink/commit or an explicit source exception reviewed by the leader, domain exclusion, complete core chain, Local/Base/Gap mapping, dimensions, masks, first-history behavior, limitations, license handling and distinction from every other candidate.
2. Implement candidates independently. Once the round's twenty-card gate passes, a candidate may start seed 66 as soon as its implementation and required CPU/CUDA checks pass; the other nineteen implementations need not all be finished. Queue only test-qualified immutable snapshots.
3. Complete all twenty comparable seed-66 results before ranking or promoting. Failures are not negative performance results: diagnose and repair; preserve attempts. If a mechanism is proven infeasible within the constraints, record its rejection and append a source-reviewed replacement amendment before training the replacement. Do not claim a completed twenty-method comparison from fewer than twenty valid results.
4. An eligible seed-66 improvement requires `S(c,66) > S(flat,66)` strictly. An exact tie does not qualify. Rank eligible methods by descending `S(c,66)`; break exact ties by frozen candidate ID ascending. Freeze the first `min(3, eligible_count)` IDs. High-missing WF1 and individual rates are reported but never substitute for this ranking.
5. Complete seeds 67 and 68 for each frozen promoted method; reuse its completed seed 66. Do not expand seed-66 negative/tied methods. Obtain exact matched Flat seeds 67/68; audit their protocol, masks and provenance. Reuse compatible existing baselines, otherwise run the missing same-protocol Flat baselines on biggpu and record why.
6. A promoted method meets the operational improvement criterion when `mean_s(S(c,s)-S(flat,s)) > 0` over exactly seeds 66,67,68. Report each seed delta, sample SD, all eight rates and high-missing means, even if some deltas are negative. This is an effect-direction criterion, not a significance test. Finish the frozen promotion set before selecting the best three-seed mean (same ID tie-break).
7. If the eligible seed-66 set is empty, or all promoted three-seed deltas are nonpositive, enter the next round without asking again. Its twenty accepted designs must be substantively new relative to all prior accepted/tested designs; a renamed primitive or hyperparameter variant does not count. Preserve cumulative rejection and duplicate checks.
8. On a verified positive three-seed result, finish records, review, commit and push scoped artifacts to `github` on the current branch and report the exact operational outcome and bias limits. External resource waiting is a queue state, not permission loss. Only user cancellation or a genuinely unrecoverable external/authority blocker interrupts the authorized loop.

## Architecture contract

- Add independent `osram_meaningful_block: str = 'none'` / `--osram-meaningful-block`, threaded through TrainConfig, model, OSRAM and result metadata. Do not repurpose the old flag. Reject unknown IDs and incompatible interventions at configuration construction and direct-model entry points.
- Proposed files: `gcnet_missing_m3/meaningful_blocks.py` for registry/common boundary, `meaningful_blocks_{graph,hypergraph,grouping,set_context,optimization,feature_reasoning}.py` for owned family cores. Reuse existing safe helpers where correct; do not import the withdrawn runner's acceptance or GPU policy. Registry entries bind a design ID to a core factory and its declared output width; avoid pretending every core must emit 128 features (NODE currently emits 192).
- Common input boundary: Local `[T,B,256]`, Base `[T,B,1024]`, Gap `[T,B,3,1024]`, availability `[T,B,3]`, `umask [B,T]`, original pre-norm anchor `[T,B,1600]`. Use only forward `:512` memory halves; reshape `512=8*64` only where the card needs actual heads. Local is never fabricated into eight memory heads. Family projections/pooling preserve each accepted mechanism rather than imposing one universal tokenization.
- Family API is explicit: `build_<family>(method, latent_dim, num_heads, value_dim) -> nn.Module` with integer `output_dim`; `forward(local[N,D], evidence[N,4,H*V], active[N,4], availability[N,3]) -> features[N,output_dim]`. Evidence order is Base, Gap-A, Gap-T, Gap-V. The outer wrapper has already removed padding/no-history rows and sanitized inactive values; families own every projection/tokenization and repeat masks after biased operations. Grouping retains its card-defined64d processing, the graph/hypergraph/set-context/optimization token interfaces retain128d where specified, and feature reasoning keeps579d fixed-slot input with TabNet64/NODE192 output. The outer bridge uses the declared `output_dim`, never assumes128. Any per-family seed setup uses the existing experiment seed via a separate explicit initializer, without changing this tensor/factory interface or global RNG.
- Apply `torch.where` before projections/normalization and after biased operations; absent Gap data, including NaN/Inf poison, must never affect a result. Gap is active only when its modality is missing and the current utterance has prior valid history. Compute history from cumulative valid mask, not time index. Skip the branch entirely on padding and the first valid utterance, including after leading padding. No extra Memory query/write/cache, cross-utterance node or normalization state, label input, auxiliary loss, paired view, completion or persistent adaptation schedule.
- Retain exact original Flat computation, parameter names, order, task head and tensor-layout behavior. Add only `zero_init_output(core_features)` to the original pre-norm anchor, then use original normalization/head. Preserve CUDA GEMM layout when sanitization changes contiguity. Added module construction runs under saved/restored CPU/CUDA RNG. All new dropout remains zero. The only allowed forward random draw is Slot Attention's source-required training slot initialization, using the dedicated branch generator below; no global RNG stream may advance for it.
- Slot Attention owns a dedicated CPU `torch.Generator` seeded by a stable, manifest-recorded derivation from the existing experiment seed (never Python's process-randomized `hash`). Its RNG state is a persistent model buffer, restored before private training epsilon draws and saved after them; only that branch state advances. Transfer the resulting epsilon to the active device without global CUDA randomness. Evaluation uses a fixed epsilon buffer constructed from a separate private generator with seed1729, with no mutable RNG/state update. Include the private training RNG buffer in strict state round trips and complete training checkpoints; resume must reproduce the next epsilon exactly. Derive from existing `seed`, not a new tunable training-protocol field.
- NODE starts with finite provisional thresholds0/log-temperatures0 and `initialized=False`; fresh zero-bridge evaluation uses these values without mutation, never NaN placeholders multiplied by a zero bridge. Data-aware initialization replaces them only on detached valid history-bearing evidence from the first normal training batch with such rows; save its initialized state. If no eligible rows exist, keep initialization pending until the next normal training batch; never use test/validation to initialize or perform an extra data pass. Hamburger's inferred factors reset every utterance/forward; no online evaluation update.
- Equilibrium Aggregation preserves its accepted full ten-step Nesterov potential-minimization core with `create_graph=True` through every inner update during training. Its internal energy defines the forward map and is not an added task loss; omit the source outer gradient-norm auxiliary as documented. Reset iterates/momentum per utterance and use per-example stopping masks. Evaluation enters both `torch.inference_mode(False)` and `torch.enable_grad()` locally, clones inputs created under inference mode into normal tensors inside that scope, and obtains only the required energy derivatives; detach temporary iterates between eval updates and never mutate parameter `.grad`. Measure this candidate's actual higher-order memory cost independently.
- Config discipline: for the same seed, raw baseline JSON plus the **single** `osram_meaningful_block` key determines candidate config. Preserve all baseline keys including seemingly inactive fields. Report separately any current-schema defaults absent in the historical JSON; compare the effective candidate to the effective `none` config and require the sole difference to be that block field. Seed changes are allowed only in explicitly registered 67/68 replication runs. Checkpoints/output/GPU/source metadata belong in the run manifest, not training-protocol overrides.

## Work packages and ownership

| Stage | Owned implementation surface | Deliverable and prerequisite |
| --- | --- | --- |
| A. Research acceptance | Existing six card families; new manifest/decision ledger | Freeze/validate the now twenty accepted distinct designs, including Equilibrium Aggregation's publisher-code exception; excluded items never counted |
| B. Core integration | One integrator owns `osram.py`, `model.py`, `train_gcnet.py`, common registry and common tests | Independent default-off flag, strict config validation, wrapper, metadata, parity tests; preserve unrelated files |
| C. Family implementations | Separate executor per bounded family, only its family file/tests | Complete card mechanism, numerical/mask/state/gradient tests, source-to-code audit and parameter/resource profile; run in parallel within six-child cap |
| D. Execution infrastructure | Separate executor owns new round runner, queue, summary and their tests | Immutable manifests, config/hash checks, GPU admission, persistent state, resume safety and result completeness |
| E. Screening and promotion | Leader plus verifier | Per-candidate preflight then twenty seed-66 runs; fixed ranking; up to three two-extra-seed runs; full results ledger |
| F. Loop and delivery | Leader | New batch if no positive three-seed mean; otherwise final report and scoped Lore commit/push; archive every batch and attempt |

Checkpoint reconstruction has one additional integrator-owned touchpoint: `experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py::_build_model` must forward the new meaningful-block keyword (one scoped plumbing edit owned by the root). Verify enabled checkpoint rebuild through that helper. This does **not** authorize inheriting that historical runner's `GPUS=(4,5,7)` or any dispatch policy; all new launches use the new healthy-UUID admission contract.

Do not start C implementation before the plan's Architect/Critic gates. Do not start E before A completes. B/C/D may overlap after review, subject to file ownership; E may overlap unfinished C after A and the first candidate's checks complete. A later code change cannot mutate a running run's snapshot.

## Durable execution and recovery

Use a new queue under `experiments/osram_meaningful20_20261003/` and later numbered-round directories. The old screen stays withdrawn and launch-blocked. One durable coordinator lock per round; atomic state files; run key includes round/design hash, candidate, seed, source hash and protocol hash. Explicit states: awaiting-source-acceptance, awaiting-implementation, awaiting-tests, ready, admission-wait, launch-intent, running, interrupted, failed, complete. Persist command/output/source/GPU/process identity before launch. On restart match PID + start ticks + host boot ID, reconcile expected output files and lock ownership, and attach to live work; never blindly resubmit uncertain launch intent. Record each retry as an attempt without overwriting outputs.

Each run executes an immutable checked snapshot, with source commit plus scoped dirty diff if necessary, all imported source hashes, design/manifest/config hashes, data/cache and environment versions, seed, hostname, GPU index/UUID, command, process/scheduler identity, logs and output paths. Revalidate snapshot hashes just before launch. Do not synchronize changes into a running snapshot.

Before each smoke test or training launch, inspect live `nvidia-smi` index-to-UUID mapping, processes, free MiB, utilization and disk. Build an explicit healthy whitelist of host indices 0/1/2/3/5/6/7 and current UUIDs, permanently excluding both GPU 4 and its observed UUID. Set `CUDA_VISIBLE_DEVICES` to the selected healthy UUID and verify it inside the child; process `cuda:0` is not host GPU 0. Latest reported 0–3 light, 5 approximately 90%, 6/7 full is only historical context, never an admission decision.

Start one profiled job on the best eligible healthy GPU, then add concurrent independent jobs on that GPU only after full train/eval peak memory is measured and the first epoch completes. Conservative admission: free memory must exceed the incoming measured peak plus `max(2 GiB, 20% of that peak)`; before a peak exists use an isolated real-shape CUDA preflight and conservative estimate. Require free disk for the run's projected peak artifacts plus at least 16 GiB reserve. Inspect utilization/CPU loading and compare total completed training work per wall time before raising concurrency; if compute is already saturated or measured total throughput declines by >10%, stop increasing, serialize pending work and lower later admission. Never change batch size, training rates or epochs to fit. Persist measured costs and waiting reasons; do not delete artifacts automatically to free space.

Add a distinct atomic **last-training checkpoint**, opt-in for this workflow, saved at safe epoch boundaries: model and mutable buffers (including Slot Attention's private training RNG state), optimizer, scheduler/scaler when used, next epoch/global step, Python/NumPy/Torch CPU/all active CUDA RNG, sampler/missing-rate-schedule state, historical per-rate BEST records and their artifact hashes. Save after all epoch work so evaluation RNG consumption is included. Only resume when architecture/source/protocol/data hashes and required fields match, restoring RNG before any data iteration or stochastic action. Verify the resumed next-step/epoch against an uninterrupted control. Existing BEST lacks these fields: loading it is a warm start/new experiment, never a full resume. An interrupted run without a compatible complete checkpoint must restart from scratch as a new attempt after the cause is fixed; keep prior outputs and reason.

BEST selection and last-training commit form one recovery transaction. Whenever an epoch improves any rate, atomically publish **one immutable full model snapshot for that epoch**, then record each improved rate's epoch/score and snapshot path/SHA256 in versioned selection metadata. Each last-training checkpoint contains the complete committed per-rate metadata, immutable model references/hashes and committed history. Canonical `best_miss_*.pt` files are compatibility views updated only by temporary-file plus atomic replacement; overwriting them must never destroy older immutable models referenced by the last committed checkpoint. Commit last-training only after all required version files are durable. On recovery, treat the last-training checkpoint as authoritative, reconstruct/atomically restore canonical BEST files and history from its version references, and discard from effective history or reconcile any uncommitted tail. Preserve tail files for audit; do not regard them as committed selections. Keep **all** immutable model/selection versions under the run's disk budget; no automatic garbage collection. This protects a crash after only some per-rate BEST files were updated but before last-training committed.

OOM, nonfinite loss, source mismatch, disk shortage, missing outputs, CUDA device mismatch and process exit receive structured categories. A failed run is never selected as a result. Do not retry the same unresolved failure repeatedly. Resource shortage waits; recoverable code/numeric faults are fixed and reverified before retry; architecture/protocol changes create a new run identity.

## Acceptance criteria

1. Exactly twenty accepted distinct source/design IDs are frozen per round, with no rejected/basic-operator/duplicate entry counted; external-source claims cite paper and inspected code.
2. `none` preserves legacy state-dict keys and initialization/training RNG; all twenty zero-output bridges preserve anchor/head outputs and original shared-parameter gradients on CPU and CUDA under the test-spec tolerance; Memory call counts remain identical.
3. Every candidate passes poison-mask, first-valid, causality/conversation independence, checkpoint/state and finite-gradient checks, and its distinguishing core is exercised rather than replaced by a stub/primitive.
4. Every eligible run differs from its same-seed effective baseline only in the new block field, and stores matching immutable source/config/mask/data/environment evidence.
5. Duplicate queue starts/SSH disconnects/reboots cannot duplicate live work; GPU 4 by index or UUID is refused; admission decisions use current measured resources and disk budget.
6. Twenty complete seed-66 results have all eight expected rates, 100-epoch histories, correct per-rate BEST metadata/artifacts, expected prediction/mask evidence and no hidden failed attempt; fixed ranking/promotion/repeat rules execute on full precision.
7. Full resume demonstrably restores all applicable training state; incompatible/BEST-only artifacts are refused for resume. Complete status requires both process completion and artifact audit.
8. All accepted/rejected/failed/negative candidates and exact comparisons are reported with test-oracle/adaptive-search caveats. Scoped artifacts are committed and pushed to `github`; no unrelated edits, weights or cache enter the commit.

Detailed runnable checks and evidence files are specified in `test-spec-osram-meaningful20-loop.md`.

## Risks, mitigations and pre-mortem

- **Twenty names hide repeated primitives or incomplete ports.** Before training, compare each full core chain and source mapping; independently audit each implemented mechanism. Current optimization cards explicitly reject DDN pooling and iSQRT covariance and they must not count.
- **A apparent gain is protocol/RNG drift.** Fail dispatch on config drift, initialization mismatch, mask mismatch or anchor/memory parity failure. Keep negative runs visible and prohibit changing the ranking after seeing results.
- **Concurrency or restart corrupts evidence.** UUID-based GPU admission, immutable snapshots, durable launch intent/identity, measured memory/disk, full checkpoint validation and fail-closed completeness protect each run. Test crash windows and restart behavior before real dispatch.
- Three seeds and repeated test-oracle search cannot establish significance or an unbiased claim. Report this limit even when the operational improvement test passes.

## Available roles and execution handoff

Available relevant roles from the installed catalog: `planner`, `architect`, `critic`, `executor`, `test-engineer`, `verifier`, `researcher`, `explore`, `debugger`, `build-fixer`, `code-reviewer`, `git-master`. Parent performs Architect then Critic sequentially; planner never approves its own document.

Preferred authorized continuation: `$ralph` single-owner persistence with native executor lanes, respecting the six-child cap. Leader integrates/configures/dispatches; up to four concurrent family executors (high reasoning), one queue/test executor (high), one bounded verifier/researcher when a slot is free (medium/high). Shared-file ownership stays with the integrator. A remaining family starts as a slot frees. Review and GPU dispatch remain owned by the leader.

Team alternative only if durable shared coordination is useful: `omx team 5:executor "Execute .omx/plans/prd-osram-meaningful20-loop.md and its test spec: four family lanes plus queue/tests; leader owns shared integration; report readiness and evidence, do not independently dispatch GPUs"`, or `$team "Implement reviewed meaningful20 PRD with explicit family/queue ownership"`. Load the team skill before launch. Verification before team shutdown: all assigned implementation tests and source audits pass, per-candidate readiness manifests exist, shared integration is clean; Ralph then verifies cross-family parity, real CUDA preflight, queue safety, complete experiment artifacts, ranking and three-seed outcome. Team implementation completion never equals research-goal completion.

## ADR

Decision: independent source-grounded registry and family cores behind one opt-in residual boundary; immutable per-run execution and fixed twenty-at-a-time adaptive screening.

Drivers: preserve causal information/training protocol, honor meaningful diversity, and sustain safe autonomous throughput.

Alternatives: separate standalone forks remain possible but duplicate protocol machinery; reopening the withdrawn basic-operator queue violates current scope.

Why chosen: A allows parallel source-specific implementation while centralizing the parts that must remain identical, and allows early verified candidates to run after the twenty-design gate without waiting for all implementations.

Consequences: added wrapper/config/checkpoint/queue work needs explicit regression coverage; source adaptations are disclosed, and test-oracle gains remain internal. No performance or fixed number of rounds is promised.

Follow-ups: sequential Architect and Critic review; freeze the twenty accepted cards and source exceptions in the manifest; audit matched Flat references; implement gated execution; continue batches until the fixed operational criterion is met or an actual external blocker remains.

Review changelog: initial planner draft; synchronized completed twenty-card intake/family widths; applied Architect's Slot Attention RNG isolation, explicit family API, EA inference-tensor handling, transactional immutable BEST recovery and checkpoint-rebuild ownership. Verification uses existing unittest tooling without new dependencies. Remaining reviewer findings pending. Source/config/hash evidence in this document was inspected locally; no new experiment, remote inspection or implementation was performed by the planner.
