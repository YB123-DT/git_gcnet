# Twenty meaningful Local/Base/Gap blocks: continuous experiment plan

Planner draft, 2026-10-03, for sequential Architect → Critic review. The user has authorized continuous execution after technical review; no additional permission pause is required. See the [canonical PRD](../../.omx/plans/prd-osram-meaningful20-loop.md) and [test specification](../../.omx/plans/test-spec-osram-meaningful20-loop.md).

## Goal and controls

Each round accepts twenty distinct complete mechanisms from non-MSA/non-MERC primary papers and inspected author code, implements them, and runs the unchanged biggpu protocol at seed66. If the fixed three-seed improvement rule is not met, automatically source another genuinely new twenty. Retain rejected, failed and negative outcomes.

Local is one 256d vector. Forward Base/Gap512 contains eight real 64d read heads. Only current, already-read, ablation-applied evidence may enter a block. No new Memory query/write/cache, label input, auxiliary loss, completion, paired views or persistent adaptation schedule is allowed. Preserve the original Flat anchor, task head and training protocol.

Introduce independent `--osram-meaningful-block=none`, family implementation files and a common safe residual boundary. A zero-initialized output projection adds the new representation to the original pre-normalization Flat anchor. Verify the off path, original parameter initialization/RNG, dropout stream and Memory calls. Keep the old `osram_readout_candidate` screen withdrawn.

## Frozen selection rule

1. All twenty source/design cards must be accepted before the first training run of a round. Rejected papers, repeated pooling/attention aliases, basic operators and width/depth variants do not fill the quota.
2. Once that gate passes, each candidate may start as soon as its own implementation and CPU/CUDA checks pass. Other candidates may still be under implementation.
3. Finish all twenty comparable seed66 results before promotion. Repair failed attempts; a demonstrably infeasible mechanism needs a documented rejection and equally reviewed replacement, preserving the failed record. Failure is not a negative performance result.
4. The primary score is the equal-weight eight-rate WF1 mean, rates0.0–0.7. Exact Flat seed66 score is **0.8106809539495711**; high-missing0.5/0.6/0.7 mean is **0.7635225088637502**. Compare source JSON precision.
5. Only strict primary improvements qualify. Rank descending primary score, break exact ties by frozen ID ascending, and freeze at most three candidates. Reuse seed66 and complete seeds67/68 for each; do not expand negative/tied candidates.
6. Audit matching Flat seeds66/67/68; reuse compatible results or run genuinely missing references on biggpu. Operational success is a strictly positive mean of the three paired eight-rate score differences. Report every seed, rate and high-missing result even when negative.
7. Finish the frozen promoted set. If none meets the three-seed rule, automatically begin a new twenty-design round; otherwise complete independent verification and scoped Git delivery.

Per-rate BEST uses the test set; adaptive architecture screening adds selection bias. All results are internal diagnostics, and a positive three-seed mean is not a significance or formal paper claim.

## Implementation and evidence

One integrator owns configuration/model/OSRAM/common boundaries. Separate executors own graph, hypergraph, grouping, set/context, optimization and feature-reasoning files, with another lane for the new runner/queue/summary. Keep unrelated dirty `gcnet/model.py`, user files and historical outputs untouched. Same-seed raw configs differ only by the new block field; replication changes only the registered seed.

Every candidate needs mechanism-specific checks, all seven availability patterns, inactive/padding NaN/Inf poison tests, first-valid bypass, future/conversation independence, unchanged Memory call counts, strict state round trips and finite gradients. Test the core independently, then verify interior parameter updates after the zero output bridge learns; initially zero interior gradients alone are not a failure. NODE initialization is train-only on normal eligible rows; Hamburger factors cannot persist across utterances/eval calls.

Slot Attention training epsilon comes only from a dedicated CPU generator derived from the experiment seed, with persistent RNG state stored as a model buffer and included in complete resume. Global CPU/CUDA RNG must not advance. Evaluation reuses a fixed epsilon buffer from a separate private seed1729 generator and changes no generator/model state.

Pass CPU and real-protocol-shape CUDA checks before admission. Record parameters, peak train/eval memory and throughput. Each run uses an immutable source snapshot with commit/diff, source/config/design/data/environment hashes, seed, command, GPU index/UUID, process identity, isolated logs and artifacts. Subsequent edits cannot alter an active snapshot.

## biggpu admission and recovery

Recheck live index/UUID mapping, processes, free memory, utilization and disk before every smoke/training launch. Host **GPU4 and its UUID are always forbidden**. The process device `cuda:0` is not the host index. Earlier observations that0–3 are light,5 is approximately90%, and6/7 are full are context only.

Prefer gradual multi-run concurrency on a healthy GPU after measuring full train/eval peaks and a completed first epoch. Admission requires incoming measured peak plus `max(2 GiB,20% of peak)` free memory, and projected artifacts plus at least16GiB disk reserve. If compute is saturated or total throughput declines by more than10%, serialize pending work/reduce future admission; never change batch32/100epochs or other protocol fields. Resource shortage waits without host migration or automatic artifact deletion.

Use coordinator locks, atomic state, durable launch intent and PID/start-time/boot-ID reconciliation so SSH disconnect/restart cannot duplicate live jobs. Categorize errors, preserve attempts and fix the cause before retries.

Existing BEST files do not contain complete training state. Add a separate optional atomic last-training checkpoint with model/optimizer/progress, applicable scheduler/scaler, RNG, sampler/missing-rate state, prior BEST tracking and protocol hashes; verify resumed execution against an uninterrupted control. Without compatible complete state, restart from scratch as a recorded new attempt. Weights-only loading must never be called full resume.

Report all results/failures and commit scoped artifacts using Lore trailers, pushing only to the `github` remote/current branch. Exclude checkpoints, cache and unrelated changes.
