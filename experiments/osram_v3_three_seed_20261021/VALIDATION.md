# V3 missing-seed completion validation

50 relevant CPU tests passed: classification configs/seeds, exact mask replay,
selection/recovery, finite queue, report completeness, supplementary MOSEI
factory and original V3 runner. Seed extensions were first tested failing at
the prior seed66-only boundary, then enabled. Both actual seed67 and68 CPU
MOSEI factories preserve original Flat parameter initialization exactly while
their own initializations differ from each other. Compileall/whitespace passed.

No numerical model, tracker, CHM/write/read/query, Flat, task or mask generator
file changed. The existing V3 worker gets an explicit optional setup factory
and bounded seed budget; its default remains seed66. Logging, initialization,
training-loader seed and reset_training_stochasticity all use the real seed.
MELD/IEMOCAP configuration changes only seed relative to the66 configuration.

Matched MOSEI Flat67/68 references are completed60epoch constant1e-3 runs from
osram_mosei_nested_gcnet_3seed_20261021/execution_v1, not old1e-4 experiments.
Same-seed config/init/split/training order/exact evaluation masks and actual
epoch training masks are checked by the reused V3 worker before accepting a
result. The current seed66 jobs continue in their original immutable sources.

Existing successful V3 GPU smokes (MOSEI s0, classification cmerc-repro) are
reused; the method and its tensor graph are unchanged by seed selection. No
extra smoke training budget is silently counted as formal results. Every new
worker retains zero-init parity, finite/nonzero task-gradient and full-state
integrity gates. Actual seed67/68 GPU progress is pending until queue admission.

The supplement waits for the predecessor classification controller to drain,
because the old immutable controller cannot account for another queue's GPU
children. This preserves total physicalGPU0/2 caps4/8 without killing or
restarting any existing worker.24 pending jobs are not24 running jobs.
