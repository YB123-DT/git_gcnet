# Code and environment audit

Audit started at feature/ecerc-features HEAD
`1db416c6f6cfd9965a0cccc876528ed09a1b0a1e`. The execution manifest records
the later implementation commit and SHA256 of every bundled source file.

## Verified native path

`ObservedSetEncoder -> OSRAMBackbone -> Base/Gap -> Local Skip + Flat -> LN -> smax_fc`.
Local is256. Base1024 and each Gap1024 contain forward512 and zero backward512.
Flat input4352 uses `LayerNorm -> Linear(4352,1600) -> GELU -> Dropout(.5)
-> Linear(1600,1600)`; LocalSkip is `Linear(256,1600)`. Changing Sequential
indices or replacing this adapter with a bare Linear invalidates old weights.

Native switches already implement the requested ablations. No new mathematical
Memory update or model architecture is needed. A2 uses `osram_gap_read='raw'`:
the effective Gap direction is the original Query, although the unchanged scan
may still compute residual diagnostics. A0/A1 zero suppressed groups before the
adapter's input LayerNorm. Returning raw contexts for diagnostics does not allow
them into the task path. No JEPA/completion loss is enabled.

## Accepted MOSEI parent and exact test masks

Parent: `/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.

Mask bank: `/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat/fixed_masks.pt`.

The parent has eight nonempty selected checkpoints and matching predictions,
plus config/history. Its initial model state digest is
`9a6a03edc69f7bd45c04464be8105eac96ec01d9a0922bbd2eb496a69ce82f77`.
This is not the older LR1e-3 MOSEI result. All new ablations start from random
initialization; loading the parent is only a preflight/inference compatibility
check, never their initialization.

## Separate frozen Nested stage

Reuse the original `FrozenNestedModel` implementation, with one parent scan and
an adapter pre-hook. Original default dim64/depth3/MLP64/groups8 adds159235
parameters. The parent remains frozen/eval while gradients pass through its
adapter, norm and task head into the Nested block. The LocalSkip must still use
the unmodified Local, with its original contiguous GEMM layout. A block-only
checkpoint requires its original parent's SHA256 and cannot load as full Flat.

MOSEI dim32, MLP128 and depth2 capacity variants are different models and must
not silently replace the default. This round does not train any stage2 model.

## Environment traps

Execution stays on biggpu with `/data2/yb/conda_envs/cmerc-repro/bin/python`:
Python3.10, Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2. Old experiment `setup()`
functions can still require the incompatible s0 Torch2.2.2/NumPy1.24.3 runtime.
The unified entry does not call those setup functions.

Bind dataset/cache environment variables **before** importing trainer/config.
Torch1.13's CUDA UUID enumeration has a known Python RNG-count mismatch: verify
the runtime sees exactly one GPU and repair both Python device_count aliases
when required before seeding/restoring RNG. Preserve full CUDA RNG state.

MOSEI official fold1 uses its own fold1 model initialization. Older MOSI factories
used a fold5-derived initialization despite official fold1 data; do not generalize
that convention to MOSEI. Host/guest random-mask sampling follows native GCNet
padded-batch rectangles. Fixed replay masks do not replace training sampling.

## Remaining limits

One seed and Test-oracle selection support internal mechanism comparison only.
A0 retains unused CHM parameters and scan cost for capacity matching; it is not
an efficient standalone Local baseline. A shared test bank improves pairing but
does not eliminate test-selection bias. Exact saved-prediction compatibility
and real-batch checks must pass before new training starts; actual measurements
belong in PREFLIGHT/RESULT, not inferred from this audit.
