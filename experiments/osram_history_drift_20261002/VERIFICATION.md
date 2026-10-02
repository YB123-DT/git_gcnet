# Verification and execution record

- Code commit: `1aae42c`, branch `feature/osram-uniform-forced-text`.
- Server: `biggpu`; host GPU6, UUID `GPU-e4cafb17-818e-216a-b94a-7440063a9153`.
- Remote root: `/data1/yb/remote_experiments/osram_history_drift_20261002`.
- Python: `/data2/yb/reproduction_workspace/envs/s0/bin/python`.
- Persistent full-run PID: `2520058`; log `full.log`; output `full/`.
- Environment: `CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2`.
- Command in isolated `code/`: `python experiments/osram_history_drift_20261002/run.py --output-root /data1/yb/remote_experiments/osram_history_drift_20261002/full`.
- Unit verification: `python -m unittest tests.test_history_drift tests.test_history_drift_analysis` — 8 passed.
- Real validation smoke (`smoke_v2/`): 536 anchor exposures, original inputs/masks exact, repeat forward exact, frozen model/checkpoint hashes unchanged.

## Numerical investigation, not a model fix

Initial `smoke/` failed the original absolute Local tolerance of 1e-6; measured
error was 1.90735e-6. The original encoder packs only observed rows before each
modality projector. Changing history changes the GEMM row count. A separate
`check_numerics.py` comparison found changed packed-row geometry produces tiny
float32 differences, while identical singleton geometry produces exact outputs.
All four float64 intervention checks had zero Local and unaffected-prefix
differences. The real experiment remains float32 with the original model.

Recorded smoke Local maximum relative L2 difference: 1.29457e-7. Checks now require
both absolute <=1e-5 and relative <=1e-6. Raw current features and availability
must still match exactly. No observed difference is replaced with zero.
Numerical evidence is retained in `numerics.json`; failed smoke is preserved on
the server. Prefix checks use atol1e-5/rtol1e-6; predictions use the same original
threshold and baseline W-F1 must exactly reproduce the stored reference metric.

This task adds diagnostic scripts/tests only. No backbone, objective, checkpoint,
data features, or ongoing training runs were modified. Analysis remains internal:
source checkpoints were historically selected with per-rate Test-oracle results.

## Completed full run

All 16 split/rate cells completed (seed66, rates0–.7, validation/test). All eight
original test W-F1 values exactly reproduce the reference within 1e-10. Frozen
state/checkpoint hashes, same-input repeats, and unaffected-prefix checks passed.
Local numerical errors remain recorded in each cell and anchor. Postprocessing
verified artifact hashes, anchor identities, strict-prior metadata, finite values,
and unchanged current isolation. Raw masks and measurements are under `results/`;
separate validation/test summaries are under `summary/`.
