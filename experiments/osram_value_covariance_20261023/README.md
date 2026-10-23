# Frozen history-Value covariance diagnostic

INTERNAL DIAGNOSTIC ONLY. No training, new model, optimizer or stability loss.

The scope is MOSI/MOSEI seed66, one protected original Flat miss0.3 checkpoint
per dataset, evaluated on **all** test utterances at the two saved exact GCNet
mask rates0.3 and0.7. This is not an eight-rate performance experiment. Original
Local, Keys, Queries, task head and write coefficients remain fixed. The original
checkpoint was selected using Test-oracle; these are internal mechanism results.

`core.py` implements the side ledger and replays independent Gaussian noise at
the **raw projected historical Value** coordinates before availability/sqrt-beta
weighting. Sigma is fixed at0.001 and0.01, with16 draws using the same unit noise
at both amplitudes. Replaying actual production Block Write preserves temporal
noise correlations. We never inject independent noise into the four read roles.

For each utterance/head we retain the complete4×4 matrix
`G = Q.T @ (alpha² Omega_previous) @ Q`, using the original residualized Gap
directions and active masks. User-defined `kappa = diag(G)`. The full read covariance
is exactly represented by per-head factors `sigma² G ⊗ I_value`, with cross-head
zeros under the independent-Value-noise assumption. This is not empirical
across-utterance covariance, raw-input robustness or sentiment uncertainty.

## Run

Run from an immutable source snapshot on `biggpu`, binding a healthy physical
GPU UUID. Host GPU4 is forbidden. Use the unchanged accepted s0 environment:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -m experiments.osram_value_covariance_20261023.run \
  --dataset CMUMOSI --reference /absolute/protected/mosi/seed66/flat \
  --gpu-uuid GPU-HEALTHY_UUID --source-commit IMMUTABLE_SHA --output /fresh/output/mosi
```

Use `--dataset CMUMOSEI` with the accepted reference
`/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat`.
`--smoke` runs only the first miss0.3 batch; smoke is not a full diagnostic.
No GPU job is launched by importing the module or running unit tests.

## Outputs and validation

- `STATUS.json`: exact checkpoint/config/data/mask identity, frozen-weight digest,
  source hashes, saved miss0.3 prediction parity, actual clean replay parity,
  linear realization errors, empirical covariance fit, GPU time/peak memory.
- `covariance/rate_*_batch_*.npz`: all16 G entries/head, all4 kappas/head and Monte
  Carlo covariance with utterance/batch/conversation keys; remote artifact only.
- `ROWS.jsonl.gz`: same samples, original labels, original predictions and16
  perturbed predictions per sigma, read-noise RMS, task variance, head kappas
  and actual head-only response. Remote artifact only, never overwrite old data.
- `AGGREGATE.json` and `RESULT.md`: full-test task shifts and strata by current
  pattern, clean correctness, clean-margin bins, history availability, kappa
  quartile and head. Binary polarity uses prediction>0 and excludes labels0;
  regression MSE includes every valid label. Corrections/harms are counts per
  draw, with separate clean-wrong/clean-correct denominators and conditional rates.

The head-only intervention is one fixed draw0 at sigma0.01 per head, not a stable
estimate. Unit first-order task variance (`gradient.T covariance gradient`) is
reported separately from actual nonlinear task response. High kappa does not
imply high prediction sensitivity, much less that a stability loss would help.

## Local tests

```bash
python -m unittest tests.test_value_covariance tests.test_value_covariance_runner -v
```

These tests require Torch/NumPy but not remote data. Real data parity, smoke and
full coverage are only claimed once their actual STATUS artifacts say complete.
