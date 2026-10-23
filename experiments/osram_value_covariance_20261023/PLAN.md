# Frozen Value-noise covariance diagnostic

INTERNAL DIAGNOSTIC ONLY. No training, optimizer, regularization or new model.

## Locked scope

- Original Flat seed66, MOSI and MOSEI; hold each dataset's retained miss0.3
  checkpoint fixed. Do not substitute the new exemplar model or a V3/Router.
- First small diagnostic uses saved exact GCNet test masks at0.3 and0.7,
  all valid test utterances (not a label-selected subset).
- User-defined kappa is `diag(G)`, not its square root and not a Gram condition
  number. Store all16 entries/head of G; full2048-read covariance is the exact
  block factorization `sigma^2 G_h ⊗ I_value`, cross-head blocks zero under
  the stated independent projected-Value noise assumption.
- Raw projected Value noise: independent Gaussian, fixed absolute sigma0.001
  and0.01,16 draws. Same unit Gaussian draws at both amplitudes; report noise
  size relative to observed Value RMS, not a raw-feature robustness claim.
- Keep actual Key/Query, embedded beta, alpha, ridge, step0.6, read-before-write,
  current Local/Skip and original classification path fixed. Future writes
  use original projected K/V, not re-encoded perturbed inputs.

## Implementation and tests

- [ ] Tests-first `tests/test_value_covariance.py`: Omega equals explicit
  Value-Jacobian Gram; correlated off-diagonal terms; first-read/padding/Gap
  masks; production perturbed block_write versus linear coefficient replay;
  causality and finite gradient-to-read task variance.
- [ ] `core.py`: read-only Omega recurrence, masked effective Queries,
  production noisy Value replay, analytic covariance and local task Jacobian.
  Reuse production block_write terms; never alter its equation or core files.
- [ ] `run.py`: strict checkpoint/data/mask identity and clean prediction parity,
  all-sample per-head G/kappa, MC covariance fit and actual perturbed task
  predictions, nonzero-label polarity corrections/harms, all-label MSE.
- [ ] `report.py`: by dataset/rate/pattern/correctness/head/kappa groups;
  actual changes and projected first-order variance remain distinct.
- [ ] Verify on CPU and a real GPU batch, freeze immutable source, run the
  two datasets with healthy UUID binding without touching existing tasks.
- [ ] Publish code, small summaries and source hashes; preserve full covariance
  and predictions remotely. Report failure/absence of sensitivity as valid results.

## Interpretation gate

Omega predicts a distribution, not an individual Gaussian realization. Validate
the realization separately with the exact linear perturbation recurrence, and
compare Monte Carlo covariance to G with sampling error explicitly reported.
Read sensitivity is not task sensitivity; smoothing task response need not
improve accuracy. Gold correctness is offline grouping only, never an input.
No stability loss is authorized by this diagnostic.
