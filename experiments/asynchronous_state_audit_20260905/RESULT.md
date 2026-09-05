# Asynchronous-state ridge audit

## Purpose

This is the Stage 0 gate for the revised Backup:

> **Global Dialogue State + Asynchronous Modality State**

It tests whether modality-specific asynchronous history is useful in the
existing frozen Student representation before implementing a complete SSM
backbone. No model parameter is updated and no test loader is iterated.

## Protocol

| Item | Value |
|---|---|
| Dataset | CMU-MOSI, regression |
| Fold | 1 |
| Seeds | 66, 67, 68 |
| Rates | 0.0, 0.5, 0.7 |
| Student source | cyclic Missing-M3 Text-anchor residual checkpoint, seed-matched |
| Frozen feature | `ObservedSetEncoder` fused node (e_t) and A/T/V Student latents (z_t^m) |
| History decay | (lambda=1.0), nearest valid observation on each side plus current observation when available |
| Probe | StandardScaler + `Ridge(alpha=10)` for every variant |
| Fitting | train split only |
| Reporting | validation MAE, R², correlation only |
| Test loader | constructed by shared loader factory but never iterated; no test metric used |

The checkpoint directory is recorded in `mosi.json`. Its checkpoints were
selected by the earlier cyclic Test-oracle diagnostic, but this audit itself
does not read test labels or predictions.

## Probe definitions

- **Local:** (e_t), width 256.
- **Generic context:** ([e_t;\operatorname{mean}(e_{\ne t})]), where the mean is within the same conversation, width 512.
- **Asynchronous:**
  [
  [e_t;\bar z_t^A,\Delta_t^A;\bar z_t^T,\Delta_t^T;\bar z_t^V,\Delta_t^V],
  ]
  width 1027. For modality (m), (ar z_t^m) is an exponentially weighted mean of the nearest valid forward/backward observations and (Delta_t^m) is the weighted signed offset.
- **Shuffled history:** same asynchronous width and current (e_t), but each modality's observed latent values are permuted within the conversation before state construction.
- **Random history:** same asynchronous width and current (e_t), but the history block is replaced by deterministic random values. This is the dimension-matched control.

All five probes use the same ridge alpha, scaler procedure, train/validation
rows, and mask schedule. The different raw widths are explicitly controlled
by the same-width random-history variant; no width-specific hyperparameter
search was performed.

## Validation summary

Metrics are means over the three probe seeds. Correlation is higher-is-better;
MAE is lower-is-better.

| Missing rate | Local corr | Generic corr | Async corr | Async − Generic corr | Async − Generic MAE | Shuffled − Generic corr | Random − Generic corr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.7385 | 0.7383 | 0.7076 | **−0.0307** | +0.0424 | −0.0201 | −0.2051 |
| 0.5 | 0.4193 | 0.4087 | 0.4390 | **+0.0302** | −0.0034 | −0.0621 | −0.1604 |
| 0.7 | 0.4619 | 0.4212 | 0.4567 | **+0.0355** | +0.0074 | −0.0619 | −0.1837 |

At both nonzero rates, async beats generic in correlation for 2/3 seeds:

| Rate | Seed deltas (async − generic correlation) | Positive seeds |
|---:|---|---:|
| 0.5 | +0.0389, −0.0616, +0.1133 | 2/3 |
| 0.7 | +0.1103, +0.0334, −0.0371 | 2/3 |

The random-history control is below generic at both nonzero rates, so the
observed nonzero-rate gain is not explained by adding 771 random history
features. The shuffled-history means are below generic at both nonzero rates,
which supports—but does not prove—that the gain depends on aligned
same-modality history.

## Gate decision

| Condition | Result |
|---|---|
| Async > generic at η=0.5 | PASS: +0.0302 correlation, 2/3 seeds positive |
| Async > generic at η=0.7 | PASS: +0.0355 correlation, 2/3 seeds positive |
| η=0 not obviously lower (delta > −0.02) | **FAIL: −0.0307** |
| Same-modality history shuffle removes gain | PASS on mean correlation at η=0.5/0.7 |
| Dimension-matched random history cannot explain gain | PASS |
| **Overall** | **FAIL — do not implement/train full SSM** |

The asynchronous state has a measurable signal under high missingness, but it
does not preserve the complete-input condition required by the Backup gate.
This is insufficient evidence for a full SSM replacement. The correct action
is to keep the audit artifact, leave the SSM backbone unimplemented, and
require a new design or a revised gate before spending a full training budget.

## Reproducibility

- Implementation: `scripts/audit_asynchronous_state.py`
- Raw output: `mosi.json`
- Unit tests: `tests/test_asynchronous_state_audit.py`
- Source branch: `feature/missing-m3-target-ple`
