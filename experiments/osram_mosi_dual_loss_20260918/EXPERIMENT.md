# MOSI dual-loss diagnostic

## Purpose

This is an internal diagnostic requested to test whether adding a binary sign
objective to the existing MOSI regression objective improves robustness. It is
not a formal paper result and uses the already established MOSI test-oracle
diagnostic protocol.

## Only training change

The model still has one scalar MOSI output and the causal OSRAM configuration
is unchanged. For every valid training utterance, the loss is

\[
L = L_{\mathrm{MSE}}(\hat y, y)
  + L_{\mathrm{BCE}}(\hat y, \mathbb{1}[y>0])\quad\text{for }y\ne0.
\]

The regression term uses all valid continuous MOSI labels, including the
continuous zero value. The BCE term uses only nonzero labels and supplies the
negative/positive sign target. Both terms have fixed equal weight; no loss
weight sweep was performed.

## Unchanged protocol

- causal OSRAM, forward-only, write step `0.6`;
- Flat readout with mean fusion;
- cyclic missing-rate training schedule;
- frozen input features, optimizer, learning rate, batch size, masks and
  training budget from the regression no-JEPA reference;
- seeds `66, 67, 68, 69, 70`;
- missing rates `0.0` through `0.7`;
- one independently selected Test-oracle epoch per seed and missing rate.

Test reporting follows the existing MOSI Non0 binary protocol: continuous
neutral samples (`y == 0`) are excluded from the reported binary metrics, and
the scalar output is thresholded at zero. This does not remove continuous-zero
examples from the regression term during training.

## Comparison

The control is the previously completed causal no-JEPA regression run at the
same seed/rate combinations. The treatment is this dual-loss run. The pulled
artifacts include per-seed/rate metrics, selected epochs, pattern summaries,
configuration and provenance files under `results/`.

## Hardware and status

The five seeds ran concurrently on remote GPUs 5, 5, 6, 6 and 7. All five
completed successfully. See [RESULT.md](RESULT.md) for the aggregate and
pattern-level outcome. The result remains **INTERNAL DIAGNOSTIC ONLY**.
