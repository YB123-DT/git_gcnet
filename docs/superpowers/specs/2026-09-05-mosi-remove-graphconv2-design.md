# MOSI Second GraphConv Ablation Design

## Question

Does GCNet's relation-agnostic second `GraphConv` erase useful relation-aware
signals produced by the first `RGCNConv` in the current Text-anchor Missing-M3
model?

## Locked comparison

- Control: `RGCNConv -> GraphConv -> Post-BiLSTM` (inherit existing results).
- Treatment: `RGCNConv -> Post-BiLSTM`.

The treatment skips only `conv2` in both Temporal and Speaker graph branches.
It retains `conv2` parameters in the module/state dict so the switch changes no
unrelated initialization, parameter keys, optimizer construction, or random
number consumption. The unused parameters must receive no gradients.

## Interface

Add a CLI/config switch named `--graph-second-layer` with choices `graphconv`
and `identity`, defaulting to `graphconv`. Pass it through `MissingM3Config`,
`MissingM3GraphModel`, `GraphModel`, and both `GraphNetwork` instances.

## Invariants

- `graphconv` exactly preserves current forward behavior and checkpoint keys.
- `identity` computes `out = conv1(...)` and does not call `conv2`.
- Temporal/Speaker topology and relation types remain unchanged.
- Post-graph BiLSTMs, branch addition, emotion head, Text-anchor fusion,
  Missing-M3 predictor, EMA teacher, and JEPA loss remain unchanged.
- No parameter-matching layer or replacement projection is introduced because
  `conv1` and `conv2` already share the same output width.

## Verification

Unit tests cover default equivalence, identity-path forward/backward behavior,
config/CLI routing, and both graph branches. The experiment uses CMU-MOSI,
seeds 66--70, 100 epochs, cyclic mixed-rate training, and per-rate independent
Test-best epoch selection. Existing Text-anchor results are inherited.

This selection rule is intentionally retained at the user's request and the
result is labeled Test-oracle diagnostic rather than a leakage-free benchmark.
