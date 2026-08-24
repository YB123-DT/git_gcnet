# MPFiLM-RGCN Implementation Plan

> **For AI workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans task by task and track every checkbox.

**Goal:** Implement the complete-preserving missing-pattern FiLM first layer in official GCNet and run a fixed-mask IEMOCAP-6 fold-5 A/B experiment.

**Architecture:** Preserve PyG RGCN relation weights, relation-wise means, root, and bias. Add source-pattern correction and target-conditioned feature-wise modulation only on edges whose endpoints are not both complete. Pass the selected three-bit mask through `GraphModel` to both temporal and speaker branches, flattened conversation-first.

**Stack:** Python 3.10, PyTorch 2.2.2, PyG 2.4.0, unittest, NumPy, IEMOCAP-6.

## Execution tasks

1. Add failing unit tests for the missing module: complete forward/backward parity, zero new gradients under complete input, seven encodings, 0.7 activation, homogeneous neighborhoods, one-neighbor means, parameter counts, and CPU/GPU FP32.
2. Implement `gcnet/mpfilm_rgcn.py` with `encode_missing_patterns`, `flatten_valid_node_masks`, and `MissingPatternFiLMRGCNConv`; run tests to green and commit.
3. Add failing integration tests, then pass masks through `GraphModel` and both `GraphNetwork` instances without changing downstream modules; run all tests and commit.
4. Add failing deterministic bank tests, then implement `gcnet/mask_bank.py`, `--mask-bank-root`, `--mask-seed`, and `--fold-index`. Store an NPZ bank and JSON manifest with hashes; run tests and commit.
5. Add experiment CLI metadata and an explicitly marked short-run mode. Run one-epoch Original/Full smoke tests at missing rate 0.7 and verify finite losses, readable class statistics, Full gradients, and retained complete parity.
6. Generate 80 locked Original-vs-Full jobs: eight missing rates, five train/mask seeds, fold 5. Schedule no more than three processes per GPU. Gate the full launch on two-seed runs at rates 0.0 and 0.7.
7. Aggregate weighted F1, accuracy, paired differences, dispersion, class coverage, and collapse diagnostics. Run pattern-only and content-only controls only after the main A/B produces a stable signal.

Smoke metrics are never promoted to research results, and no model/loss/mask/selection change is permitted while the formal batch is running.
