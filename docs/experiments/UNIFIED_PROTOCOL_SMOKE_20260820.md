# Unified protocol smoke — 2026-08-20

Purpose: validate the complete real-data execution path, not estimate final
performance. All jobs used seed 66, missing rate 0.3, two epochs, the common
stability reconstruction (`rate=0.1`, `weight=0.01`), and the official Python
3.8 / PyTorch 1.8 environment.

Output root:

`/data2/yb/experiments/gcnet_unified_smoke_20260820`

| Dataset | Baseline W-F1 | JEPA W-F1 | test calls | paired audit |
|---|---:|---:|---:|---|
| IEMOCAP-Four, fold 5 | 0.1462 | 0.1462 | 1 / 1 | PASS |
| IEMOCAP-Six, fold 5 | 0.1583 | 0.1939 | 1 / 1 | PASS |
| CMU-MOSI official split | 0.5226 | 0.4920 | 1 / 1 | PASS |
| CMU-MOSEI official split | 0.7429 | 0.7390 | 1 / 1 | PASS |

All four pair audits verified equal environment, Git state, feature metadata,
split, sampler order, mask schedule and realized test rate, shared
initialization, common stability configuration, and seed bundle. The eight
manifests were present and every manifest recorded `test_call_count=1`.

The smoke exposed and fixed two real integration defects before formal runs:

1. IEMOCAP contains letter-suffixed conversation IDs such as
   `Ses03M_impro05a` and `Ses05M_script01_1b`; the LOSO parser now accepts the
   observed optional `a/b` suffix while retaining full-string validation.
2. MOSI/MOSEI previously rebuilt all utterance features for every process.
   They now use the same source-fingerprinted, process-locked atomic cache as
   IEMOCAP.

The two-epoch scores above must not be cited as model results. Their only valid
interpretation is that all eight training/validation/checkpoint/test/manifest
paths completed with finite values and passed the fairness audit.
