# Frozen-parent original Nested: completed three-seed comparison

All128 added67/68 branches completed; reuse64 verified66 branches. All192 selections passed independent checkpoint/prediction/parent/mask/metric validation. No retraining occurred during postprocessing recovery.

INTERNAL Test-oracle, per-rate WF1 selection for MOSI/MOSEI and ACC selection for classification. Class UA/WF1 are from the ACC-selected same predictions. IEMOCAP-6 headlines use equal fivefold then equal eight-rate means. Extra stage2 budget is not capacity/budget matched to emotion-only Flat.

## Eight-rate results, each seed separately

| Dataset | Seed | Flat ACC | Nested ACC | ΔACC(pp) | Flat W-F1 | Nested W-F1 | ΔWF1(pp) | Flat UA | Nested UA | ΔUA(pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | 66 | 79.535 | 79.821 | 0.286 | 79.460 | 79.778 | 0.318 | — | — | — |
| MOSI | 67 | 79.668 | 80.164 | 0.495 | 79.480 | 80.078 | 0.598 | — | — | — |
| MOSI | 68 | 79.173 | 79.459 | 0.286 | 79.074 | 79.415 | 0.341 | — | — | — |
| MOSI | 66/67/68 | 79.459 | 79.815 | 0.356 | 79.338 | 79.757 | 0.419 | — | — | — |
| MOSEI | 66 | 83.627 | 83.923 | 0.296 | 83.478 | 83.755 | 0.277 | — | — | — |
| MOSEI | 67 | 83.396 | 83.596 | 0.200 | 83.248 | 83.450 | 0.201 | — | — | — |
| MOSEI | 68 | 83.469 | 83.654 | 0.186 | 83.305 | 83.523 | 0.218 | — | — | — |
| MOSEI | 66/67/68 | 83.497 | 83.724 | 0.227 | 83.344 | 83.576 | 0.232 | — | — | — |
| MELD | 66 | 61.786 | 61.949 | 0.163 | 58.890 | 59.038 | 0.149 | 38.420 | 38.598 | 0.178 |
| MELD | 67 | 61.470 | 61.528 | 0.057 | 58.324 | 58.480 | 0.156 | 37.225 | 37.910 | 0.685 |
| MELD | 68 | 61.413 | 61.671 | 0.259 | 58.312 | 58.779 | 0.467 | 37.537 | 38.207 | 0.670 |
| MELD | 66/67/68 | 61.557 | 61.716 | 0.160 | 58.509 | 58.766 | 0.257 | 37.728 | 38.238 | 0.511 |
| IEMOCAP-6 | 66 | 62.011 | 62.439 | 0.428 | 61.382 | 62.138 | 0.756 | 59.945 | 60.937 | 0.993 |
| IEMOCAP-6 | 67 | 61.624 | 62.132 | 0.508 | 61.415 | 62.011 | 0.596 | 60.052 | 60.517 | 0.465 |
| IEMOCAP-6 | 68 | 61.507 | 62.500 | 0.993 | 61.289 | 62.508 | 1.219 | 60.307 | 61.534 | 1.227 |
| IEMOCAP-6 | 66/67/68 | 61.714 | 62.357 | 0.643 | 61.362 | 62.219 | 0.857 | 60.101 | 60.996 | 0.895 |

## Three-seed mean ± sample std

| Dataset | Metric | Flat | Frozen-parent Nested | Paired mean gain(pp) |
|---|---|---:|---:|---:|
| MOSI | ACC | 79.459 ± 0.256 | 79.815 ± 0.353 | 0.356 |
| MOSI | WF1 | 79.338 ± 0.229 | 79.757 ± 0.332 | 0.419 |
| MOSEI | ACC | 83.497 ± 0.118 | 83.724 ± 0.174 | 0.227 |
| MOSEI | WF1 | 83.344 ± 0.119 | 83.576 ± 0.159 | 0.232 |
| MELD | ACC | 61.557 ± 0.201 | 61.716 ± 0.214 | 0.160 |
| MELD | UA | 37.728 ± 0.620 | 38.238 ± 0.345 | 0.511 |
| MELD | WF1 | 58.509 ± 0.330 | 58.766 ± 0.279 | 0.257 |
| IEMOCAP-6 | ACC | 61.714 ± 0.264 | 62.357 ± 0.197 | 0.643 |
| IEMOCAP-6 | UA | 60.101 ± 0.186 | 60.996 ± 0.511 | 0.895 |
| IEMOCAP-6 | WF1 | 61.362 ± 0.065 | 62.219 ± 0.258 | 0.857 |

## High missing0.5/0.6/0.7, three-seed means

| Dataset | Flat ACC | Nested ACC | ΔACC | Flat W-F1 | Nested W-F1 | ΔWF1 | Flat UA | Nested UA | ΔUA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | 73.340 | 73.882 | 0.542 | 73.148 | 73.803 | 0.655 | — | — | — |
| MOSEI | 80.169 | 80.499 | 0.330 | 79.999 | 80.285 | 0.285 | — | — | — |
| MELD | 57.493 | 57.595 | 0.102 | 53.208 | 53.273 | 0.064 | 32.279 | 32.416 | 0.136 |
| IEMOCAP-6 | 59.277 | 60.033 | 0.757 | 58.843 | 59.837 | 0.994 | 57.969 | 58.735 | 0.767 |

## Interpretation and recovery

All three seeds have positive eight-rate ACC and W-F1 deltas; class UA deltas are also positive. IEMOCAP-6 has the largest observed ACC/UA gains. MELD changes are modest, especially high-missing W-F1. Three seeds and Test-oracle selection do not establish statistical significance or independent generalization.

Training source849079b remained immutable. The collector originally failed after training because legacy MOSI66 alone includes branch_off_* diagnostics. Collector7aa6fb1 fixes the metric-schema union/intersection mismatch; per-seed legacy diagnostics remain, while unavailable diagnostics are not fabricated for67/68. POSTPROCESS_RECOVERY.json preserves the original failure and confirms no model, mask, checkpoint selection or training changes.

No new epochs/experiments were started. All original Flat/Nested checkpoints remain protected. Signature/NeuMiss checkpoint deletion was separately authorized; the112 tiny Signature references were deferred until this collection succeeded, then removed by the bounded watcher. Saved Signature results/predictions/masks/source remain unchanged.

