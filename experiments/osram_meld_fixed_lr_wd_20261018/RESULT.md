# MELD fixed learning-rate weight-decay follow-up

INTERNAL TEST-ORACLE SCREEN. LR1e-5, original Flat1600 + Local Skip.
Only weight_decay changes; GCNet random-mask-v1, cmerc-repro,50epochs,batch32.
The aggregate below contains TWO paired seeds67/68, not three. WD2e-4 controls are reused.

| Seed | weight_decay | ACC | UA | W-F1 | high ACC | high UA |
|---|---:|---:|---:|---:|---:|---:|
| 67 | 0 | 61.456 | 37.214 | 58.309 | 57.101 | 32.343 |
| 68 | 0 | 61.418 | 37.981 | 58.452 | 57.407 | 31.962 |
| 67 | 1e-05 | 61.470 | 37.225 | 58.324 | 57.126 | 32.368 |
| 68 | 1e-05 | 61.413 | 37.537 | 58.312 | 57.420 | 31.979 |
| 67 | 0.0001 | 61.446 | 37.141 | 58.296 | 57.139 | 32.383 |
| 68 | 0.0001 | 61.422 | 37.550 | 58.321 | 57.471 | 31.994 |
| 67 | 0.0002 | 61.461 | 36.991 | 58.267 | 57.165 | 32.197 |
| 68 | 0.0002 | 61.389 | 36.248 | 57.985 | 57.458 | 30.694 |
| 67 | 0.0005 | 61.466 | 37.114 | 58.305 | 57.203 | 32.248 |
| 68 | 0.0005 | 61.355 | 37.678 | 58.392 | 57.458 | 31.831 |

| weight_decay | seeds67/68 mean ACC | mean UA | mean W-F1 |
|---:|---:|---:|---:|
| 0.0002 | 61.425 | 36.620 | 58.126 |
| 0 | 61.437 | 37.598 | 58.381 |
| 1e-05 | 61.442 | 37.381 | 58.318 |
| 0.0001 | 61.434 | 37.346 | 58.308 |
| 0.0005 | 61.410 | 37.396 | 58.349 |

Same-seed fresh initialization and all50 actual train/test mask streams are checked against controls.
UA/W-F1 correspond to the same ACC-selected forward. Selected epochs can differ.
