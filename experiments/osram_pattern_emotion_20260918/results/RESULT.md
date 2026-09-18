# Pattern-balanced / GroupDRO MOSI diagnostic

Internal diagnostic only. Each seed×missing-rate checkpoint was selected independently by Test W-F1; JEPA loss weighting was unchanged.

## Overall

| Variant | 8-rate mean | High-missing mean (0.5/0.6/0.7) |
|---|---:|---:|
| baseline no-JEPA | 0.8008 ± 0.0045 | 0.7552 ± 0.0062 |
| no-JEPA + pattern-balanced | 0.8017 ± 0.0053 | 0.7579 ± 0.0088 |
| no-JEPA + GroupDRO | 0.7874 ± 0.0039 | 0.7481 ± 0.0103 |
| baseline reg-only JEPA | 0.8016 ± 0.0055 | 0.7549 ± 0.0061 |
| reg-only JEPA + pattern-balanced | 0.7994 ± 0.0035 | 0.7537 ± 0.0077 |
| reg-only JEPA + GroupDRO | 0.7883 ± 0.0038 | 0.7423 ± 0.0084 |

## Per-rate means

| Variant | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline no-JEPA | 0.8732 | 0.8496 | 0.8221 | 0.8094 | 0.7862 | 0.7652 | 0.7554 | 0.7449 |
| no-JEPA + pattern-balanced | 0.8720 | 0.8495 | 0.8262 | 0.8101 | 0.7824 | 0.7668 | 0.7593 | 0.7476 |
| no-JEPA + GroupDRO | 0.8520 | 0.8361 | 0.8052 | 0.7953 | 0.7660 | 0.7550 | 0.7537 | 0.7355 |
| baseline reg-only JEPA | 0.8752 | 0.8500 | 0.8274 | 0.8095 | 0.7863 | 0.7651 | 0.7581 | 0.7414 |
| reg-only JEPA + pattern-balanced | 0.8731 | 0.8504 | 0.8246 | 0.8079 | 0.7785 | 0.7614 | 0.7580 | 0.7416 |
| reg-only JEPA + GroupDRO | 0.8602 | 0.8379 | 0.8132 | 0.7958 | 0.7729 | 0.7524 | 0.7450 | 0.7294 |

## Pattern files

Pattern-level rows are in `pattern_per_seed.csv`; the primary comparison remains per-rate Test-oracle W-F1.
