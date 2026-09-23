# cfg84 Fixed-Modality Robustness Ablation

> INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT

## Protocol

- Dataset: CMU-MOSI test split.
- Model: original cfg84 no-JEPA causal OSRAM (`output=1600`, `K/V=64`).
- Seeds: 66, 67, 68, 69, 70.
- No retraining: every test utterance was forced to use exactly one of `A`, `L`, `V`, `AL`, `AV`, `LV`, or `ALV`.
- The existing per-rate Test-oracle checkpoint was reused without selecting on the new results: single-modality patterns use the `miss=0.7` checkpoint, pairs use `miss=0.3`, and `ALV` uses `miss=0.0`.
- `L` denotes the language/Text modality.

This is an evaluation-only robustness audit. It does not estimate the performance of seven models trained separately for seven observed sets.

## Five-seed result

| Observed set | Observed modalities | W-F1 mean ± SD | Accuracy mean | Δ W-F1 vs ALV |
|---|---:|---:|---:|---:|
| A | 1 | 39.447 ± 9.283 | 47.043 | -48.612 |
| L | 1 | 85.531 ± 1.071 | 85.671 | -2.528 |
| V | 1 | 58.107 ± 3.734 | 58.811 | -29.952 |
| AL | 2 | 85.774 ± 1.000 | 85.945 | -2.285 |
| AV | 2 | 59.397 ± 3.744 | 59.817 | -28.662 |
| LV | 2 | 86.609 ± 0.939 | 86.646 | -1.450 |
| ALV | 3 | **88.059 ± 0.541** | **88.079** | 0.000 |

## Per-seed W-F1

| Seed | A | L | V | AL | AV | LV | ALV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 35.643 | 85.620 | 53.381 | 86.478 | 57.958 | 86.754 | 88.205 |
| 67 | 45.277 | 86.804 | 61.188 | 86.988 | 63.295 | 87.805 | 88.321 |
| 68 | 43.034 | 86.200 | 61.515 | 85.566 | 62.476 | 85.553 | 88.730 |
| 69 | 48.210 | 84.041 | 54.905 | 85.432 | 59.273 | 87.145 | 87.677 |
| 70 | 25.073 | 84.992 | 59.546 | 84.407 | 53.983 | 85.789 | 87.362 |

## Interpretation

The robustness is strongly Text-dominated:

- `L` alone retains 85.531 W-F1, only 2.528 points below `ALV`.
- Every Text-present set is strong: `L=85.531`, `AL=85.774`, and `LV=86.609`.
- Every no-Text set is weak: `A=39.447`, `V=58.107`, and `AV=59.397`.
- The macro average over Text-present patterns is 86.493, versus 52.317 over no-Text patterns, a 34.176-point gap.
- Audio and Visual add modest gains once Text is available; neither substitutes for Text. `AV` improves only 1.290 points over `V` and remains 28.662 points below `ALV`.
- Audio-only is both the weakest and least stable condition (`39.447 ± 9.283`).

Therefore, cfg84 is robust to losing either Audio or Visual while retaining Text, but not robust to losing Text. The main missing-modality bottleneck is the no-Text family `{A, V, AV}`, rather than a uniform decline with fewer modalities.

## Caveat

The experiment follows the project's existing per-rate Test-oracle internal protocol and uses different selected epochs for the single-, dual-, and full-modality severity levels. Consequently, the table is appropriate for internal robustness diagnosis but not a formal independently selected test result or a pure same-checkpoint causal contribution estimate.
