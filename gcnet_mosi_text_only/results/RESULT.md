# CMU-MOSI strict Text-only diagnostic

## Model and protocol

The diagnostic consumes only the frozen `deberta-large-4-UTT` feature and uses `LayerNorm -> Linear(200) -> GELU -> one-layer BiGRU(100x2) -> regression head`. Audio and visual are not moved to the GPU or passed to the model. GCNet, JEPA, MMoE, modality completion, availability conditioning, and multimodal fusion are absent.

Five seeds (66--70) were trained for 100 epochs on the official CMU-MOSI split. Each checkpoint was selected by validation weighted F1, and test was evaluated once after selection.

## Results

| Seed | Best epoch | Validation W-F1 | Test W-F1 |
|---:|---:|---:|---:|
| 66 | 59 | 86.61 | 82.81 |
| 67 | 23 | 87.04 | 84.62 |
| 68 | 12 | 87.54 | 85.24 |
| 69 | 36 | 88.46 | 84.90 |
| 70 | 27 | 87.49 | 84.99 |
| **Mean ± SD** | — | **87.43 ± 0.70** | **84.51 ± 0.98** |

Additional five-seed test means are Acc-2 `84.82`, MAE `0.741`, and correlation `0.808`. The model has 388,849 parameters. No seed collapsed.

## Interpretation

The strict frozen-feature Text-only baseline does not reach 87 test W-F1. It is 0.58 points below the prior Text-Anchored multimodal model (85.09) and approximately 2.11 points below the prior complete M3 result (86.62), although those comparisons are descriptive rather than a single paired experiment.

This result rejects the simple explanation that a 87+ frozen-text classifier is being suppressed by GCNet or multimodal fusion. Under this controlled lightweight temporal readout, non-text modalities and/or the richer multimodal model add useful signal. It also narrows the MOSI ceiling problem toward the frozen feature/readout/generalization combination: training W-F1 approaches 97--99% while test remains around 85%, and seed 66 shows a 3.80-point validation-to-test gap.

The result does not prove that every possible Text-only architecture is capped at 84.51. It establishes the score of the locked BiGRU diagnostic and shows that another unmotivated GCNet replacement is unlikely to be the shortest route to 87.5.

Machine-readable artifacts are stored under `results/formal/`. Checkpoints and prediction NPZ files remain remote and are not committed.

