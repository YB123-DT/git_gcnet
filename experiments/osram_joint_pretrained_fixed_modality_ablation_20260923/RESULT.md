# Joint-Pretrained Fixed-Modality Robustness

> INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT

## Model and protocol

This is the three-dataset joint-pretrained **dual-projector completion** model:

- ordinary online projectors, OSRAM, and emotion classifier are trained on MOSI;
- a frozen jointly pretrained projector bank and `SourceOnlyM3Predictor` predict missing slots;
- predicted slots affect the current OSRAM read path but never persistent writes.

The fixed-modality protocol is identical to the cfg84 audit. All test
utterances are forced to `A`, `L`, `V`, `AL`, `AV`, `LV`, or `ALV` without
retraining. Existing `miss=0.7`, `0.3`, and `0.0` checkpoints are reused for
single-, dual-, and full-modality patterns respectively. Only seeds 66--68
exist for this pretrained variant, so the comparison below is paired over
those three seeds.

## Paired three-seed comparison

| Observed set | cfg84 no-JEPA | Pretrained completion | Paired Δ | Positive seeds |
|---|---:|---:|---:|---:|
| A | 41.318 ± 5.041 | **49.029 ± 4.801** | +7.711 | 3/3 |
| L | **86.208 ± 0.592** | 84.726 ± 0.190 | -1.482 | 0/3 |
| V | **58.694 ± 4.605** | 49.119 ± 7.724 | -9.575 | 1/3 |
| AL | **86.344 ± 0.720** | 85.073 ± 0.771 | -1.271 | 1/3 |
| AV | **61.243 ± 2.874** | 58.060 ± 1.925 | -3.183 | 1/3 |
| LV | **86.704 ± 1.127** | 85.251 ± 1.103 | -1.453 | 1/3 |
| ALV | **88.419 ± 0.276** | 87.122 ± 0.516 | -1.297 | 0/3 |

Aggregate pattern macros:

| Pattern family | cfg84 no-JEPA | Pretrained completion | Δ |
|---|---:|---:|---:|
| Text present: L/AL/LV/ALV | **86.919** | 85.543 | -1.376 |
| Text missing: A/V/AV | **53.752** | 52.070 | -1.682 |

## Pretrained completion per-seed W-F1

| Seed | A | L | V | AL | AV | LV | ALV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 53.848 | 84.507 | 57.895 | 84.339 | 59.166 | 85.498 | 86.533 |
| 67 | 48.993 | 84.840 | 46.107 | 85.002 | 55.838 | 84.045 | 87.331 |
| 68 | 44.246 | 84.832 | 43.355 | 85.876 | 59.178 | 86.209 | 87.500 |

## Conclusion

The jointly pretrained completion branch does not repair the no-Text
bottleneck. It improves `A` substantially, but this isolated gain is offset by
a larger `V` regression and an `AV` regression. The paired no-Text macro falls
from 53.752 to 52.070, while every Text-present pattern also decreases.

Therefore, the earlier aggregate No-Go result is confirmed under an exact
fixed-observed-set intervention: pretrained missing-latent completion does not
provide robust substitutable Text information to the classifier.

The table remains an internal per-rate Test-oracle diagnostic; it is not a
formal test result or a comparison of separately trained fixed-pattern models.
