# cfg84 fixed observed-set ablation

Evaluation-only MOSI diagnostic using the original five cfg84 no-JEPA models.
Every test utterance is forced to one of `A`, `L`, `V`, `AL`, `AV`, `LV`, or
`ALV`; the model is never retrained.

To avoid selecting a checkpoint on the new diagnostic, each pattern uses the
existing per-rate checkpoint closest to its exact missing fraction:

- `ALV`: miss=0.0 checkpoint;
- `AL`, `AV`, `LV`: miss=0.3 checkpoint;
- `A`, `L`, `V`: miss=0.7 checkpoint.

This answers how robust the same mixed-rate-trained model is to a fixed
observed modality set. It is not a comparison of seven separately trained
single-/dual-modality models.
