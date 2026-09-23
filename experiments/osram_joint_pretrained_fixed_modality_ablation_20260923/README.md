# Joint-pretrained fixed-modality robustness ablation

Evaluation-only CMU-MOSI diagnostic for the three-dataset joint-pretrained
dual-projector completion model. Every test utterance is forced to exactly one
of `A`, `L`, `V`, `AL`, `AV`, `LV`, or `ALV`; no model is retrained.

The protocol matches the cfg84 fixed-modality audit: single-modality patterns
reuse the already selected `miss=0.7` checkpoint, dual-modality patterns reuse
`miss=0.3`, and `ALV` reuses `miss=0.0`. This is an internal Test-oracle
diagnostic, not a formal paper result.
