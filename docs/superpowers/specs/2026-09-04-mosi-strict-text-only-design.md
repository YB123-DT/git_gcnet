# MOSI strict Text-only diagnostic design

## Question

Measure the downstream ceiling of the frozen official DeBERTa utterance feature without multimodal fusion, missing-modality learning, or graph reasoning.

## Model

The only input is `deberta-large-4-UTT`. A valid-utterance LayerNorm and linear projection map it to width 200. A one-layer bidirectional GRU with 100 units per direction models segments within each MOSI video. A LayerNorm--GELU--dropout--linear regression head produces one signed sentiment score per valid segment. Packed sequences prevent padding leakage.

Audio and visual tensors may be returned by the inherited loader but must never enter model forward. The model contains no GCNet, speaker graph, Student/Teacher projector, JEPA, MMoE, reconstruction, modality completion, or availability embedding.

## Protocol

- CMU-MOSI official train/validation/test membership.
- Frozen `deberta-large-4-UTT` feature bank.
- Seeds 66--70, 100 epochs, batch size 32.
- AdamW, learning rate `1e-3`, weight decay `1e-5`, dropout `0.5`, gradient clipping `1.0`.
- Select one checkpoint per seed by validation weighted F1; evaluate test once afterward.
- MOSI regression MSE training and the existing nonzero-label signed Acc-2/W-F1 metric implementation.

## Interpretation

- Mean test W-F1 at least 87 suggests multimodal node construction/backbone suppresses a strong text representation.
- Mean 85--87 suggests the frozen text feature/readout already explains most of the current ceiling.
- Mean below 85 suggests the feature bank, text encoder, optimization, or metric protocol needs inspection before further multimodal architecture work.

The diagnostic is not a paper method and is not compared against missing-rate results.

