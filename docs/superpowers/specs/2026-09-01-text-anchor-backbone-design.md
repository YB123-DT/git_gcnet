# Text-Anchored Residual Backbone

## Decision

Retain the existing frozen Audio/Text/Visual utterance features and replace the
failed symmetric six-direction SAM fusion with one asymmetric complete-MOSI
candidate. Text is the primary semantic stream. Audio and Visual can only add
bounded, channel-wise residuals computed by Text-query cross-attention.

## Model

Each modality is projected to a shared width and receives one masked temporal
self-attention block. Two directed cross-modal paths are evaluated:

```text
Text queries -> Audio keys/values
Text queries -> Visual keys/values
```

The complete-modality representation is

```text
h = LN(Text + sigmoid(g_A) * Context_A + sigmoid(g_V) * Context_V)
```

The gates are vectors, not scalar modality weights. Missing keys never enter
attention. If Text is absent in later missing-rate work, the model falls back to
masked pooling over the observed Audio/Visual streams; this fallback is tested
but is not optimized in the Stage-1 complete-MOSI gate.

## Locked Experiment

- Dataset: CMU-MOSI, missing rate 0 only.
- Features and official split: identical to the completed SAM and inherited
  `m3_mosi` control.
- Seeds: 66--70.
- Checkpoint selection: minimum validation MSE; test evaluated once afterward.
- Control: inherited `m3_mosi` mean W-F1 86.62; no control rerun.
- Pass: positive five-seed mean delta, at least 3/5 positive paired seeds, no
  collapsed prediction.
- No JEPA, EMA teacher, completion, ensemble, upstream encoder changes, or
  test-oracle selection in this gate.

## Stop Rule

If the gate fails, archive the result and do not tune this fusion or extend it
to missing rates.
