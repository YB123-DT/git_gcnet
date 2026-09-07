# OSRAM emotion-only objective diagnostic

## Question

Does the current JEPA gradient help or hurt the OSRAM emotion backbone? This
is the direct follow-up to the MMoE routing and gradient-path audits.

## Conditions

- `Joint`: inherited Full-4E OSRAM reference, `classification + 0.1 * JEPA`.
- `Emotion-only`: newly trained Full-4E OSRAM with the same model, masks,
  cyclic schedule, optimizer and checkpoint protocol, but
  `training_objective=emotion-only`.

Only the objective is changed. With `emotion-only`, the predictor and EMA
teacher are not used in the training forward; the classification path remains
the same OSRAM path.

## Locked protocol

- CMU-MOSI, fold 1
- Seeds 66--70
- Cyclic rates 0.0--0.7
- H8, key/value 32, output 700
- Frozen wav2vec-large-c-UTT, deberta-large-4-UTT and manet_UTT
- 100 epochs, batch size 32, LR 1e-3
- Eight-rate Test-oracle diagnostic; per-rate extraction is retained

This is internal diagnosis only, not a formal validation-selected result.

## Interpretation

- Emotion-only higher: JEPA coupling is currently harmful or noisy for OSRAM.
- Joint higher: JEPA provides useful regularization despite weak direct
  coupling.
- Equal: the main bottleneck is neither JEPA nor MMoE, so inspect the frozen
  feature/emotion readout interface next.
