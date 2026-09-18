# Continuous-rate training-mask diagnostic

This is an internal MOSI diagnostic, not a formal paper result. The model is
the inherited no-JEPA causal OSRAM configuration: forward-only OSRAM,
`osram_write_step=0.6`, Flat readout, the original emotion loss, optimizer,
features, and inference path.

## Only changed factor

Training utterances use an independent rate `r ~ Uniform(0, 1)`. Each of the
three modalities is retained with probability `1-r`; with probability 0.25,
Text is additionally forced missing. Empty valid utterances are repaired by
retaining one modality, while preserving a forced Text omission when possible.
Padding is never sampled. The test masks remain the inherited official fixed
rate masks.

## Selection protocol

Five MOSI seeds (66--70) were trained for 100 epochs. For every seed and each
of the eight test rates (0.0--0.7), the checkpoint with the highest Test
weighted-F1 was selected independently. No eight-rate mean was used to select
the reported checkpoints. The comparison baseline is the already completed
no-JEPA cyclic run under the same test masks and per-rate selection protocol.

## Outcome

See `RESULT.md` and `results/summary.json`. The uniform-mask training run
reaches an 8-rate mean of 79.74% versus 80.08% for the cyclic baseline
(-0.33 pp), and a high-missing mean of 75.23% versus 75.52% (-0.29 pp).
T-missing sample-pooled W-F1 changes from 66.18% to 65.88% (-0.31 pp).
The diagnostic therefore does not support replacing cyclic training with this
continuous-rate/forced-Text mask on MOSI.

## Verification

- Remote pytest: 184 passed, one PyG deprecation warning.
- Real MOSI one-batch forward/backward smoke: finite loss (1.40645), no NaN.
- Smoke mask audit: sampled-rate mean 0.49868, realized missing fraction
  0.45354, forced-Text fraction 0.26690.
- Full run: seeds 66--70 completed with exit code 0.
- Official test mask hashes matched the inherited no-JEPA reference for every
  seed.
