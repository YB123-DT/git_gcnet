# IEMOCAPSix stratified target-gradient diagnostic

## Setup

- Dataset: IEMOCAPSix, fold 5
- Seed: 66
- Checkpoint: existing `stratified + JEPA` checkpoint, selected at epoch 94
- Batch size: 32 conversations
- Training split: 120 conversations, four batches
- Rate assignment: exactly 15 conversations for each rate from 0.0 through 0.7
- Measured parameters: shared `missing_predictor.mmoe.experts.*`
- Diagnostic only: no optimizer update or new training

## Target counts per batch

| Batch | Missing A | Missing T | Missing V |
| ---: | ---: | ---: | ---: |
| 1 | 426 | 462 | 476 |
| 2 | 500 | 493 | 497 |
| 3 | 476 | 474 | 460 |
| 4 | 368 | 375 | 381 |

All three missing targets coexist in every stratified batch and have similar supervision counts.

## Shared-expert gradient cosine

| Target pair | Four batch values | Mean | Negative batches |
| --- | --- | ---: | ---: |
| Audio–Text | 0.0152, 0.0032, 0.0285, 0.0221 | 0.0172 | 0 / 4 |
| Audio–Visual | 0.0067, 0.0324, 0.0242, 0.0241 | 0.0218 | 0 / 4 |
| Text–Visual | -0.0014, -0.0074, 0.0083, 0.0061 | 0.0014 | 2 / 4 |

The three objectives are almost orthogonal. The two negative values are very close to zero and do not indicate strong destructive opposition.

## Shared-expert gradient norm

| Target | Mean norm |
| --- | ---: |
| Audio | 0.6206 |
| Text | 0.6917 |
| Visual | 0.4882 |

The maximum/minimum ratio is 1.42. Visual is the smallest target gradient rather than the dominant one.

## Verdict

The Visual-target domination observed in the old MOSI `all-rates-per-batch` checkpoint does not transfer to the new IEMOCAPSix stratified protocol. A fixed Visual down-weight or a global gradient equalizer is therefore not justified as a cross-dataset method at this stage.

The stable cross-protocol observation is weak target cooperation: pairwise cosines remain near zero. Target-private capacity can still be tested for that reason, but it must not be motivated as a universal Visual-gradient correction. MOSI needs its own stratified checkpoint before making a current-protocol MOSI claim.
