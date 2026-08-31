# CMU-MOSI Soft-Ordinal Task Head

## Status

**COMPLETE — FAIL.** Five-seed training and all artifact audits finished.

## Question

The paired regression control predicts the continuous MOSI score with MSE and
classifies by its sign. A previously completed two-logit hard-CE treatment
discarded sentiment magnitude and did not improve five-seed performance. This
experiment asks whether a one-logit classification objective can retain the
continuous ordering while directly learning the fixed zero decision boundary.

## Treatment

For continuous label `y` and signed logit `z`:

```text
soft_target = (clamp(y, -3, 3) + 3) / 6
task_loss = BCEWithLogits(z, soft_target)
prediction = positive iff z > 0
```

`y=0` participates in the training loss with target `0.5`, but is excluded
from the standard MOSI Acc-2/W-F1 calculation. The threshold is fixed at zero;
no validation or test threshold calibration is allowed.

The isolated treatment entry is:

```text
python -m gcnet_missing_m3_soft_ordinal.train_gcnet
```

The existing `gcnet_missing_m3.train_gcnet` entry remains the regression
version. Both entries share the same model, features, masks, JEPA objective,
GCNet backbone, optimizer, and evaluation lifecycle.

## Locked protocol

- Dataset: CMU-MOSI, official split, fold 1;
- frozen features: wav2vec-large-c-UTT, DeBERTa-large-4-UTT, MANet-UTT;
- one model trains on all eight missing rates per source batch;
- Slot observed-set fusion, both graph branches;
- hidden 200, temporal/speaker windows 2/2, time attention disabled;
- LR `5e-4`, L2 `1e-5`, JEPA weight `0.1`;
- batch size 32, 100 epochs, seeds 66--70;
- best epoch selected only by validation eight-rate mean W-F1;
- Original, paired regression, and historical hard-binary results are inherited.

The paired regression control is the completed `lr=5e-4` five-seed run:

```text
miss0 mean W-F1:       85.6163%
eight-rate mean W-F1: 78.8680%
high 0.4--0.7 mean:   75.1207%
```

## Verification before formal launch

- 148 shared/new tests passed on `biggpu`;
- regression and soft-ordinal use identical classifier shape and parameter count;
- old regression and hard-binary tests remained green;
- one real one-epoch MOSI integration produced 8/8 prediction NPZ files;
- every NPZ had finite signed logits and its binary predictions exactly equaled
  `signed_logits > 0`;
- all eight test mask hashes exactly matched paired regression seed 66;
- the one-epoch checkpoint was deleted after the integration audit.

## Launch record

Five jobs were started concurrently with at most two jobs per healthy V100:

| Seed | GPU |
|---:|---:|
| 66 | 2 |
| 67 | 2 |
| 68 | 3 |
| 69 | 3 |
| 70 | 7 |

Remote result root:

```text
/data2/yb/remote_experiments/missing_m3_mosi_soft_ordinal_head_20260831/formal
```

## Five-seed results

All values below are test weighted F1 percentages. `Delta` is the eight-rate
mean difference from the same-seed `lr=5e-4` regression control.

| Seed | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 84.21 | 83.36 | 81.47 | 79.22 | 75.00 | 72.73 | 71.97 | 70.15 | 77.26 | -1.98 |
| 67 | 85.63 | 83.38 | 82.90 | 79.27 | 75.05 | 75.20 | 76.26 | 73.96 | 78.96 | +0.25 |
| 68 | 84.97 | 82.59 | 81.49 | 79.45 | 77.17 | 76.29 | 74.44 | 70.78 | 78.40 | -0.21 |
| 69 | 84.53 | 80.76 | 78.33 | 79.24 | 75.42 | 72.79 | 70.32 | 70.85 | 76.53 | -2.47 |
| 70 | 84.29 | 82.68 | 79.77 | 77.87 | 74.52 | 75.07 | 73.85 | 74.07 | 77.77 | -1.02 |

## Aggregate comparison

| Metric | Soft-Ordinal | Regression | Delta | Positive seeds |
|---|---:|---:|---:|---:|
| Miss0 | 84.7289 | 85.6163 | -0.8874 | 1/5 |
| Eight-rate mean | 77.7824 | 78.8680 | -1.0856 | 1/5 |
| High missing 0.4--0.7 | 73.7937 | 75.1207 | -1.3270 | 2/5 |

The treatment is lower at every missing-rate mean. Its smallest deficit is at
rate 0.2 (`-0.2932` point); its largest is at rate 0.4 (`-1.9627` points).

## Final audit

- 5/5 histories contain exactly 100 epochs;
- 40/40 prediction NPZ files independently reproduce stored W-F1, macro F1,
  and accuracy;
- 40/40 mask hashes match the same-seed regression control;
- 40/40 predictions contain both classes, so the negative result is not a
  class-collapse artifact;
- 40/40 signed-logit arrays are finite, and every saved prediction equals
  `signed_logits > 0` exactly;
- minimum signed-logit standard deviation is `0.6286`;
- parameter count is `32,089,733`, identical to regression.

## Verdict

**FAIL.** The preregistered gate required positive eight-rate and nonzero-rate
mean deltas plus at least 3/5 positive seeds. The observed eight-rate delta is
`-1.0856` with only 1/5 positive seeds.

The implementation worked as designed, but the linear soft target makes weak
positive and negative labels deliberately close to `0.5`. The evidence is
consistent with that calibration objective weakening the hard sign margin that
MOSI Acc-2/W-F1 ultimately rewards. This is an interpretation, not proof of a
unique causal mechanism.

Per the locked boundary, this closes the task-head route: no threshold tuning,
class weighting, focal loss, margin add-on, or regression/classification dual
head will be pursued under this experiment identity.
