# CMU-MOSI Soft-Ordinal Task Head

## Status

Formal five-seed training is running on `biggpu`.

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

Final tables and the gate verdict will be appended only after all five jobs and
all forty prediction artifacts pass independent audit.
