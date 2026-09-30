# One-stage L1 Gate: completed results

INTERNAL TEST-ORACLE BEST CHECKPOINTS, NOT FORMAL PAPER RESULTS.
All3 seeds completed100 epochs, exit codes0. All24 best prediction W-F1 values
independently recomputed using nonzero labels, sign threshold>0 and weighted F1;
all24 best checkpoints retained. Existing canonical-mask validation passed.

| W-F1 %, mean ± sample SD | Original Flat | Joint L1 Gate | Paired delta pp |
|---|---:|---:|---:|
| Eight-rate |80.559 ±0.507|80.203 ±0.605|−0.356 ±0.344|
| High missing .5/.6/.7 |75.594 ±1.016|75.153 ±1.031|−0.441 ±0.439|

| Seed | Flat eight-rate | L1 eight-rate | Delta | Flat high | L1 high | Delta |
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.505|−0.563|76.352|75.472|−0.880|
|67|80.556|80.597|+0.041|75.990|75.987|−0.003|
|68|80.053|79.506|−0.547|74.440|74.001|−0.439|

Prior one-stage L2 comparison was80.286/eight-rate and75.955/high-missing.
At this fixed coefficient L1 does not improve those means either. This is one
fixed-lambda experiment, not a search or evidence all L1 formulations fail.

## Does the Gate stay near1?

The ranges below are over active evidence types and eight selected BEST
checkpoints for each seed, not final training statistics. Each cell metric is
weighted by its active evaluation utterances. Near-one means abs(g−1)<=0.01;
saturation means g<=0.81 or g>=1.19.

| Seed | Gate mean range | Mean abs(g−1) range | Near-one fraction range | Saturation range |
|---|---:|---:|---:|---:|
|66|0.8861–0.9909|0.0186–0.1139|0–33.25%|0%|
|67|0.9463–0.9813|0.0187–0.0537|0%|0%|
|68|0.8420–1.1690|0.0862–0.1987|0–6.50%|0–98.06%|

Thus this L1/lambda0.001 run did not achieve stable near-identity gating.
Seed67 avoids saturation but also has no active gates within0.01 of1 at the
selected checkpoints. Some seed68 evidence/rate cells approach the boundaries.
Do not infer stability from a gate mean near1 alone or from the earlier smoke.

Final epoch training (not main best comparison) records weighted penalties
about0.000095–0.000194 vs emotion losses0.768–1.136. Loss magnitudes alone do not
establish relative gradient strength, so this is not a proof of the causal
reason for drift. No coefficient tuning or additional training was performed.

Code4c98d3e; launch ea12a5d. Full archived metrics/history/config/provenance in
results/. Original Flat and previous L2/Stage2 weights were not altered.
