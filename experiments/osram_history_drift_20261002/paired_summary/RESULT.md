# Identical Text-history deletion: Flat vs paired-view A/B

Seed66, existing per-rate Test-oracle checkpoints; INTERNAL diagnostic only.
No training. Original saved masks and same-current/different-history anchor IDs reused.
Both W-F1 columns evaluate the SAME anchors, excluding label0, threshold prediction>0.
Means are unweighted over eight rates; counts sum repeated rate exposures, not unique samples.
Prediction shift includes neutral anchors, matching the previous diagnostic.

## validation

|Model|Original history W-F1|Text-deleted history W-F1|Delta pp|Flip %|Correct→wrong / wrong→correct|Prediction shift|
|---|---:|---:|---:|---:|---:|---:|
|Flat|74.601|74.654|+0.053|2.294|10 / 12|0.06282|
|A|74.075|73.948|-0.127|1.932|10 / 9|0.04646|
|B|73.123|72.922|-0.201|1.078|6 / 4|0.04874|

## test

|Model|Original history W-F1|Text-deleted history W-F1|Delta pp|Flip %|Correct→wrong / wrong→correct|Prediction shift|
|---|---:|---:|---:|---:|---:|---:|
|Flat|80.644|80.113|-0.531|3.587|52 / 40|0.06894|
|A|80.309|79.695|-0.614|2.268|37 / 22|0.04955|
|B|79.884|79.632|-0.252|1.802|24 / 22|0.04941|

## Separate original full-test-set reference

Normal full-test eight-rate W-F1: Flat81.068%, A80.386%, B80.267%.
These use all eligible test utterances, NOT the anchor subset in the tables above.
A=paired-view task-only; B=paired-view task+InfoNCE. All single seed.
Checkpoint epochs may differ; this evaluates the existing selected systems, not a controlled epoch comparison.
Lower sensitivity alone is not evidence of higher sentiment accuracy or causal training benefit.
