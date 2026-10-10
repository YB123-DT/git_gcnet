# Constant100 → lower-LR continuation50: Nested three seeds

INTERNAL DIAGNOSTIC ONLY. Per-rate Test-oracle BEST, not a formal paper result.

All seeds66/67/68 completed150 epochs, verified final provenance and50 actual
learning-rate traces. Original100 full states restored; Adam moments/RNG/selection
retained. Epoch101–150 constantLR1e-4 versus original1e-3; no architecture/loss
change. Evaluation mask hashes match originals. Server biggpu GPU7; implementation
924d40a for66 and5781797 for67/68. Launch records give source/provenance/commands.

## Original100 versus cumulative150 BEST

W-F1%, differences in percentage points. High means rates0.5/0.6/0.7.

|Seed|Original8-rate|Cumulative150|Difference|Original high|Cumulative high|Difference|
|---|---:|---:|---:|---:|---:|---:|
|66|80.992147|81.162868|+0.170722|76.077229|76.077229|0|
|67|80.850540|80.850540|0|75.497246|75.497246|0|
|68|79.623766|79.673827|+0.050061|74.180973|74.314468|+0.133495|
|Mean|80.488818|80.562412|+0.073594|75.251816|75.296314|+0.044498|

Cumulative1508-rate sample SD across seeds:0.785222pp (ddof1).
Only two of24 seed/rate BEST values refreshed:
seed66rate0.3:80.522617→81.888389 at epoch144;
seed68rate0.7:68.309248→68.709734 at epoch104.
Seed67 retains all original BEST checkpoints; it has no new performance gain.

|Rate|Seed66|Seed67|Seed68|Mean|
|---|---:|---:|---:|---:|
|0.0|88.078|88.845|87.418|88.114|
|0.1|86.358|86.721|84.697|85.925|
|0.2|83.735|86.714|82.918|84.456|
|0.3|81.888|81.092|80.666|81.216|
|0.4|81.012|76.941|78.748|78.900|
|0.5|77.675|74.998|77.856|76.843|
|0.6|75.032|75.931|76.378|75.780|
|0.7|75.525|75.562|68.710|73.266|

## New epochs101–150 ONLY, per-rate BEST

|Seed|8-rate mean|High-missing mean|
|---|---:|---:|
|66|80.509444|75.514779|
|67|78.217165|73.507036|
|68|78.910563|73.204930|
|Mean|79.212391|74.075582|

These are maxima restricted to new epochs, not cumulative scores or the final
epoch prediction. All three new-only means remain below their original100
per-rate BEST means. Cumulative selection cannot decline because it retains old
checkpoints; its small increase alone is not evidence of broadly better training.

Conclusion: continuation yields only a small cumulative gain, not a substantial
or uniform three-seed improvement. No automatic further sweep is justified here.
Flat67/68 have no complete recovery checkpoint, so no paired multi-seed Flat
continuation is claimed. Seed66 Flat continuation refreshed none of8 BEST values.

Evidence: final metrics/provenance archived alongside this report; complete
histories67/68 also archived. Seed66 detailed new-only values in SEED66_RESULT.md.
Original checkpoints and full recovery artifacts remain on remote; no weights
are uploaded to Git. Initial67/68 metadata-check failures occurred before training
and were preserved, then corrected without changing effective configuration.
