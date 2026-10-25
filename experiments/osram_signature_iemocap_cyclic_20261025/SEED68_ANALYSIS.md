# Seed68: assess ACC and UA together

The requested mechanism-analysis anchor is the completed IEMOCAP-6 seed68,
all five LOSO folds and eight original GCNet missing rates. Same frozen Flat
parents and saved masks as the completed Signature training; per-rate Test-ACC
selection,with UA from those same selected predictions. No UA reselection.
Results remain internal Test-oracle exploration. Seed68 is an analysis anchor,
not a replacement for the previously reported three-seed performance mean.

## Eight-rate mean over five equally weighted folds

| Model | ACC | UA | W-F1 |
| --- | ---: | ---: | ---: |
| Original frozen Flat |61.507197|60.307075|61.289083|
| Frozen Flat + Signature |62.273414|61.236214|62.272122|
| Change,percentage points |+0.766216|+0.929139|+0.983039|

## Per-rate fivefold arithmetic mean

| Rate | Flat ACC | Signature ACC | ΔACC | Flat UA | Signature UA | ΔUA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|0.0|64.54834|65.27885|+0.73051|63.47967|64.74092|+1.26125|
|0.1|63.43797|64.14755|+0.70957|62.11716|63.12502|+1.00787|
|0.2|63.43639|63.99937|+0.56298|61.66607|62.70484|+1.03876|
|0.3|62.45829|62.90832|+0.45003|61.22421|61.68340|+0.45920|
|0.4|61.30662|62.03708|+0.73046|60.15420|61.03240|+0.87819|
|0.5|60.22772|61.38329|+1.15558|58.61743|59.99077|+1.37334|
|0.6|59.18874|60.11015|+0.92141|58.32488|59.18931|+0.86443|
|0.7|57.45351|58.32270|+0.86919|56.87298|57.42306|+0.55008|

All eight rate means improve both metrics. High0.5–0.7 mean ACC improves
58.956655→59.938714 (+0.982058pp),UA57.938431→58.867712 (+0.929282pp).

## Per-fold eight-rate arithmetic mean

| Fold | Flat ACC | Signature ACC | ΔACC | Flat UA | Signature UA | ΔUA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1|61.19811|61.80808|+0.60998|60.98856|61.73960|+0.75104|
|2|63.46792|63.84587|+0.37795|62.22709|63.11673|+0.88963|
|3|58.63607|59.76737|+1.13129|58.22460|58.96289|+0.73829|
|4|60.53241|61.35086|+0.81845|58.09622|59.18310|+1.08687|
|5|63.70148|64.59489|+0.89341|61.99890|63.17877|+1.17986|

All five fold means improve both metrics. At the finer40 fold/rate conditions,
35improve ACC,33improve UA,and30improve both. Aggregate positivity does not mean
every individual condition improves.

## Classwise recall: same40 predictions,not independently selected

Class names have not been independently verified here; retain actual label IDs.
Each row averages recall equally over the40 fold/rate conditions,matching the
main UA arithmetic aggregation. Do not replace this with a pooled confusion UA.

| Label ID | Flat recall | Signature recall | ΔRecall(pp) |
| --- | ---: | ---: | ---: |
|0|40.61508|42.86591|+2.25083|
|1|71.25757|71.82348|+0.56591|
|2|56.90430|59.35814|+2.45384|
|3|61.32215|61.96817|+0.64601|
|4|69.56160|68.12616|−1.43544|
|5|62.18175|63.27542|+1.09368|

Five of six class recalls improve; label4 is a genuine regression and must not
be hidden. This supports a broader change in class discrimination,not just a
gain confined to the most frequently predicted class. It does not yet explain
which Signature statistic produces the change.

Across the eight evaluations,2034 rate-utterance records are corrected and1569
are harmed versus their own Flat parent (net465). These are repeated per-rate
records,NOT2034 unique utterances. Each rate's corrections exceed its harms.

## Next frozen mechanism audit

Apply the existing endpoint-preserving interior shuffles and second-level
antisymmetric-area removal to these exact seed68 selected checkpoints,without
training or changing Local/Base/Gap. Evaluate ACC AND six-class UA for every
fold/rate,with class recall/corrections/harms and shuffle-eligible strata.
Do not transfer the MOSI mechanism conclusion to IEMOCAP in advance.

Primary support would be consistent losses in both ACC and UA when the relevant
path statistic is removed. An ACC/UA tradeoff is not counted as a uniform
improvement. No change would leave the claimed mechanism unverified,even though
the learned Signature branch itself has a measured benefit.
