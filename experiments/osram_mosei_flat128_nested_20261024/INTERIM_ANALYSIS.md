# Why the current average barely changes

INTERNAL TEST-ORACLE SCREEN. Interim evidence, not a final screen conclusion.
Only existing selected predictions and training logs were read; no new training,
GPU inference, branch interventions or configuration changes were performed.

## Measured rather than assumed

Two completed seed66 candidates were compared with the accepted Flat128 seed66
on identical labels, availability, conversation IDs and utterance indices. All
16 prediction files per comparison match their selected SHA256 ledgers. Binary
polarity uses prediction>0 and excludes gold0, consistent with the current task.

| Candidate | Eight-rate W-F1 | Corrections | Harms | Polarity changes | Rate-utterance records |
|---|---:|---:|---:|---:|---:|
| N05:32/3/groups4/MLP128 | 83.375762 | 691 | 712 | 1403 | 29072 |
| N10:64/3/groups2/MLP128 | 83.377462 | 842 | 846 | 1688 | 29072 |

Flat128 seed66 W-F1:83.387044. Counts aggregate eight rates: a single utterance
can occur eight times; these are not29072 unique test utterances. Corrections
minus harms is not a weighted-F1 formula. These comparisons include joint
training and different per-rate selected epochs; they do NOT isolate the direct
effect of switching Nested on/off in one checkpoint.

For N10,5.81% of rate-utterance polarity decisions differ from Flat. Prediction
changes are therefore not negligible, but correct changes do not outnumber
new mistakes. At miss.1 W-F1 changes+0.4785pp and at miss.6+0.2736pp; miss.0,
.2,.3,.4,.5,.7 all decrease, yielding overall-0.0096pp. Similar mean scores
are not evidence of identical model behavior.

The existing first-epoch logs show finite nonzero gradients through Nested's
tokenizer, rooted layers, pooling, readout and decoders. This excludes a simple
permanently disconnected branch explanation, not every possible optimization
issue. Training MSE at epoch60: Flat0.396012; N100.396605; N050.398602. There is
no measured large training-fit advantage in these two candidates either.

## What the architecture does and does not imply

`gcnet_missing_m3/nested_sweep.py:38` constructs edges from read-role identity,
head/group index and Local connectivity. The GIN-style update sums fixed graph
neighbors, then rooted subgraphs use mean pooling. These are structural priors,
not measured utterance relevance or a learned guarantee of helpful corrections.
All21 candidates alter dimension/depth/grouping/MLP width within this same
prior, using the same already-read Local/Base/Gap and original MSE objective.
No candidate gains access to a new historical state or unobserved modality.

Thus this is a test of capacity allocation and one graph inductive bias. More
parameters can change decisions without improving their direction. The current
evidence does not support a reliable task benefit of that prior at Flat128;
it does not prove an information-theoretic limitation or that all second blocks
must fail. Fixed topology as the underlying cause remains a hypothesis, not a
mechanism conclusion demonstrated by these paired predictions.

## Boundary

Continue the already-approved finite queue unchanged. Do not add widths, layers,
losses, diagnostics or seeds based on this interim result. Remaining jobs can
change the final ranking. Per-rate test-oracle selection remains exploratory;
do not call these results independent generalization or statistical significance.
Machine-readable source paths, hashes, selected epochs and all per-rate counts
are in `PAIRED_PREDICTION_AUDIT.json`.
