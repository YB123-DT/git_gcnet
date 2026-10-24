# Further stratification and whether correction scaling is justified

INTERNAL TEST-ORACLE SCREEN. Existing saved predictions only; no new training
or inference. Compare completed N10 and Flat128 seed66 on identical masks,
labels, conversation IDs and indices, with original per-rate selected epochs.
Machine-readable counts and provenance are in `A_ONLY_POSITIVE_DETAIL.json`.

## Cohort: current A-only, gold sentiment positive

2365 rate-utterance records:43 corrections,121 harms, net-78 correct. Records
include repeated utterances across rates:121 harms involve110 unique utterances,
43 corrections involve42 unique utterances; four utterances occur in both sets
under different rates. No utterance text/features or large prediction cache is
added to Git.

| Rate | N | Corrections | Harms | Net correct |
|---|---:|---:|---:|---:|
| .0 | 0 | N/A | N/A | N/A |
| .1 | 17 | 1 | 0 | +1 |
| .2 | 71 | 0 | 5 | -5 |
| .3 | 163 | 4 | 10 | -6 |
| .4 | 280 | 0 | 20 | -20 |
| .5 | 464 | 12 | 16 | -4 |
| .6 | 590 | 16 | 24 | -8 |
| .7 | 780 | 10 | 46 | -36 |

Different rates also use different selected model epochs. This is descriptive
stratification, not the causal effect of increasing missing rate alone.

| Position in conversation | N | Corrections | Harms | Net correct |
|---|---:|---:|---:|---:|
| First | 401 | 2 | 0 | +2 |
| 2–3 | 581 | 5 | 57 | -52 |
| 4–8 | 754 | 24 | 36 | -12 |
| 9+ | 629 | 12 | 28 | -16 |

57/121 harms occur at turns2–3; their net loss52 accounts for two-thirds of the
positive-cohort net loss78. First-turn differences cannot be the direct Nested
correction because the original empty-history guard skips that branch; joint
training can still change the main-path parameters.

| Absolute Flat FULL prediction | N | Corrections | Harms | Net correct |
|---|---:|---:|---:|---:|
| <=.25 | 400 | 33 | 85 | -52 |
| (.25,.5] | 675 | 8 | 31 | -23 |
| (.5,1] | 922 | 2 | 5 | -3 |
| >1 | 368 | 0 | 0 | 0 |

85/121 harms (70.25%) are near the original Flat's zero boundary. Among the294
originally-correct positive records in the first bin,85 become incorrect (28.91%).
This margin is NOT a Local-only prediction or an independent uncertainty score.
Position and margin are overlapping stratifications:57+85 cannot be summed.

| Past Text availability | N | Corrections | Harms | Net correct |
|---|---:|---:|---:|---:|
| Never seen | 728 | 3 | 42 | -39 |
| Seen | 1637 | 40 | 79 | -39 |

Conditional harm proportions among originally-correct positive records are
almost identical (42/708=5.932%;79/1333=5.926%). A larger raw harm count in the
seen group does not show that observed Text history is more harmful. Both
groups lose positive recall, so lack of past Text is not a sufficient explanation.

Gold-positive strength (offline analysis only):0<y<=.5 has15 corrections/41harms;
.5<y<=1 has12/48; y>1 has16/32. Harms are not confined to near-neutral labels.

## Can adjusting Nested correction magnitude improve the score?

Possible, not established by these cross-model counts. Current actual placement
is input-side: `osram.py:2268` calls MeaningfulInputAdapter; `:2313` retains
original LocalSkip and passes corrected inputs through the same Flat adapter.
The right frozen intervention is, for each task-input slot X,

`X(gamma) = X_raw + gamma * (X_nested - X_raw)`.

This includes Local only inside Flat input, Base forward512 and active Gap
forward512. LocalSkip, CHM state/query/write, inactive gaps and backward halves
must stay unchanged. A single shared gamma in {0,.5,1} would be a minimal
diagnostic of whether reducing the existing correction helps; no training is
required. THIS NOTE DOES NOT LAUNCH THAT INTERVENTION.

Crucially gamma0 means the Nested-trained checkpoint with its branch bypassed,
NOT the separately-trained Flat128 checkpoint. It cannot undo backbone drift
caused by joint training. Scores must compare gamma values within the same
checkpoint and exact masks, keeping corrections/harms and all-rate/high-rate
W-F1 together rather than selecting a coefficient solely on this positive cohort.

Boundary-concentrated harms motivate this check but do not prove excessive
correction: reducing strength could also remove33 near-boundary corrections.
Do not infer a deployable gate/threshold from these test-label strata.

## Relevant previous experiment, not a current result

`experiments/osram_nested_residual_scale_20261012/RESULT.md` already tested old
MOSI Nested checkpoints, s0, three seeds, one slot at a time with
gamma={0,.5,1,1.5,2}. Its largest test-selected eight-rate mean change was
+0.013327pp (Gap-A x2). It supplies no evidence of a large general strength fix,
but is not interchangeable with current new-Missing/cmerc MOSEI Flat128/N10.
No current gamma-replay score has been measured or claimed in this analysis.
