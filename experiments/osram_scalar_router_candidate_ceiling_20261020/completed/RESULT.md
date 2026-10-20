# Scalar Router candidate capacity audit

INTERNAL POST-HOC ORACLE DIAGNOSTIC ONLY

No training, model loading, inference, mask generation, or weight edits. Saved selected NPZs only.
Full means the full readout of the SAME Joint checkpoint, not separately-trained Flat.
Candidates: Full; remove Gap-T←T (candidate_B); remove Base←T (candidate_G).
Eligible: current Text absent and strictly-past Text observed. Outside eligible all candidates are identical.
Polarity excludes gold0 and uses prediction>0, exactly matching original evaluation; MSE includes gold0.
Rates retain their originally selected checkpoints. Eight-rate metrics are arithmetic means, not a deployment checkpoint.
Counts summed over rates are repeated utterance-rate occurrences, not independent unique utterances.
Unanimity is reported WITHIN eligible rows. Neutral labels and ineligible copies do not inflate it.

## Eight-rate candidate support and actual routing (eligible nonzero labels)

| Case | N | Same polarity | Correctable / Full errors | Router corrections / harms | Missed corrections |
|---|---:|---:|---:|---:|---:|
| MOSI_main_seed66 | 1686 | 88.849% | 81 / 592 | 49 / 42 | 32 |
| MOSI_main_seed67 | 1686 | 86.121% | 92 / 578 | 39 / 35 | 53 |
| MOSI_main_seed68 | 1686 | 91.222% | 67 / 594 | 41 / 51 | 26 |
| MOSEI_main_seed66 | 7506 | 88.076% | 354 / 1675 | 2 / 5 | 352 |
| MOSI_lr1e5_seed66 | 1686 | 89.739% | 67 / 624 | 15 / 13 | 52 |

## Whole-cohort W-F1, original arithmetic rate averaging

| Case | Scope | Joint Full | Actual Router | Correctable-only oracle | Oracle−Full pp |
|---|---|---:|---:|---:|---:|
| MOSI_main_seed66 | mean | 78.8481 | 79.0120 | 80.4101 | +1.5620 |
| MOSI_main_seed66 | high | 72.7921 | 72.9527 | 75.2757 | +2.4836 |
| MOSI_main_seed67 | mean | 79.1558 | 79.2353 | 80.9314 | +1.7756 |
| MOSI_main_seed67 | high | 73.0054 | 73.0129 | 76.1725 | +3.1671 |
| MOSI_main_seed68 | mean | 78.7194 | 78.5607 | 80.0409 | +1.3215 |
| MOSI_main_seed68 | high | 72.3238 | 71.9077 | 74.4069 | +2.0831 |
| MOSEI_main_seed66 | mean | 83.2829 | 83.2709 | 84.4403 | +1.1575 |
| MOSEI_main_seed66 | high | 79.8559 | 79.8350 | 81.9144 | +2.0585 |
| MOSI_lr1e5_seed66 | mean | 76.3219 | 76.3643 | 77.5995 | +1.2776 |
| MOSI_lr1e5_seed66 | high | 71.0061 | 70.9540 | 73.1762 | +2.1701 |

Oracle preserves Full-correct predictions and only repairs Full errors for which a correct candidate exists.
Among correct candidates choose minimum squared error. This uses labels AFTER evaluation, not a trained/deployable Router.
It gives attainable discrete-candidate classification accuracy with oracle selection, not a proven global maximum of W-F1.

## MSE on eligible rows, pooled across rates (gold0 included)

| Case | Full | Router | Router−Full | Discrete candidate floor | Convex hull floor |
|---|---:|---:|---:|---:|---:|
| MOSI_main_seed66 | 2.267972 | 2.287514 | +0.019542 | 2.050143 | 2.050026 |
| MOSI_main_seed67 | 2.221095 | 2.236728 | +0.015633 | 1.981295 | 1.981134 |
| MOSI_main_seed68 | 2.240691 | 2.269155 | +0.028463 | 2.089635 | 2.089610 |
| MOSEI_main_seed66 | 0.924539 | 0.924069 | -0.000469 | 0.748155 | 0.747630 |
| MOSI_lr1e5_seed66 | 2.323952 | 2.330088 | +0.006136 | 2.122998 | 2.122886 |

## Paired MSE changes by actual polarity outcome (eligible nonzero labels)

| Case | Outcome | N | Full MSE | Router MSE | Router−Full |
|---|---|---:|---:|---:|---:|
| MOSI_main_seed66 | corrections | 49 | 2.609442 | 2.206999 | -0.402443 |
| MOSI_main_seed66 | harms | 42 | 2.534628 | 2.961119 | 0.426491 |
| MOSI_main_seed66 | both_correct | 1052 | 1.453916 | 1.559186 | 0.105270 |
| MOSI_main_seed66 | both_wrong | 543 | 3.987633 | 3.853125 | -0.134509 |
| MOSI_main_seed67 | corrections | 39 | 2.345738 | 2.083784 | -0.261954 |
| MOSI_main_seed67 | harms | 35 | 2.288129 | 2.592901 | 0.304772 |
| MOSI_main_seed67 | both_correct | 1073 | 1.461810 | 1.557749 | 0.095939 |
| MOSI_main_seed67 | both_wrong | 539 | 3.927343 | 3.791871 | -0.135471 |
| MOSI_main_seed68 | corrections | 41 | 1.892751 | 1.609071 | -0.283680 |
| MOSI_main_seed68 | harms | 51 | 2.541653 | 2.910869 | 0.369216 |
| MOSI_main_seed68 | both_correct | 1041 | 1.434280 | 1.535680 | 0.101400 |
| MOSI_main_seed68 | both_wrong | 553 | 3.966658 | 3.856490 | -0.110167 |
| MOSEI_main_seed66 | corrections | 2 | 0.281215 | 0.277002 | -0.004213 |
| MOSEI_main_seed66 | harms | 5 | 1.663602 | 1.669192 | 0.005591 |
| MOSEI_main_seed66 | both_correct | 5826 | 0.681102 | 0.682958 | 0.001856 |
| MOSEI_main_seed66 | both_wrong | 1673 | 2.370232 | 2.364399 | -0.005833 |
| MOSI_lr1e5_seed66 | corrections | 15 | 2.953975 | 2.853415 | -0.100560 |
| MOSI_lr1e5_seed66 | harms | 13 | 2.094785 | 2.201251 | 0.106465 |
| MOSI_lr1e5_seed66 | both_correct | 1049 | 1.391277 | 1.416699 | 0.025422 |
| MOSI_lr1e5_seed66 | both_wrong | 609 | 4.116006 | 4.091420 | -0.024586 |

Discrete floor chooses the closest candidate by gold label. Convex floor projects the label onto [min candidate,max candidate].
Both are post-hoc regression bounds, not deployed scores. Convex hull endpoints may require limiting one-hot weights.
A same-polarity convex mixture cannot change polarity, but may still improve or harm MSE.
Do not infer that Memory is useless or that training LR is optimal from this conditional readout audit.
Per-rate all/eligible counts, oracle metrics and candidate metrics: PER_RATE.csv.
MSE differences for corrections, harms, both-correct, both-wrong and unanimity: MSE_GROUPS.csv.
All prediction/checkpoint/mask provenance and numerical checks: SUMMARY.json.
