# Matched Flat and two B6 configurations

INTERNAL TEST-ORACLE THREE-SEED REPLICATION

| Configuration | Seed | Eight-rate ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| Flat_matched | 66 | 60.781 | 37.107 | 57.275 |
| Flat_matched | 67 | 60.321 | 36.161 | 56.566 |
| Flat_matched | 68 | 60.805 | 35.645 | 56.735 |
| Flat_matched | mean ± sample SD | 60.635 ± 0.273 | 36.304 ± 0.742 | 56.859 ± 0.370 |
| B6_default | 66 | 61.216 | 35.935 | 57.254 |
| B6_default | 67 | 60.560 | 36.011 | 56.547 |
| B6_default | 68 | 60.704 | 34.606 | 56.601 |
| B6_default | mean ± sample SD | 60.827 ± 0.345 | 35.518 ± 0.790 | 56.801 ± 0.394 |
| B6_tuned | 66 | 61.193 | 36.008 | 57.257 |
| B6_tuned | 67 | 60.445 | 37.193 | 56.884 |
| B6_tuned | 68 | 60.790 | 35.376 | 56.675 |
| B6_tuned | mean ± sample SD | 60.809 ± 0.374 | 36.192 ± 0.923 | 56.939 ± 0.295 |

ACC-selected exact-epoch UA/W-F1; eight independent rate selections, not one deployment checkpoint.
Two selected B6 seed66 results are reused with original source provenance; seven new jobs use frozen9fa9f57.
Original accepted Flat60.693/UA36.255 is a different hyperparameter baseline and is not replaced.
Common-mask replay remains separate in each worker summary. Test-best table is not paired-mask performance.
Seed66 selected configurations; this is exploratory, not an independent test estimate.

## Completion and matched-control differences

All seven new trainings completed50epochs. The two B6seed66sources were reused,
so the requested comparison comprises three configurations×three seeds,not nine
new trainings. All72BEST weights(56new+16reused)remain present remotely.
FINAL_CHECKS.json confirms source-summary hashes,50contiguous epochs for each
new run,all8rates per seed,90CSVrows and recomputed mean/sampleSD.

Six new67/68workers failed only their auxiliary positional common-mask replay;
their original failures remain preserved. Requested Test-best metrics were
recovered on CPU from selected-epoch artifacts with hash,checkpoint ACC,
utterance-key,label,speaker and mask checks. No retraining or new inference.
Flat66completed its original full evaluation. Thus training/Test-best completion
is established, but cross-seed common-mask evaluation is **not** complete.
The authoritative recovered remote report is `testbest_recovery_v1/`.

| Three-seed configuration | DeltaACC(pp) | DeltaUA(pp) | DeltaW-F1(pp) |
|---|---:|---:|---:|
| B6_default versus Flat_matched | +0.192 | -0.786 | -0.058 |
| B6_tuned versus Flat_matched | +0.174 | -0.112 | +0.080 |

Both B6variants have slightly higher averageACC here,but neither improvesACC
andUAjointly. Default B6UAdeclines in all three seeds. Tuned B6UAimproves only
in seed67; its averageUAremains below matched Flat. ACCgains occur in66/67,
not68. This does not establish statistically reliable improvement or prove
which internal computation caused these differences. No significance claim is
made from three test-selected seeds. Do not compare selected seed66ACC61.2as
if it were the three-seed mean or replace the accepted historical Flat scores.
