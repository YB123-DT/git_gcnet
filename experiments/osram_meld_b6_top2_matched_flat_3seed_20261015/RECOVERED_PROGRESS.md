# Matched Flat and two B6 configurations

INTERNAL TEST-ORACLE THREE-SEED REPLICATION

| Configuration | Seed | Eight-rate ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| Flat_matched | 67 | 60.321 | 36.161 | 56.566 |
| Flat_matched | 68 | 60.805 | 35.645 | 56.735 |
| B6_default | 66 | 61.216 | 35.935 | 57.254 |
| B6_tuned | 66 | 61.193 | 36.008 | 57.257 |

ACC-selected exact-epoch UA/W-F1; eight independent rate selections, not one deployment checkpoint.
Two selected B6 seed66 results are reused with original source provenance; seven new jobs use frozen9fa9f57.
Original accepted Flat60.693/UA36.255 is a different hyperparameter baseline and is not replaced.
Common-mask replay remains separate in each worker summary. Test-best table is not paired-mask performance.
Seed66 selected configurations; this is exploratory, not an independent test estimate.
