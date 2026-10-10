# Flat versus original Nested: other datasets BEST epochs

INTERNAL DIAGNOSTIC ONLY

Historical Flat selection metric unrecorded; Nested IEMOCAP selects ACC, MOSEI selects W-F1. These are stored selected epochs, not reselection by UA.

Descriptive distribution of selected epochs over seed/fold/rate; not a shared global BEST epoch or optimal stopping recommendation

| Dataset | Model | N | Median | Range | Epoch90–100 | Epoch100 |
|---|---|---:|---:|---|---:|---:|
| CMUMOSEI | Flat | 24 | 26.0 | 7–39 | 0 | 0 |
| CMUMOSEI | Nested | 24 | 26.0 | 19–36 | 0 | 0 |
| IEMOCAPFour | Flat | 120 | 68.5 | 36–99 | 20 | 0 |
| IEMOCAPFour | Nested | 120 | 63.0 | 34–100 | 16 | 2 |
| IEMOCAPSix | Flat | 120 | 72.0 | 45–100 | 10 | 2 |
| IEMOCAPSix | Nested | 120 | 66.0 | 43–100 | 7 | 2 |

## CMUMOSEI: exact stored epochs

| Seed | Fold | Model | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 1 | Flat | 18 | 29 | 21 | 29 | 18 | 27 | 8 | 26 |
| 66 | 1 | Nested | 29 | 20 | 20 | 29 | 29 | 26 | 27 | 29 |
| 67 | 1 | Flat | 26 | 13 | 26 | 26 | 26 | 39 | 27 | 27 |
| 67 | 1 | Nested | 23 | 27 | 19 | 32 | 21 | 27 | 19 | 36 |
| 68 | 1 | Flat | 27 | 27 | 27 | 26 | 7 | 25 | 28 | 39 |
| 68 | 1 | Nested | 34 | 25 | 25 | 26 | 34 | 26 | 25 | 22 |

## IEMOCAPFour: exact stored epochs

| Seed | Fold | Model | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 1 | Flat | 46 | 54 | 46 | 46 | 46 | 46 | 61 | 46 |
| 66 | 1 | Nested | 34 | 34 | 34 | 61 | 40 | 66 | 61 | 61 |
| 66 | 2 | Flat | 38 | 79 | 55 | 52 | 38 | 63 | 66 | 97 |
| 66 | 2 | Nested | 46 | 46 | 78 | 78 | 46 | 100 | 62 | 96 |
| 66 | 3 | Flat | 87 | 84 | 89 | 67 | 75 | 67 | 68 | 77 |
| 66 | 3 | Nested | 42 | 51 | 42 | 51 | 66 | 87 | 51 | 87 |
| 66 | 4 | Flat | 87 | 80 | 56 | 82 | 77 | 95 | 77 | 85 |
| 66 | 4 | Nested | 82 | 90 | 70 | 82 | 73 | 95 | 83 | 88 |
| 66 | 5 | Flat | 47 | 53 | 69 | 72 | 69 | 69 | 79 | 79 |
| 66 | 5 | Nested | 51 | 51 | 51 | 42 | 54 | 54 | 54 | 54 |
| 67 | 1 | Flat | 94 | 94 | 97 | 97 | 94 | 40 | 47 | 60 |
| 67 | 1 | Nested | 85 | 76 | 75 | 76 | 76 | 76 | 75 | 75 |
| 67 | 2 | Flat | 48 | 48 | 36 | 97 | 48 | 76 | 36 | 36 |
| 67 | 2 | Nested | 52 | 51 | 99 | 99 | 51 | 96 | 51 | 66 |
| 67 | 3 | Flat | 42 | 42 | 42 | 36 | 42 | 36 | 42 | 61 |
| 67 | 3 | Nested | 50 | 50 | 50 | 83 | 83 | 37 | 47 | 37 |
| 67 | 4 | Flat | 94 | 94 | 87 | 83 | 57 | 57 | 56 | 94 |
| 67 | 4 | Nested | 83 | 67 | 99 | 83 | 90 | 67 | 99 | 61 |
| 67 | 5 | Flat | 54 | 54 | 46 | 46 | 46 | 49 | 46 | 46 |
| 67 | 5 | Nested | 49 | 49 | 53 | 53 | 53 | 53 | 53 | 79 |
| 68 | 1 | Flat | 45 | 48 | 45 | 81 | 84 | 84 | 84 | 73 |
| 68 | 1 | Nested | 63 | 63 | 63 | 78 | 78 | 97 | 64 | 75 |
| 68 | 2 | Flat | 79 | 98 | 65 | 89 | 89 | 98 | 55 | 55 |
| 68 | 2 | Nested | 51 | 51 | 81 | 94 | 55 | 55 | 59 | 94 |
| 68 | 3 | Flat | 81 | 97 | 81 | 81 | 97 | 96 | 97 | 97 |
| 68 | 3 | Nested | 71 | 45 | 71 | 45 | 35 | 71 | 71 | 71 |
| 68 | 4 | Flat | 85 | 99 | 85 | 81 | 78 | 73 | 78 | 73 |
| 68 | 4 | Nested | 90 | 80 | 90 | 83 | 83 | 83 | 100 | 83 |
| 68 | 5 | Flat | 57 | 57 | 44 | 44 | 50 | 77 | 50 | 98 |
| 68 | 5 | Nested | 56 | 56 | 57 | 55 | 55 | 55 | 55 | 55 |

## IEMOCAPSix: exact stored epochs

| Seed | Fold | Model | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 1 | Flat | 85 | 85 | 70 | 67 | 61 | 100 | 60 | 67 |
| 66 | 1 | Nested | 58 | 75 | 58 | 62 | 75 | 58 | 53 | 60 |
| 66 | 2 | Flat | 76 | 77 | 91 | 95 | 78 | 59 | 95 | 63 |
| 66 | 2 | Nested | 83 | 76 | 83 | 83 | 83 | 83 | 69 | 83 |
| 66 | 3 | Flat | 61 | 61 | 72 | 65 | 72 | 80 | 53 | 60 |
| 66 | 3 | Nested | 59 | 59 | 59 | 60 | 60 | 59 | 52 | 60 |
| 66 | 4 | Flat | 75 | 75 | 75 | 88 | 75 | 75 | 66 | 85 |
| 66 | 4 | Nested | 49 | 44 | 51 | 85 | 52 | 91 | 85 | 51 |
| 66 | 5 | Flat | 62 | 51 | 62 | 62 | 60 | 62 | 62 | 81 |
| 66 | 5 | Nested | 63 | 63 | 65 | 88 | 61 | 61 | 59 | 59 |
| 67 | 1 | Flat | 63 | 60 | 71 | 46 | 64 | 100 | 88 | 67 |
| 67 | 1 | Nested | 70 | 70 | 70 | 62 | 64 | 71 | 70 | 70 |
| 67 | 2 | Flat | 55 | 78 | 99 | 78 | 99 | 99 | 78 | 89 |
| 67 | 2 | Nested | 69 | 69 | 78 | 69 | 69 | 79 | 78 | 87 |
| 67 | 3 | Flat | 54 | 54 | 54 | 70 | 71 | 70 | 71 | 59 |
| 67 | 3 | Nested | 51 | 51 | 62 | 65 | 91 | 56 | 93 | 51 |
| 67 | 4 | Flat | 88 | 88 | 72 | 72 | 72 | 72 | 72 | 72 |
| 67 | 4 | Nested | 88 | 88 | 74 | 68 | 43 | 58 | 58 | 51 |
| 67 | 5 | Flat | 57 | 60 | 61 | 50 | 69 | 61 | 61 | 71 |
| 67 | 5 | Nested | 70 | 57 | 70 | 69 | 57 | 69 | 93 | 57 |
| 68 | 1 | Flat | 73 | 78 | 73 | 78 | 78 | 99 | 78 | 59 |
| 68 | 1 | Nested | 86 | 61 | 70 | 61 | 61 | 70 | 61 | 70 |
| 68 | 2 | Flat | 74 | 83 | 74 | 68 | 72 | 77 | 77 | 77 |
| 68 | 2 | Nested | 88 | 64 | 64 | 70 | 82 | 82 | 66 | 94 |
| 68 | 3 | Flat | 60 | 70 | 80 | 70 | 83 | 68 | 68 | 68 |
| 68 | 3 | Nested | 66 | 57 | 57 | 75 | 57 | 57 | 57 | 87 |
| 68 | 4 | Flat | 80 | 72 | 72 | 88 | 72 | 88 | 88 | 88 |
| 68 | 4 | Nested | 73 | 73 | 61 | 51 | 50 | 47 | 49 | 50 |
| 68 | 5 | Flat | 45 | 45 | 45 | 45 | 70 | 45 | 82 | 90 |
| 68 | 5 | Nested | 79 | 66 | 66 | 66 | 100 | 45 | 100 | 66 |

Metrics and config files from all66 runs were read. No training, inference or checkpoint reselection.
IEMOCAP: five folds per seed, so120 selected epochs per model; MOSEI:24 per model.
Source paths and metric hashes are saved in SUMMARY.json.

## Interpretation and exact endpoint cases

MOSEI selects within epoch7–39 for Flat and19–36 for Nested, with median26 in
both. There is no endpoint-selected checkpoint in these runs. This alone does
not prove later checkpoints cannot improve, but does not suggest a truncated
100epoch budget from the observed selection locations.

Nested selects earlier by median on IEMOCAP than Flat; there is no uniform
Nested convergence delay. Four/Six each have two Nested selections at epoch100:

- Four: seed66 fold2 miss=.5; seed68 fold4 miss=.6.
- Six: seed68 fold5 miss=.4 and .6.

Flat Six also has endpoint selections: seed66 fold1 miss=.5 and seed67 fold1
miss=.5. These are individual fold/rate results, not evidence the full dataset
needs longer training. No extensions are launched by this analysis.
