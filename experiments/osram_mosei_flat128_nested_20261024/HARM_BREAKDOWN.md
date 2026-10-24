# Which conditions account for canceled corrections?

INTERNAL TEST-ORACLE SCREEN. Frozen saved-prediction analysis only, no new
training/inference. N10 (dim64/depth3/groups2/MLP128), seed66, compared with the
same-width Flat128 seed66. Both completed60epochs. Exact source paths, per-rate
selected prediction SHA256 and selected epochs are recorded in
`PAIRED_PREDICTION_AUDIT.json`. All input hashes and labels/availability/IDs/index
alignment were verified again for this breakdown at2026-10-24T23:33:54UTC.

Polarity prediction:>0 positive, otherwise negative; gold0 is excluded. Counts
are pooled across eight rates, NOT unique utterances and NOT mean W-F1 changes.
Availability means modalities actually observed in the current utterance, not
persistent-missing conversation conditions.

| Current observed modalities | Rate-utterance records | Flat wrong -> Nested correct | Flat correct -> Nested wrong | Net correct count |
|---|---:|---:|---:|---:|
| A | 3692 | 112 | 174 | -62 |
| T | 3717 | 74 | 85 | -11 |
| V | 3645 | 167 | 137 | +30 |
| AT | 2647 | 85 | 78 | +7 |
| AV | 2676 | 132 | 115 | +17 |
| TV | 2724 | 70 | 62 | +8 |
| ATV | 9971 | 202 | 195 | +7 |
| Total | 29072 | 842 | 846 | -4 |

## A-only polarity directions

| Gold polarity | N | Corrections | Harms | Net correct count |
|---|---:|---:|---:|---:|
| Positive | 2365 | 43 (negative->positive) | 121 (positive->negative) | -78 |
| Negative | 1327 | 69 (positive->negative) | 53 (negative->positive) | +16 |

A-only therefore loses62correct records, largely because of lost positive-class
correctness. It is not a uniform decline across all missing-Text observations:
AV gains17 and V-only gains30correct records. These counts do not establish why
the graph training produced these conditional changes.

Across all modes/rates, positives have453corrections/428harms (+25correct), while
negatives have389corrections/418harms (-29correct). A pooled class conclusion
therefore differs from the conditional A-only conclusion; do not collapse them.

## Strict-past Text availability

History flags use all valid earlier utterances in the same conversation, including
neutral-label utterances. No current availability enters its own past flag.

| Current/history Text | N | Corrections | Harms | Positive net | Negative net | Total net |
|---|---:|---:|---:|---:|---:|---:|
| Current T observed | 19059 | 431 | 420 | -5 | +16 | +11 |
| Current T absent, past T observed | 7506 | 323 | 323 | +69 | -69 | 0 |
| Current T absent, past T never observed | 2507 | 88 | 103 | -39 | +24 | -15 |

Even the middle group's identical totalcorrect count hides opposite class changes.
It does not prove that historical Text has no information or that Memory was
unused. These are two trained models with different per-rate selected epochs,
not the causal effect of toggling Nested inside a single checkpoint. No new
mechanism, threshold, loss, experiment or queue alteration is authorized by
this offline breakdown.
