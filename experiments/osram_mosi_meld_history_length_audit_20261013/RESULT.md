# MOSI versus MELD history-length audit

**DATASET DIAGNOSTIC ONLY.** This report describes the exact dataset pickles
used by the current experiments. It does not by itself establish that history
length causes the observed Memory-performance difference.

## Sources

- MOSI: `dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl` in the biggpu OSRAM
  runtime checkout.
- MELD: the ECERC official
  `meld_emotion_semantic_features_roberta.pkl` recorded by the MELD runs.
- Splits are read directly from each pickle. No model inference or training is
  performed.

## Conversation length

| Dataset | Split | Dialogues | Utterances | Mean length | Median | P25 | P75 | P90 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | train | 52 | 1,284 | 24.692 | 24.0 | 16.75 | 30.25 | 33.9 | 63 |
| MOSI | validation | 10 | 229 | 22.900 | 23.5 | 16.50 | 29.50 | 34.0 | 34 |
| MOSI | test | 31 | 686 | **22.129** | **21.0** | 16.00 | 28.00 | 31.0 | 43 |
| MELD | train | 1,038 | 9,989 | 9.623 | 9.0 | 5.00 | 14.00 | 18.0 | 24 |
| MELD | validation | 114 | 1,109 | 9.728 | 10.0 | 5.00 | 13.00 | 17.0 | 23 |
| MELD | test | 280 | 2,610 | **9.321** | **9.0** | 5.00 | 13.00 | 17.0 | 33 |

On test, a MOSI conversation is about `2.37x` longer on average. The average
number of strictly prior turns available to one evaluated utterance is 11.847
for MOSI and 5.904 for MELD (`2.01x`). First utterances, which have no causal
history at all, account for 4.52% of MOSI test utterances and 10.73% of MELD
test utterances.

## Speaker-conditioned history

| Dataset | Split | Speakers/dialogue | Same-speaker prior turns/utterance | Has prior same-speaker turn | Previous same-speaker distance | Adjacent same-speaker |
|---|---|---:|---:|---:|---:|---:|
| MOSI | test | 1.000 | **11.847** | **95.481%** | 1.000 | 100.000% |
| MELD | test | 2.664 | **2.321** | **71.418%** | 2.205 | 24.249% |

MOSI is a single-speaker sequence in this loader. MELD has multiple speakers;
its same-speaker history per utterance is about `5.10x` smaller than MOSI.

For descriptive continuity only:

- MOSI adjacent non-neutral utterances have the same binary polarity 65.837%
  of the time on test.
- MELD adjacent utterances have the same seven-class emotion label 41.931% of
  the time on test.
- Within MELD test, adjacent same-speaker pairs keep the same label 57.699% of
  the time, versus 36.884% across a speaker switch.

These label relations are not model inputs and the MOSI polarity statistic is
not directly comparable to MELD seven-class equality. They only describe why
raw conversation length may overstate the amount of task-relevant history.

## Interpretation

History length is a plausible contributor to the cross-dataset result:

1. MELD provides roughly half as many prior turns per target utterance.
2. It provides only about one fifth as many same-speaker prior turns.
3. More MELD utterances occur at the history-free first position.
4. Speaker switching makes the available history less homogeneous than MOSI.

However, these marginal dataset statistics do not prove causation. The direct
test is to stratify the frozen MELD `Full - Local` intervention by strictly
prior-turn count and same-speaker-prior count. If Memory gain rises reliably
with these counts, short/sparse history is supported as a mechanism; if it
does not, history length is not the main bottleneck.
